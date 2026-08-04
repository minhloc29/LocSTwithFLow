import os
import json
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from operator import itemgetter
import torch
try:
    import wandb
except ImportError:
    wandb = None
from hmflow.utils import set_random_seed, get_current_time, merge_fold_results
from hmflow.data.dataset import HESTDatasetPath, MultiHESTDataset, padding_batcher, HESTDataset
from hmflow.data.normalize_utils import get_normalize_method
from hmflow.model.denoiser import HFlowDenoiser as Denoiser
from hmflow.model.hflow_config import HFlowConfig
from hmflow.flow.interpolant import Interpolant
from hmflow.app.flow.test import test
from hmflow.hest_utils.utils import save_pkl


def main(args, split_id, train_sample_ids, test_sample_ids, val_save_dir, checkpoint_save_dir):
    if getattr(args, 'model', 'spatial_flow') == 'triplex':
        from hmflow.app.flow.train_triplex import main as triplex_main
        return triplex_main(args, split_id, train_sample_ids, test_sample_ids, val_save_dir, checkpoint_save_dir)

    normalize_method = get_normalize_method(args.normalize_method)

    print("Dataset Loading")
    sample_id_paths = [
        HESTDatasetPath(
            name=sample_id,
            h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
            gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list),
        ) for sample_id in train_sample_ids
    ]
    train_dataset = MultiHESTDataset(sample_id_paths, 
                                     distribution=args.patch_distribution, 
                                     normalize_method=normalize_method,
                                     sample_times=args.sample_times)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=padding_batcher())

    # using the same train sample ids for validation
    sample_id_paths = [
        HESTDatasetPath(
                name=sample_id,
                h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
                h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
                gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list
            ),
        ) for sample_id in test_sample_ids
    ]
    val_loaders = [
        torch.utils.data.DataLoader(
            HESTDataset(
                sample_id_path, distribution="constant_1.0", 
                normalize_method=normalize_method,
                sample_times=1
            ),
            batch_size=1, collate_fn=padding_batcher()
        ) for sample_id_path in sample_id_paths
    ]

    device = args.device if torch.cuda.is_available() else "cpu"

    # Build HFlow config from args
    hflow_config = HFlowConfig(
        n_region_queries=args.n_region_queries,
        region_hidden_dim=args.hidden_dim,
        slide_hidden_dim=args.hidden_dim,
        hflow_representation=args.hflow_representation,
        hflow_cross_scale=args.hflow_cross_scale,
        hflow_region_discovery=args.hflow_region_discovery,
        hflow_assignment_temperature=args.hflow_assignment_temperature,
        hflow_assignment_entropy_weight=args.hflow_assignment_entropy_weight,
        use_time_hierarchy_gate=args.use_time_hierarchy_gate,
        modulation_mode=args.modulation_mode,
        gate_mode=args.gate_mode,
        gate_hidden=args.gate_hidden,
    )
    model = Denoiser(args, hflow_config=hflow_config).to(device)

    # ── Complexity accounting for the hierarchical modulation ──
    if hasattr(model, "blocks"):
        mod_params = sum(
            p.numel() for b in model.blocks for n, p in b.named_parameters()
            if "hierarchy_adaln" in n or "hierarchy_gate" in n
        )
        ln_params = sum(
            p.numel() for b in model.blocks for n, p in b.named_parameters()
            if "_adaln_norm" in n
        )
        extra = mod_params + ln_params
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[*] Modulation extra params: {extra} (+{ln_params} LN) "
              f"({100.0 * extra / total_params:.4f}% of total)" if total_params else "")

        # Extra FLOPs from the modulation MLPs (2 * out * in MACs, per block per step).
        extra_flops = 0
        for b in model.blocks:
            for name, g in (("hierarchy_adaln", getattr(b, "hierarchy_adaln", None)),
                            ("hierarchy_gate", getattr(b, "hierarchy_gate", None))):
                if g is None:
                    continue
                for m in g.modules():
                    if isinstance(m, torch.nn.Linear):
                        extra_flops += 2 * m.out_features * m.in_features
        if extra_flops:
            print(f"[*] Modulation extra FLOPs (per block, per timestep): {extra_flops:,}")

    diffusier = Interpolant(
        args.prior_sampler,
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Training")
    best_pearson, best_val_dict = -1, None
    early_stop_step = 0
    epoch_iter = tqdm(range(1, args.epochs + 1), ncols=100)
    for epoch in epoch_iter:
        avg_loss = 0
        model.train()

        for step, batch in enumerate(train_loader):
            batch = [x.to(device) for x in batch]
            img_features, coords, gene_exp = batch

            #STEP 1:
            noisy_exp, t_steps = diffusier.corrupt_exp(gene_exp)
            # noisy_exp: xt = (1-t)x0 + x1
            # t_steps: t
            
            # STEP 2:
            pred_exp, loss = model(
                exp=noisy_exp, 
                img_features=img_features, 
                coords=coords, 
                labels=gene_exp, 
                t_steps=t_steps
            )

            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            if args.use_wandb and wandb is not None:
                wandb.log({f"{args.dataset}/Train/{split_id}/loss": loss.cpu().item()})

            avg_loss += loss.cpu().item()
        
        avg_loss /= len(train_loader)
        epoch_iter.set_description(f"epoch: {epoch}, avg_loss: {avg_loss:.3f}")

        if args.save_step > 0 and epoch % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"{epoch}.pth"))

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            val_perf_dict, pred_dump = test(args, diffusier, model, val_loaders, return_all=True)
            if val_perf_dict["all"]['pearson_mean'] > best_pearson:
                best_pearson = val_perf_dict["all"]['pearson_mean']
                best_val_dict = val_perf_dict
                for patch_name, dataset_res in val_perf_dict.items():
                    with open(os.path.join(val_save_dir, f'{patch_name}_results.json'), 'w') as f:
                        json.dump(dataset_res, f, sort_keys=True, indent=4)
                
                # save_pkl(os.path.join(val_save_dir, 'inference_dump.pkl'), pred_dump)
                early_stop_step = 0

            else:
                early_stop_step += 1
                if early_stop_step >= 100:
                    print("Early stopping")
                    break

            if args.use_wandb and wandb is not None:
                for patch_name, dataset_res in val_perf_dict.items():
                    wandb.log({
                        f"{args.dataset}/Val/{split_id}/{patch_name}/pearson_mean": dataset_res['pearson_mean'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/pearson_std": dataset_res['pearson_std'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/l2_error_q1": dataset_res['l2_error_q1'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/l2_error_q2": dataset_res['l2_error_q2'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/l2_error_q3": dataset_res['l2_error_q3'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/r2_score_q1": dataset_res['r2_score_q1'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/r2_score_q2": dataset_res['r2_score_q2'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/r2_score_q3": dataset_res['r2_score_q3'],
                    })

    return best_val_dict["all"]


def run(args):
    # get train/test splits
    split_dir = os.path.join(args.source_dataroot, args.dataset, 'splits')
    splits = os.listdir(split_dir)
    all_split_results = []
    
    for i in range(len(splits) // 2):  # each split has a train and test file so we divide by 2
        print(f"Running dataset {args.dataset} split {i}")

        train_df = pd.read_csv(os.path.join(split_dir, f'train_{i}.csv'))
        test_df = pd.read_csv(os.path.join(split_dir, f'test_{i}.csv'))

        train_sample_ids = train_df['sample_id'].tolist()
        test_sample_ids = test_df['sample_id'].tolist()

        kfold_save_dir = os.path.join(args.save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)
        checkpoint_save_dir = os.path.join(kfold_save_dir, 'checkpoints')
        os.makedirs(checkpoint_save_dir, exist_ok=True)

        results = main(args, i, train_sample_ids, test_sample_ids, kfold_save_dir, checkpoint_save_dir)
        all_split_results.append(results)

    kfold_results = merge_fold_results(all_split_results)
    with open(os.path.join(args.save_dir, f'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--datasets', nargs='+', default=["all"], help="LUNG, READ, HCC, COAD")
    parser.add_argument('--use_wandb', default=True)
    parser.add_argument('--source_dataroot', default="/home/username/Anonymous_hmflow/dataset/")
    parser.add_argument('--embed_dataroot', type=str, default="/home/username/Anonymous_hmflow/dataset/embed_dataroot")
    parser.add_argument('--gene_list', type=str, default='var_50genes.json')
    parser.add_argument('--save_dir', type=str, default="results_dir/")
    parser.add_argument('--feature_encoder', type=str, default='uni_v1_official', help="uni_v1_official | resnet50_trunc | ciga | gigapath")
    parser.add_argument('--normalize_method', type=str, default="log1p")
    parser.add_argument('--exp_code', type=str, default="test")
    
    # training hyperparameters
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--sample_times', type=int, default=10, help='Number of times to sample patches from each image')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--clip_norm', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataloader')
    parser.add_argument('--loss_func', type=str, default='mse', help="mse | mae | pearson")
    parser.add_argument('--patch_distribution', type=str, default='uniform')
    parser.add_argument('--n_genes', type=int, default=50)

    # flow matching hyperparameters
    parser.add_argument('--n_sample_steps', type=int, default=3)
    parser.add_argument('--prior_sampler', type=str, default="zinb", help="gaussian | uniform | zero | zinb")
    parser.add_argument('--zinb_logits', type=float, default=0.1)
    parser.add_argument('--zinb_total_count', type=float, default=1)
    parser.add_argument('--zinb_zi_logits', type=float, default=0., help="Prob for zero inflation")  # before sigmoid

    # model hyperparameters
    parser.add_argument('--backbone', type=str, default="spatial_transformer")
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--pairwise_hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--n_neighbors', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--feature_dim', type=int, default=1024, help="uni:1024, ciga:512")
    parser.add_argument('--norm', type=str, default='layer', help="batch | layer")
    parser.add_argument('--activation', type=str, default='swiglu', help="relu | gelu | swiglu")

    # hierarchical flow matching hyperparameters (HFlow-ST)
    parser.add_argument('--hflow_representation', type=str, default='slide_region_patch',
                        help="flat | slide_patch | slide_region_patch")
    parser.add_argument('--hflow_cross_scale', type=str, default='bidirectional',
                        help="none | top_down | bottom_up | bidirectional")
    parser.add_argument('--hflow_region_discovery', type=str, default='learnable',
                        help="learnable | grid | kmeans | assignment")
    parser.add_argument('--n_region_queries', type=int, default=32,
                        help="Number of learnable region tokens")
    parser.add_argument('--hflow_assignment_temperature', type=float, default=1.0,
                        help="Temperature for dynamic region assignments")
    parser.add_argument('--hflow_assignment_entropy_weight', type=float, default=0.1,
                        help="Entropy regularization weight for dynamic assignments")

    # SSIM spatial-structure metrics (added to validation)
    parser.add_argument('--ssim', action='store_true', default=False,
                        help="Compute per-slide spatial SSIM (SSIM-All and SSIM-HVG) during validation")
    parser.add_argument('--ssim_grid', type=int, default=256,
                        help="Rasterization grid size for SSIM maps (e.g. 256)")
    parser.add_argument('--ssim_hvg', type=int, default=25,
                        help="Number of highly variable genes (by GT variance) for SSIM-HVG")

    # Time-dependent hierarchical fusion gate
    parser.add_argument('--use_time_hierarchy_gate', type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=True, help="Enable timestep-dependent Patch/Region/Slide modulation")
    parser.add_argument('--modulation_mode', type=str, default='adaln',
                        choices=['static', 'scalar', 'adaln'],
                        help="Modulation mode: 'static' (original), 'scalar' (softmax gate), "
                             "'adaln' (per-level scale/shift/gate, proposed)")
    parser.add_argument('--gate_mode', type=str, default='learnable',
                        choices=['static', 'fixed', 'learnable'],
                        help="Legacy scalar-gate mode (only used when modulation_mode='scalar')")
    parser.add_argument('--gate_hidden', type=int, default=128,
                        help="Hidden dim of the (legacy) scalar hierarchy gate MLP")

    # model selection: spatial_flow (default) or triplex (adapted TRIPLEX)
    parser.add_argument('--model', type=str, default='spatial_flow',
                        choices=['spatial_flow', 'triplex'],
                        help="Which model to train: 'spatial_flow' (HFlow/STFlow) or 'triplex'.")

    # TRIPLEX model hyperparameters (defaults mirror TRIPLEX config/ST/andersson/TRIPLEX.yaml)
    parser.add_argument('--triplex_emb_dim', type=int, default=512)
    parser.add_argument('--triplex_depth1', type=int, default=1)
    parser.add_argument('--triplex_depth2', type=int, default=3)
    parser.add_argument('--triplex_depth3', type=int, default=3)
    parser.add_argument('--triplex_num_heads1', type=int, default=4)
    parser.add_argument('--triplex_num_heads2', type=int, default=16)
    parser.add_argument('--triplex_num_heads3', type=int, default=16)
    parser.add_argument('--triplex_mlp_ratio1', type=float, default=4)
    parser.add_argument('--triplex_mlp_ratio2', type=float, default=4)
    parser.add_argument('--triplex_mlp_ratio3', type=float, default=1)
    parser.add_argument('--triplex_dropout1', type=float, default=0.2)
    parser.add_argument('--triplex_dropout2', type=float, default=0.1)
    parser.add_argument('--triplex_dropout3', type=float, default=0.3)
    parser.add_argument('--triplex_kernel_size', type=int, default=3)
    parser.add_argument('--triplex_pos_layer', type=str, default='APEG',
                        choices=['APEG', 'MLP', 'None', 'spatialformer'])
    parser.add_argument('--triplex_n_neighbors', type=int, default=25,
                        help="Number of spatial neighbours per spot (5x5 block = 25).")
    args = parser.parse_args()

    args.d_model = args.hidden_dim
    args.d_edge_model = args.pairwise_hidden_dim
    args.act = args.activation

    args.feature_dim = {
        "uni_v1_official": 1024,
        "gigapath": 1536,
        "ciga": 512,
    }[args.feature_encoder]

    set_random_seed(args.seed)

    if args.exp_code is None:
        exp_code = f"{args.backbone}_{get_current_time()}"
    else:
        exp_code = args.exp_code + f"_{args.feature_encoder}" + f"_{args.backbone}_{get_current_time()}"
    save_dir = os.path.join(args.save_dir, exp_code)
    os.makedirs(save_dir, exist_ok=True)
    
    if args.use_wandb and wandb is not None:
        wandb.init(project="spatial_transcriptomics", name=exp_code)
        wandb.config.update(args)

    print(f"Save dir: {save_dir}")
    print(args)

    if args.datasets[0] == "all":
        args.datasets = ["LUNG", "HCC", "COAD", "SKCM", "PAAD", "READ", "LYMPH_IDC", "PRAD", "IDC", "CCRCC"]

    for dataset in args.datasets:
        args.dataset = dataset
        args.save_dir = os.path.join(save_dir, dataset)
        os.makedirs(args.save_dir, exist_ok=True)

        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        run(args)

    if args.use_wandb and wandb is not None:
        wandb.finish()
