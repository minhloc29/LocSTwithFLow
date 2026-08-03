"""Training / evaluation loop for the TRIPLEX model inside STFlow's pipeline.

Provides the same k-fold structure, split handling, checkpointing and result
aggregation as ``hmflow/app/flow/train.py``, but trains the adapted
``TriplexModel`` (supervised spatial gene-expression regression) instead of the
flow-matching denoiser. Outputs are written in STFlow's format
(``*_results.json`` per split + aggregated ``results_kfold.json``).
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from operator import itemgetter

try:
    import wandb
except ImportError:
    wandb = None

from hmflow.utils import merge_fold_results
from hmflow.data.dataset import HESTDatasetPath, HESTDataset, MultiHESTDataset, padding_batcher
from hmflow.data.normalize_utils import get_normalize_method
from hmflow.model.triplex.triplex import TriplexModel
from hmflow.app.flow.test import metric_func


def _build_datasets(args, train_sample_ids, test_sample_ids, normalize_method):
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               collate_fn=padding_batcher())

    sample_id_paths = [
        HESTDatasetPath(
            name=sample_id,
            h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
            gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list),
        ) for sample_id in test_sample_ids
    ]
    val_loaders = [
        torch.utils.data.DataLoader(
            HESTDataset(sample_id_path, distribution="constant_1.0",
                        normalize_method=normalize_method, sample_times=1),
            batch_size=1, collate_fn=padding_batcher()
        ) for sample_id_path in sample_id_paths
    ]
    return train_loader, val_loaders


@torch.no_grad()
def _evaluate(args, model, val_loaders):
    """Run full-slide inference per validation loader and return a metric dict."""
    device = args.device if torch.cuda.is_available() else "cpu"
    model.eval()
    res_dict = {}
    all_pred, all_gt = [], []
    for loader in val_loaders:
        cur_pred, cur_gt = [], []
        for batch in loader:
            batch = [x.to(device) for x in batch]
            img_features, coords, labels = batch
            pred = model.inference(img_features, coords)  # [1, N, G]
            cur_pred.append(pred.squeeze(0).cpu().numpy())
            cur_gt.append(labels.squeeze(0).cpu().numpy())
        cur_pred = np.concatenate(cur_pred, axis=0)
        cur_gt = np.concatenate(cur_gt, axis=0)
        cur_res = metric_func(cur_pred, cur_gt, loader.dataset.gene_list)
        cur_res.update({'n_test': len(cur_gt)})
        res_dict[loader.dataset.name] = cur_res
        all_pred.append(cur_pred)
        all_gt.append(cur_gt)

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    cur_res = metric_func(all_pred, all_gt, val_loaders[0].dataset.gene_list)
    cur_res.update({'n_test': len(all_gt)})
    res_dict["all"] = cur_res
    return res_dict


def main(args, split_id, train_sample_ids, test_sample_ids, val_save_dir, checkpoint_save_dir):
    normalize_method = get_normalize_method(args.normalize_method)

    print("Dataset Loading")
    train_loader, val_loaders = _build_datasets(args, train_sample_ids, test_sample_ids, normalize_method)

    device = args.device if torch.cuda.is_available() else "cpu"

    model = TriplexModel(
        feature_dim=args.feature_dim,
        emb_dim=args.triplex_emb_dim,
        num_genes=args.n_genes,
        depth1=args.triplex_depth1,
        depth2=args.triplex_depth2,
        depth3=args.triplex_depth3,
        num_heads1=args.triplex_num_heads1,
        num_heads2=args.triplex_num_heads2,
        num_heads3=args.triplex_num_heads3,
        mlp_ratio1=args.triplex_mlp_ratio1,
        mlp_ratio2=args.triplex_mlp_ratio2,
        mlp_ratio3=args.triplex_mlp_ratio3,
        dropout1=args.triplex_dropout1,
        dropout2=args.triplex_dropout2,
        dropout3=args.triplex_dropout3,
        kernel_size=args.triplex_kernel_size,
        pos_layer=args.triplex_pos_layer,
        n_neighbors=args.triplex_n_neighbors,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Training")
    best_pearson, best_val_dict = -1, None
    early_stop_step = 0
    epoch_iter = tqdm(range(1, args.epochs + 1), ncols=100)
    for epoch in epoch_iter:
        avg_loss = 0
        model.train()

        for batch in train_loader:
            batch = [x.to(device) for x in batch]
            img_features, coords, gene_exp = batch

            out = model(img_features, coords, gene_exp)
            loss = out['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            if args.use_wandb and wandb is not None:
                wandb.log({f"{args.dataset}/Triplex/Train/{split_id}/loss": loss.cpu().item()})

            avg_loss += loss.cpu().item()

        avg_loss /= len(train_loader)
        epoch_iter.set_description(f"epoch: {epoch}, avg_loss: {avg_loss:.3f}")

        if args.save_step > 0 and epoch % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"{epoch}.pth"))

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            val_perf_dict = _evaluate(args, model, val_loaders)
            curr = val_perf_dict["all"]['pearson_mean']
            if curr is None or not np.isfinite(curr):
                curr = float('-inf')
            if curr > best_pearson:
                best_pearson = val_perf_dict["all"]['pearson_mean']
                best_val_dict = val_perf_dict
                for patch_name, dataset_res in val_perf_dict.items():
                    with open(os.path.join(val_save_dir, f'{patch_name}_results.json'), 'w') as f:
                        json.dump(dataset_res, f, sort_keys=True, indent=4)
                early_stop_step = 0
            else:
                early_stop_step += 1
                if early_stop_step >= 100:
                    print("Early stopping")
                    break

            if args.use_wandb and wandb is not None:
                for patch_name, dataset_res in val_perf_dict.items():
                    wandb.log({
                        f"{args.dataset}/Triplex/Val/{split_id}/{patch_name}/pearson_mean": dataset_res['pearson_mean'],
                        f"{args.dataset}/Triplex/Val/{split_id}/{patch_name}/pearson_std": dataset_res['pearson_std'],
                        f"{args.dataset}/Triplex/Val/{split_id}/{patch_name}/l2_error_q1": dataset_res['l2_error_q1'],
                        f"{args.dataset}/Triplex/Val/{split_id}/{patch_name}/l2_error_q2": dataset_res['l2_error_q2'],
                        f"{args.dataset}/Triplex/Val/{split_id}/{patch_name}/l2_error_q3": dataset_res['l2_error_q3'],
                        f"{args.dataset}/Triplex/Val/{split_id}/{patch_name}/r2_score_q1": dataset_res['r2_score_q1'],
                        f"{args.dataset}/Triplex/Val/{split_id}/{patch_name}/r2_score_q2": dataset_res['r2_score_q2'],
                        f"{args.dataset}/Triplex/Val/{split_id}/{patch_name}/r2_score_q3": dataset_res['r2_score_q3'],
                    })

    if best_val_dict is None:
        # All validation pearson values were NaN (e.g. degenerate inputs).
        # Write a placeholder so downstream k-fold aggregation does not crash.
        _evaluate(args, model, val_loaders)  # ensure directories exist (no-op metrics)
        for patch_name, dataset_res in [("all", {"pearson_mean": float('nan'),
                                                 "pearson_std": float('nan'), "n_test": 0})]:
            with open(os.path.join(val_save_dir, f'{patch_name}_results.json'), 'w') as f:
                json.dump(dataset_res, f, sort_keys=True, indent=4)
        return {"pearson_mean": float('nan'), "pearson_std": float('nan'),
                "pearson_corrs": [], "mean_per_split": [float('nan')]}

    return best_val_dict["all"]


def run(args):
    """Run all k-fold splits for the current dataset and aggregate results."""
    import pandas as pd

    split_dir = os.path.join(args.source_dataroot, args.dataset, 'splits')
    splits = os.listdir(split_dir)
    all_split_results = []

    for i in range(len(splits) // 2):
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
