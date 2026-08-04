import torch
import numpy as np
from scipy.stats import pearsonr


def spatial_ssim(gt_gene, pred_gene, coords, grid_size=256, method="linear"):
    """SSIM between the GT and predicted 2D expression maps for one gene.

    gt/pred_gene: [N] per-spot expression for the gene.
    coords:       [N, 2] spot coordinates in a single slide frame.
    Rasterizes the irregular spot values into a dense grid via griddata,
    then computes structural_similarity. Guards constant maps (range ~ 0).
    """
    if coords.shape[0] < 4:
        return float("nan")
    x = coords[:, 0]; y = coords[:, 1]
    # Normalize extrema to a stable pixel grid; fall back gracefully if degenerate.
    if (np.ptp(x) < 1e-9) or (np.ptp(y) < 1e-9):
        return float("nan")
    grid_x, grid_y = np.mgrid[
        x.min():x.max():grid_size * 1j,
        y.min():y.max():grid_size * 1j,
    ]
    from scipy.interpolate import griddata
    gt_img = griddata(coords, gt_gene, (grid_x, grid_y), method=method)
    pred_img = griddata(coords, pred_gene, (grid_x, grid_y), method=method)
    # Fill holes (NaNs outside the convex hull) with the local median / 0.
    gt_img = np.nan_to_num(gt_img, nan=np.nanmedian(gt_img) if np.isfinite(gt_img).any() else 0.0)
    pred_img = np.nan_to_num(pred_img, nan=0.0)
    data_range = float(gt_img.max() - gt_img.min())
    if data_range < 1e-9:   # constant GT map -> SSIM meaningless
        return float("nan")
    from skimage.metrics import structural_similarity
    return float(structural_similarity(gt_img, pred_img, data_range=data_range))


def ssim_metrics(gt, pred, coords, genes, hvg_n=25, grid_size=256):
    """Per-gene SSIM (SSIM-All) and SSIM over the top-K highly variable genes.

    gt/pred: [N, G]; coords: [N, 2]; genes: list[G].
    HVG subset is picked by GT variance, so the selection is model-independent.
    """
    n_genes = gt.shape[1]
    var = gt.var(axis=0)
    hv_idx = np.argsort(-var)[: min(hvg_n, n_genes)]
    ssim_g, ssim_hvg = [], []
    for g in range(n_genes):
        s = spatial_ssim(gt[:, g], pred[:, g], coords, grid_size=grid_size)
        ssim_g.append(s)
        if g in hv_idx:
            ssim_hvg.append(s)
    ssim_g = np.asarray(ssim_g, dtype=float)
    ssim_hvg = np.asarray(ssim_hvg, dtype=float)

    def stat(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.std())

    ssim_all_mean, ssim_all_std = stat(ssim_g)
    ssim_hvg_mean, ssim_hvg_std = stat(ssim_hvg)
    return {
        "ssim_all_mean": ssim_all_mean,
        "ssim_all_std": ssim_all_std,
        "ssim_hvg_mean": ssim_hvg_mean,
        "ssim_hvg_std": ssim_hvg_std,
        "n_hvg": int(len(hv_idx)),
    }


def metric_func(preds_all: np.ndarray, y_test: np.ndarray, genes: list):
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []

    n_nan_genes = 0
    for i, target in enumerate(range(y_test.shape[1])):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]

        errors.append(float(np.mean((preds - target_vals) ** 2)))
        r2_scores.append(float(1 - np.sum((target_vals - preds) ** 2) / np.sum((target_vals - np.mean(target_vals)) ** 2)))
        pearson_corr, _ = pearsonr(target_vals, preds)
        pearson_corr = float(pearson_corr)
        pearson_corrs.append(pearson_corr)

        if np.isnan(pearson_corr):
            n_nan_genes += 1

        pearson_genes.append({'name': genes[i], 'pearson_corr': pearson_corr})

    if n_nan_genes > 0:
        print(f"Warning: {n_nan_genes} genes have NaN Pearson correlation")

    return {
        'l2_errors': list(errors),
        'r2_scores': list(r2_scores),
        'pearson_corrs': pearson_genes,
        'pearson_mean': float(np.mean(pearson_corrs)),
        'pearson_std': float(np.std(pearson_corrs)),
        'l2_error_q1': float(np.percentile(errors, 25)),
        'l2_error_q2': float(np.median(errors)),
        'l2_error_q3': float(np.percentile(errors, 75)),
        'r2_score_q1': float(np.percentile(r2_scores, 25)),
        'r2_score_q2': float(np.median(r2_scores)),
        'r2_score_q3': float(np.percentile(r2_scores, 75)),
    }


@torch.no_grad()
def test(args, diffusier, model, loader_list, return_all=False):
    model.eval()
    all_pred, all_gt = [], []
    all_coords, all_slide_id, all_region_assign = [], [], []
    res_dict = {}
    capture_regions = hasattr(model, '_last_region_assignment') or True  # try regardless; guarded below

    # ---- Hierarchy gate logging (timestep-dependent weights) ----
    # The gate weights are identical across blocks (same t), so log block 0.
    gate_t, gate_a = [], []  # (t, (mean_patch, mean_region, mean_slide))
    block0 = model.blocks[0] if hasattr(model, "blocks") and len(model.blocks) else None
    has_gate = bool(getattr(block0, "hierarchy_gate", None) or getattr(block0, "gate_mode", "static") != "static")

    for loader in loader_list:
        cur_pred, cur_gt, cur_coords, cur_region = [], [], [], []

        for step, batch in enumerate(loader):
            batch = [x.to(args.device) for x in batch]
            img_features, coords, labels = batch
            assert img_features.shape[0] == 1, "Batch size must be 1 for inference"

            exp_t1 = diffusier.sample_from_prior(labels.shape, labels.device)
            ts = torch.linspace(
                0.01, 1.0, args.n_sample_steps
            )[:, None].expand(args.n_sample_steps, exp_t1.shape[0]).to(args.device)

            pred = None

            for step, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
                pred, hierarchy_state = model.inference(
                    exp_t1, img_features, coords,
                    t1
                )
                if has_gate and block0 is not None:
                    gw = getattr(block0, "_last_gate_weights", None)
                    gt = getattr(block0, "_last_gate_t", None)
                    if gw is not None and gt is not None:
                        gate_t.append(float(gt.squeeze().mean().item()))
                        gate_a.append((float(gw[0].mean().item()),
                                       float(gw[1].mean().item()),
                                       float(gw[2].mean().item())))
                d_t = t2 - t1

                if step == args.n_sample_steps - 2:
                    break
                else:
                    exp_t1 = diffusier.denoise(pred, exp_t1, t1, d_t)

            sample = pred
            cur_pred.append(sample.squeeze(0).cpu().numpy())
            cur_gt.append(labels.squeeze(0).cpu().numpy())
            cur_coords.append(coords.squeeze(0).cpu().numpy())

            # Region assignment from the FINAL Euler step's forward pass.
            # Shape as produced by DynamicRegionAssignment: [B=1, N, K] -> squeeze to [N, K]
            region_assign = getattr(model, '_last_region_assignment', None)
            if region_assign is not None:
                cur_region.append(region_assign.squeeze(0).cpu().numpy())
            else:
                cur_region.append(None)

        cur_pred = np.concatenate(cur_pred, axis=0)
        cur_gt = np.concatenate(cur_gt, axis=0)
        cur_coords_arr = np.concatenate(cur_coords, axis=0)
        cur_res_dict = metric_func(cur_pred, cur_gt, loader.dataset.gene_list)
        # SSIM is spatial and therefore per-slide: compute on this slide's frame.
        if getattr(args, "ssim", False):
            try:
                cur_res_dict.update(ssim_metrics(
                    cur_gt, cur_pred, cur_coords_arr,
                    loader.dataset.gene_list,
                    hvg_n=getattr(args, "ssim_hvg", 25),
                    grid_size=getattr(args, "ssim_grid", 256),
                ))
            except Exception as e:
                print(f"[!] SSIM failed for {loader.dataset.name}: {e}")
        cur_res_dict.update({'n_test': len(cur_gt)})
        res_dict[loader.dataset.name] = cur_res_dict

        all_pred.append(cur_pred)
        all_gt.append(cur_gt)
        all_coords.append(cur_coords_arr)
        all_slide_id.append(np.full(len(cur_gt), loader.dataset.name, dtype=object))

        if all(r is not None for r in cur_region):
            all_region_assign.append(np.concatenate(cur_region, axis=0))
        else:
            all_region_assign.append(None)

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    all_slide_id_flat = np.concatenate(all_slide_id, axis=0)
    all_coords_flat = np.concatenate(all_coords, axis=0)

    cur_res_dict = metric_func(all_pred, all_gt, loader_list[0].dataset.gene_list)
    cur_res_dict.update({'n_test': len(all_gt)})
    res_dict["all"] = cur_res_dict

    if return_all:
        dump = {
            'preds_all': all_pred,
            'targets_all': all_gt,
            'coords_all': all_coords_flat,
            'slide_id_all': all_slide_id_flat,
            'gene_list': loader_list[0].dataset.gene_list,
        }
        if all(r is not None for r in all_region_assign):
            dump['region_assignments_all'] = np.concatenate(all_region_assign, axis=0)
        else:
            dump['region_assignments_all'] = None
            print("Note: region_assignments not captured — apply patch_capture_diagnostics.py first "
                  "if you want the region-tissue correspondence analysis.")

        if gate_t:
            dump['gate_t'] = gate_t
            dump['gate_weights'] = gate_a   # list of (mean_patch, mean_region, mean_slide)
            # Save a standalone npz for plotting gate_curve without re-running inference.
            try:
                import os
                npz_path = os.path.join(getattr(args, 'val_save_dir', None) or '.', 'gate_log.npz')
                np.savez(
                    npz_path,
                    gate_t=np.asarray(gate_t, dtype=np.float32),
                    gate_weights=np.asarray(gate_a, dtype=np.float32),
                )
                print(f"[*] Saved hierarchy gate log -> {npz_path}")
            except Exception as e:  # plotting must never break validation
                print(f"[!] Failed to save gate log: {e}")
        return res_dict, dump

    return res_dict