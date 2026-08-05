"""Smoke test for CAM-style deep-supervision trajectory gene loss.

Replicates `trajectory_gene_loss` from hmflow/app/flow/train.py (so it doesn't pull
in train.py's scanpy/anndata import chain) against a tiny real flat HFlowDenoiser,
checking that the losses are finite, at the right scale, and gradients flow.

Validates the design per review feedback:
  * Original single-step FM loss is left untouched (deep supervision is a *separate*
    additive term, not a rewrite of the FM objective).
  * No autograd.grad / cosine-velocity objective — intermediate states are decoded
    through the same gene head and supervised directly (deep supervision).
  * States are detached between steps (O(1) memory, no T-deep ODE graph).
"""
import argparse
import torch

from hmflow.model.denoiser import HFlowDenoiser as Denoiser, pcc_loss
from hmflow.model.hflow_config import HFlowConfig
from hmflow.flow.interpolant import Interpolant


def trajectory_gene_loss(args, diffusier, model, img_features, coords, gene_exp):
    """Mirror of hmflow/app/flow/train.py:trajectory_gene_loss."""
    B = gene_exp.shape[0]
    device = gene_exp.device
    T = args.n_traj_steps
    shape = gene_exp.shape
    pad_mask = img_features.sum(-1) == 0
    gt = gene_exp[~pad_mask]

    t_min = 1e-3
    t_max = 1.0 - t_min
    ts = torch.linspace(t_max, t_min, T + 1, device=device)
    dts = ts[:-1] - ts[1:]

    x = diffusier.sample_from_prior(shape, device)
    gene_losses = []

    for s in range(T):
        t_s = ts[s].expand(B)
        pred = model.inference(x, img_features, coords, t_s)[0]
        d_t = dts[s].expand(B)
        y_hat = pred[~pad_mask]
        gl = (pcc_loss(y_hat, gt) if args.gene_loss == "pearson"
              else torch.nn.functional.mse_loss(y_hat, gt))
        gene_losses.append(gl)
        x = diffusier.denoise(pred, x, t_s, d_t).detach()

    gene_losses = torch.stack(gene_losses)
    weights = getattr(args, "traj_step_weights", None)
    if weights is not None:
        w = torch.as_tensor(weights, dtype=gene_losses.dtype, device=device)
        w = w / w.sum()
        traj_gene_loss = (gene_losses * w).sum()
    else:
        traj_gene_loss = gene_losses.mean()
    return traj_gene_loss, gene_losses


def main():
    torch.manual_seed(0)
    args = argparse.Namespace(
        n_traj_steps=5, traj_lambda=0.1, gene_loss="pearson",
        traj_step_weights=None,
        # realistic dims matching train.py defaults (MLPAttnEdge hardcodes n_genes=50).
        d_model=128, feature_dim=1024, n_genes=50, n_neighbors=8, n_layers=2,
        n_heads=4, dropout=0.0, attn_dropout=0.0, activation="gelu",
        d_edge_model=128, act="gelu",
    )
    model = Denoiser(args, hflow_config=HFlowConfig(
        n_region_queries=4, region_hidden_dim=128, slide_hidden_dim=128,
        hflow_representation="flat", hflow_cross_scale="bidirectional",
        hflow_region_discovery="learnable", hflow_assignment_temperature=1.0,
        hflow_assignment_entropy_weight=0.0, use_time_hierarchy_gate=False,
        modulation_mode="static", gate_mode="static", gate_hidden=128,
    ))
    model.train()

    diffusier = Interpolant("gaussian", device="cpu", normalize=False)

    B, N, D, G = 2, 6, 1024, 50
    img_features = torch.randn(B, N, D) * 0.5
    coords = torch.rand(B, N, 2) * 10
    gene_exp = torch.rand(B, N, G)

    traj_gene_loss, per_step = trajectory_gene_loss(
        args, diffusier, model, img_features, coords, gene_exp
    )

    print(f"per-step gene losses: {[round(v.item(), 4) for v in per_step]}")
    print(f"mean traj_gene_loss={traj_gene_loss.item():.4f}")
    assert torch.isfinite(traj_gene_loss).item(), "traj_gene_loss is not finite"
    # Pearson loss is bounded (pcc in [-1,1], negated) -> |gl| <= 1
    for v in per_step:
        assert torch.isfinite(v).item(), "a step gene loss is not finite"
        assert abs(v.item()) <= 1.0 + 1e-6, f"step gene loss {v.item()} out of bound"

    # Sanity: ground-truth prediction should give near -1 per step (cancels, sum small)
    # We just check a perfect-correlation input to pcc_loss:
    a = torch.randn(10, 5)
    print(f"pcc_loss(a,a)={pcc_loss(a, a.clone()).item():.4f} (expect ~ -1.0)")
    assert abs(pcc_loss(a, a.clone()).item() + 1.0) < 1e-4, "pcc_loss sanity failed"

    # Gradient flow: weight the combined loss (fm-style dummy 0 + lambda*traj) backward.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    combined = args.traj_lambda * traj_gene_loss
    combined.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"params with grad: {n_grad} / {len(list(model.parameters()))}")
    assert n_grad > 0, "no gradients flowed to model params"

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
