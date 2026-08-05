"""Smoke test for the CAM-style adjoint_train_step (no dataset/data needed).

Builds a tiny flat HFlowDenoiser + Interpolant on CPU, feeds random tensors and
runs adjoint_train_step, checking the losses are finite and grads flow.
"""
import argparse
import torch

from hmflow.model.denoiser import HFlowDenoiser as Denoiser, pcc_loss
from hmflow.flow.interpolant import Interpolant
from hmflow.app.flow.train import adjoint_train_step


def main():
    torch.manual_seed(0)
    args = argparse.Namespace(
        n_traj_steps=5,
        traj_lambda=1.0,
        gene_loss="pearson",
        device=0,
        d_model=16,
        feature_dim=8,
        n_genes=6,
        n_neighbors=3,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
        attn_dropout=0.0,
        activation="gelu",
        pairwise_hidden_dim=8,
        n_region_queries=4,
        region_hidden_dim=16,
        slide_hidden_dim=16,
        hflow_representation="flat",
        hflow_cross_scale="bidirectional",
        hflow_region_discovery="learnable",
        hflow_assignment_temperature=1.0,
        hflow_assignment_entropy_weight=0.0,
        use_time_hierarchy_gate=False,
        modulation_mode="static",
        gate_mode="static",
        gate_hidden=16,
    )
    from hmflow.model.hflow_config import HFlowConfig

    model = Denoiser(args, hflow_config=HFlowConfig(
        n_region_queries=args.n_region_queries,
        region_hidden_dim=args.region_hidden_dim,
        slide_hidden_dim=args.slide_hidden_dim,
        hflow_representation="flat",
        hflow_cross_scale=args.hflow_cross_scale,
        hflow_region_discovery=args.hflow_region_discovery,
        hflow_assignment_temperature=args.hflow_assignment_temperature,
        hflow_assignment_entropy_weight=args.hflow_assignment_entropy_weight,
        use_time_hierarchy_gate=args.use_time_hierarchy_gate,
        modulation_mode=args.modulation_mode,
        gate_mode=args.gate_mode,
        gate_hidden=args.gate_hidden,
    ))
    model.train()

    diffusier = Interpolant("gaussian", device="cpu", normalize=False)

    B, N, D, G = 2, 6, 8, 6
    img_features = torch.randn(B, N, D) * 0.5
    coords = torch.rand(B, N, 2) * 10
    gene_exp = torch.rand(B, N, G)

    loss, fm_loss, traj_loss = adjoint_train_step(
        args, diffusier, model, img_features, coords, gene_exp
    )

    print(f"loss={loss.item():.4f}, fm_loss={fm_loss.item():.4f}, traj_loss={traj_loss.item():.4f}")
    assert torch.isfinite(loss).item(), "loss not finite"
    assert torch.isfinite(fm_loss).item(), "fm_loss not finite"
    assert torch.isfinite(traj_loss).item(), "traj_loss not finite"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    n_grad = len(grads)
    print(f"params with grad: {n_grad} / {len(list(model.parameters()))}")
    assert n_grad > 0, "no gradients flowed to model params"

    # sanity of pcc_loss: perfect correlation should give ~ -1
    a = torch.randn(10, 5)
    b = a.clone()
    print(f"pcc_loss(a,a)={pcc_loss(a, b).item():.4f} (expect ~ -1.0)")
    assert abs(pcc_loss(a, b).item() + 1.0) < 1e-4

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
