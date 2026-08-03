"""Synthetic end-to-end smoke test for the triplex training pipeline.

Fabricates a tiny HEST-bench-style dataset (embed h5, adata, splits, gene list),
then runs ``train_triplex.run`` for a couple of epochs and asserts STFlow-format
outputs (per-split ``*_results.json`` + aggregated ``results_kfold.json``) are
written with sane metric values.

Run:
    cd STFlow && python scripts/smoke_triplex.py
"""

import os
import json
import h5py
import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc

ROOT = "/tmp/triplex_smoke"
SRC = os.path.join(ROOT, "dataset")       # source_dataroot
EMB = os.path.join(ROOT, "embed")         # embed_dataroot
SAVE = os.path.join(ROOT, "results")      # save_dir
DATASET = "LUNG"
N_SPOTS = 30
N_GENES = 20
FEAT_DIM = 8
N_SAMPLES = 2  # slides


def make_data():
    adata_dir = os.path.join(SRC, DATASET, "adata")
    spl_dir = os.path.join(SRC, DATASET, "splits")
    emb_dir = os.path.join(EMB, DATASET, "uni_v1_official", "fp32")
    for d in (adata_dir, spl_dir, emb_dir):
        os.makedirs(d, exist_ok=True)

    # gene list
    genes = [f"G{i}" for i in range(N_GENES)]
    with open(os.path.join(SRC, DATASET, "var_50genes.json"), "w") as f:
        json.dump({"genes": genes}, f)

    sample_ids = []
    # shared latent per slide so embeddings correlate with expression -> finite pearson
    rng = np.random.default_rng(0)
    for s in range(N_SAMPLES):
        sid = f"slide{s}"
        sample_ids.append(sid)

        latent = rng.randn(N_SPOTS, 6)            # low-dim latent drives both
        W_expr = rng.randn(6, N_GENES)
        W_feat = rng.randn(6, FEAT_DIM)

        # adata (obs_names = barcodes so load_adata can index by barcode)
        barcodes = [f"{sid}_b{i}" for i in range(N_SPOTS)]
        expr = np.abs(latent @ W_expr) * 3.0 + 0.05
        X = scipy.sparse.csr_matrix(expr)
        ad = sc.AnnData(X=X,
                        obs=pd.DataFrame(index=barcodes),
                        var=pd.DataFrame(index=genes))
        ad.write_h5ad(os.path.join(adata_dir, f"{sid}.h5ad"))

        # embed h5: embeddings correlate with expression via shared latent
        embeddings = (latent @ W_feat + 0.1 * rng.randn(N_SPOTS, FEAT_DIM)).astype(np.float32)
        with h5py.File(os.path.join(emb_dir, f"{sid}.h5"), "w") as f:
            f.create_dataset("embeddings", data=embeddings)
            f.create_dataset("coords", data=rng.randn(N_SPOTS, 2).astype(np.float32))
            f.create_dataset("barcodes", data=np.array(barcodes).astype("S"))

    # splits: 1 train, 1 test
    train0 = sample_ids[:1]
    test0 = sample_ids[1:]
    pd.DataFrame({"sample_id": train0}).to_csv(os.path.join(spl_dir, "train_0.csv"), index=False)
    pd.DataFrame({"sample_id": test0}).to_csv(os.path.join(spl_dir, "test_0.csv"), index=False)


class Args:
    pass


def main():
    make_data()
    from hmflow.app.flow import train_triplex as TT

    a = Args()
    a.source_dataroot = SRC
    a.embed_dataroot = EMB
    a.save_dir = SAVE
    a.dataset = DATASET
    a.datasets = [DATASET]
    a.feature_encoder = "uni_v1_official"
    a.gene_list = "var_50genes.json"
    a.normalize_method = "log1p"
    a.patch_distribution = "constant_1.0"
    a.sample_times = 1
    a.batch_size = 1
    a.device = 0
    a.lr = 1e-2
    a.epochs = 30
    a.eval_step = 5
    a.save_step = 1
    a.clip_norm = 1.0
    a.use_wandb = False

    # model args (small)
    a.feature_dim = FEAT_DIM
    a.n_genes = N_GENES
    a.triplex_emb_dim = 16
    a.triplex_depth1 = 1
    a.triplex_depth2 = 2
    a.triplex_depth3 = 2
    a.triplex_num_heads1 = 2
    a.triplex_num_heads2 = 4
    a.triplex_num_heads3 = 4
    a.triplex_mlp_ratio1 = 2
    a.triplex_mlp_ratio2 = 2
    a.triplex_mlp_ratio3 = 2
    a.triplex_dropout1 = 0.0
    a.triplex_dropout2 = 0.0
    a.triplex_dropout3 = 0.0
    a.triplex_kernel_size = 3
    a.triplex_pos_layer = "APEG"
    a.triplex_n_neighbors = 4

    TT.run(a)

    # Assert outputs
    kfold = os.path.join(SAVE, "results_kfold.json")
    assert os.path.isfile(kfold), "results_kfold.json missing"
    with open(kfold) as f:
        res = json.load(f)
    print("results_kfold.json keys:", list(res.keys()))
    print("pearson_mean:", res.get("pearson_mean"))
    assert "pearson_mean" in res and np.isfinite(res["pearson_mean"]), "no finite pearson_mean"
    assert "pearson_corrs" in res and len(res["pearson_corrs"]) == N_GENES

    split_dir = os.path.join(SAVE, "split0")
    all_res = os.path.join(split_dir, "all_results.json")
    assert os.path.isfile(all_res), "all_results.json missing"
    ckpts = os.listdir(os.path.join(split_dir, "checkpoints"))
    print("checkpoints:", ckpts)
    assert ckpts, "no checkpoints saved"

    print("\nSMOKE TEST PASSED: triplex train pipeline wrote STFlow-format outputs")


if __name__ == "__main__":
    main()
