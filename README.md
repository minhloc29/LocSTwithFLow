# Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching

This is our PyTorch implementation for the paper:

> Tinglin Huang, Tianyu Liu, Mehrtash Babadi, Wengong Jin, and Rex Ying (2025). Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching. Paper in [arxiv](https://arxiv.org/pdf/2506.05361).

We recently extended STFlow and released **STPath**, a generative pretrained model capable of directly predicting the expression levels of 38,984 genes from histology images without further fine-tuning. Feel free to check out the [paper](https://www.biorxiv.org/content/10.1101/2025.04.19.649665v2.abstract) and [code](https://github.com/Graph-and-Geometric-Learning/STPath), which includes an easy-to-use API.

## Organization

The organization of this repository is as follows (all code lives under the `hmflow/` package):
- `hmflow/app/`: contains the training pipelines for pathology foundation models and flow matching
    - `hest/`: training pipeline for pathology foundation models, mainly from HEST pipeline
    - `flow/`: training pipeline for flow matching (including `train.py` and `test.py`)
- `hmflow/data/`: contains the dataloader for the flow matching model
- `hmflow/model/`: contains the implementation of the denoiser, including the hierarchical / cross-scale (HFlow) blocks
- `hmflow/flow/`: contains the interpolant and noise schedules for flow matching
- `hmflow/hest_utils/`: contains the utility functions for pathology foundation models, mainly from HEST pipeline
- `hmflow/utils/`: general utilities (random seeding, metrics, result merging)
- `corruption.py`: reference implementation of local artefact corruption utilities (zero / gaussian / dropout / blur)
- `local_corruption.py`: robustness evaluation script benchmarking model performance under local morphological corruption
- `scripts/`: diagnostic and plotting utilities (e.g. retention plots, W2 / OT sanity checks)
- `results/`: evaluation outputs (corruption, mosaic and STFlow baselines)


## Usage

Please run install the package by running:

```
$ pip install -e .
```

Download HEST benchmark datasets and pretrained weights of UNI and GigaPath models using the following script:
```
from huggingface_hub import snapshot_download, hf_hub_download

source_dataroot = ""/home/username/STFlow/dataset/"
weights_root = "/home/username/STFlow/dataset/weights_root"

snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=weights_root, allow_patterns=['fm_v1/*'])
snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=source_dataroot, ignore_patterns=['fm_v1/*'])
hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=os.path.join(weights_root, "uni/"))
hf_hub_download("prov-gigapath/prov-gigapath", filename="pytorch_model.bin", local_dir=os.path.join(weights_root, "gigapath/"))
```

Testing foundation models with the following script, which will save the extracted features in the `embed_dataroot`:
```
$ python hmflow/app/hest/benchmark.py \
        --datasets all \
        --encoders uni_v1_official \
        --weights_root /path/to/weights_root \
        --source_dataroot /path/to/source_dataroot \
        --embed_dataroot /path/to/embed_dataroot \
        --batch_size 128
```

Training the flow matching model with the following script:
```
$ python hmflow/app/flow/train.py \
        --datasets all \
        --feature_encoder uni_v1_official \
        --source_dataroot /path/to/source_dataroot \
        --embed_dataroot /path/to/embed_dataroot \
        --save_dir results_dir \
        --batch_size 2 \
        --n_layers 4 \
        --n_sample_steps 5
```

Optionally control the hierarchical (HFlow) representation and cross-scale interactions during training:
```
        --hflow_representation slide_region_patch \
        --hflow_cross_scale bidirectional \
        --hflow_region_discovery learnable \
        --n_region_queries 32
```

Evaluating model robustness under local morphological corruption (zero / gaussian / dropout / blur artefacts) across datasets and splits:
```
$ python local_corruption.py \
        --checkpoint_root /path/to/save_dir \
        --datasets all \
        --source_dataroot /path/to/source_dataroot \
        --embed_dataroot /path/to/embed_dataroot \
        --representation slide_region_patch \
        --corruption_types gaussian \
        --n_corrupt_seeds 3 \
        --save_dir_root results/corruption_eval
```

## Reference

If you find our work useful in your research, please consider citing our paper:

```
@inproceedings{huang2025stflow,
  title={Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching},
  author={Huang, Tinglin and Liu, Tianyu and Babadi, Mehrtash and Jin, Wengong and Ying, Rex},
  booktitle={International Conference on Machine Learning},
  year={2025}
}

@article{huang2025stpath,
  title={STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics and Whole Slide Images},
  author={Huang, Tinglin and Liu, Tianyu and Babadi, Mehrtash and Ying, Rex and Jin, Wengong},
  journal={bioRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
