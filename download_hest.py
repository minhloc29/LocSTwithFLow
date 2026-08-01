# # # from huggingface_hub import snapshot_download

# # # DATA_ROOT = r"C:\Users\BKAI\locodo\LocSTwithFlow"

# # # # Download everything EXCEPT foundation model weights (features, adata, splits)
# # # snapshot_download(
# # #     repo_id="MahmoodLab/hest-bench", 
# # #     repo_type='dataset', 
# # #     local_dir=DATA_ROOT, 
# # #     ignore_patterns=['fm_v1/*']
# # # )



# # # python stflow/app/hest/benchmark.py --datasets all --encoders uni_v1_official --weights_root /hihi --source_dataroot hihi --embed_dataroot hihi --batch_size 128
    
    

# # # $env:PYTHONPATH += ";C:\Users\BKAI\locodo\LocSTwithFLow"

# # # python stflow/app/hest/benchmark.py --datasets all --encoders uni_v1_official --weights_root dataset/weights_root --source_dataroot dataset --embed_dataroot dataset --batch_size 128


# # # python stflow/app/flow/train.py --datasets LUNG --feature_encoder uni_v1_official --source_dataroot dataset --embed_dataroot dataset --batch_size 2 --n_layers 4 --n_sample_steps 5

# # from huggingface_hub import hf_hub_download
# # hf_hub_download('MahmoodLab/UNI', filename='pytorch_model.bin', 
# #                 local_dir='/weights/uni/')



# # python stflow/app/hest/benchmark.py --datasets LUNGS --feature_encoder uni_v1_official --weights_root 'C:/Users/BKAI/locodo/LocSTwithFLow/stflow/weights/weights' --batch_size 128

# import os

# from huggingface_hub import snapshot_download, hf_hub_download

# source_dataroot = "dataset"
# weights_root = "dataset/weights_root"

# snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=weights_root, allow_patterns=['fm_v1/*'])
# snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=source_dataroot, ignore_patterns=['fm_v1/*'])
# hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=os.path.join(weights_root, "uni/"))
# hf_hub_download("prov-gigapath/prov-gigapath", filename="pytorch_model.bin", local_dir=os.path.join(weights_root, "gigapath/"))



# python stflow/app/flow/train.py --datasets LUNG --source_dataroot dataset --embed_dataroot dataset --batch_size 2 --n_layers 4 --n_sample_steps 5         
                                  
                                  
# python hmflow/app/hest/benchmark.py --datasets LYMPH_IDC --encoders uni_v1_official --weights_root dataset/weights_root --source_dataroot dataset --embed_dataroot dataset --batch_size 128


from pathlib import Path
import scanpy as sc

root = Path(r"dataset\COAD")

for f in root.rglob("*.h5ad"):
    adata = sc.read_h5ad(f, backed="r")
    print(f.stem, adata.n_obs)