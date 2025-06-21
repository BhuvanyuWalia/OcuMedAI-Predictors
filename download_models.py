import os
from huggingface_hub import hf_hub_download

model_dir = "predictors"
os.makedirs(model_dir, exist_ok=True)

files = {
    "DR_predictor.h5": "DR_predictor.h5",
    "HTN_InceptionV3_regression.h5": "HTN_InceptionV3_regression.h5",
    "hba1c_xgboost_predictor.pkl": "hba1c_xgboost_predictor.pkl",
    "hba1c_scaler.pkl": "hba1c_scaler.pkl"
}

for local_name, repo_file in files.items():
    print(f"Downloading {local_name}...")
    output_path = os.path.join(model_dir, local_name)
    hf_hub_download(
        repo_id="bwalia5/ocumedai-models",
        filename=repo_file,
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
