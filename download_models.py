import gdown
import os

model_dir = "predictors"
os.makedirs(model_dir, exist_ok=True)

files = {
    "DR_predictor.h5": "1f4PContWp4N324gA9USqBRoIaqTgG4xI",
    "HTN_InceptionV3_regression.h5": "10UtQJlJwPMABGhoyKiJ3Ik4a1ixnbaau",
    "hba1c_xgboost_predictor.pkl": "1ZUlUIDKNyJUekyMioE4HvUBscmNSzPtW",
    "hba1c_scaler.pkl": "1YTURQDXWtXJDFX7NrBKlRjJsVF-rU4gO",
}

for filename, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    out_path = os.path.join(model_dir, filename)
    if not os.path.exists(out_path):
        print(f"Downloading {filename}...")
        gdown.download(url, out_path, quiet=False, use_cookies=False)

