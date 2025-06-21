import os
import requests
from zipfile import ZipFile

MODEL_URL = "https://huggingface.co/BhuvanyuWalia/ocumedai-models/resolve/main/predictors.zip"
ZIP_PATH = "predictors.zip"
EXTRACT_PATH = "predictors"

if not os.path.exists(EXTRACT_PATH):
    print("Downloading models from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)

    with ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("Models downloaded and extracted.")
