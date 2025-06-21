import os

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import tensorflow as tf
import joblib
from PIL import Image
import io

app = FastAPI()

# CORS for both local and deployed frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8111",
        "https://ocumedai.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENV flag to skip model loading during initial deploy
SKIP_MODELS = os.getenv("SKIP_MODELS", "false").lower() == "true"

# Minimal test route
@app.get("/")
def home():
    return {"message": "OcuMedAI FastAPI is online"}

# Skip all model logic if flag is set
if SKIP_MODELS:
    print("🟡 Skipping model loading due to SKIP_MODELS=True")
else:
    # Ensure models are downloaded
    if not os.path.exists("predictors/DR_predictor.h5"):
        import download_models

    # Load models
    model_dr = tf.keras.models.load_model("predictors/DR_predictor.h5")
    model_htn = tf.keras.models.load_model("predictors/HTN_InceptionV3_regression.h5")
    model_hba1c = joblib.load("predictors/hba1c_xgboost_predictor.pkl")
    scaler = joblib.load("predictors/hba1c_scaler.pkl")

    # Image Preprocessing
    def preprocess_image(image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        return np.expand_dims(img_array, axis=0)

    # Prediction Route
    @app.post("/predict")
    async def predict(
        image: UploadFile = File(...),
        age: int = Form(...),
        sex: str = Form(...),
        BMI: float = Form(...),
        smokingStatus: str = Form(...)
    ):
        img_data = await image.read()
        input_img = preprocess_image(img_data)

        # DR Prediction
        dr_pred = model_dr.predict(input_img)
        dr_level = np.argmax(dr_pred)
        dr_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']
        dr_result = dr_labels[dr_level]

        # HTN Prediction
        htn_pred = model_htn.predict(input_img)
        htn_percent = float(np.clip(htn_pred[0][0] * 100, 0, 100))
        htn_binary = 1 if htn_percent >= 50 else 0

        # HbA1c Prediction
        sex_val = 1 if sex == "Male" else 0
        smoke_val = 1 if smokingStatus == "Yes" else 0
        diabetes_est = 1 if dr_level > 0 or htn_percent > 60 else 0

        features = np.array([[age, sex_val, BMI, htn_binary, diabetes_est, smoke_val]])
        features_scaled = scaler.transform(features)
        hba1c = float(model_hba1c.predict(features_scaled)[0])

        # Atherosclerosis Risk
        htn_scaled = htn_percent / 100
        dr_scaled = dr_level / 4
        hba1c_scaled = hba1c / 10
        athero_risk = (
            0.4 * htn_scaled +
            0.3 * dr_scaled +
            0.4 * hba1c_scaled
        ) * 100

        return {
            "diabeticRetinopathyLevel": dr_result,
            "hypertensionRisk": round(htn_percent, 2),
            "hba1cLevel": round(hba1c, 2),
            "atherosclerosisRisk": round(athero_risk, 2)
        }
