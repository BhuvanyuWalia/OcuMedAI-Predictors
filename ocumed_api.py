# ocumed_api.py
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

# Allow access from your Node.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8111"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
model_dr = tf.keras.models.load_model("predictors/DR_predictor.h5")
model_htn = tf.keras.models.load_model("predictors/HTN_InceptionV3_regression.h5")
model_hba1c = joblib.load("predictors/hba1c_xgboost_predictor.pkl")
scaler = joblib.load("predictors/hba1c_scaler.pkl")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    BMI: float = Form(...),
    smokingStatus: str = Form(...),
    hypertensionHistory: str = Form(...)
):
    img_data = await image.read()
    input_img = preprocess_image(img_data)

    # Predict DR
    dr_pred = model_dr.predict(input_img)
    dr_level = np.argmax(dr_pred)
    dr_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']
    dr_result = dr_labels[dr_level]

    # Predict HTN
    htn_pred = model_htn.predict(input_img)
    htn_percent = float(np.clip(htn_pred[0][0] * 100, 0, 100))

    # Predict HbA1c
    sex_val = 1 if sex == "Male" else 0
    smoke_val = 1 if smokingStatus == "Yes" else 0
    # htn_hist_val = 1 if hypertensionHistory == "Yes" else 0
    features = np.array([[age, sex_val, BMI, smoke_val, htn_percent]])
    features_scaled = scaler.transform(features)
    hba1c = float(model_hba1c.predict(features_scaled)[0])

    # Atherosclerosis risk (derived)
    athero_risk = min((hba1c * 10 + htn_percent * 0.6), 100)

    return {
        "diabeticRetinopathyLevel": dr_result,
        "hypertensionRisk": round(htn_percent, 2),
        "hba1cLevel": round(hba1c, 2),
        "atherosclerosisRisk": round(athero_risk, 2)
    }
