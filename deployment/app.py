from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Multiclass Classification API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "models", "encoder.pkl"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class InputData(BaseModel):
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: float
    f7: float
    f8: float
    f9: float
    f10: float


@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/predict")
def predict(data: InputData):
    try:
        input_data = np.array([[
            data.f1, data.f2, data.f3, data.f4, data.f5,
            data.f6, data.f7, data.f8, data.f9, data.f10
        ]], dtype=np.float64)

        scaled_data = scaler.transform(input_data)
        class_index = int(model.predict(scaled_data)[0])
        prediction = encoder.inverse_transform([class_index])[0]

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0].tolist()

        return {
            "prediction": str(prediction),
            "class_index": class_index,
            "probabilities": probabilities
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")