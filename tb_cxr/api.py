from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile

from tb_cxr.config import CLASSIFIER_PATH, UNET_PATH, WORK_SIZE
from tb_cxr.data import classifier_inputs, square
from tb_cxr.models import load_pipeline, predict_masks, predict_tb

pipeline = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.update(load_pipeline(UNET_PATH, CLASSIFIER_PATH))
    yield

app = FastAPI(title="TB-CXR screening", lifespan=lifespan)

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "variant": pipeline["variant"], "threshold": pipeline["threshold"]}

@app.post("/predict")
async def predict(file: UploadFile) -> dict:
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=400, detail="Unreadable image file.")
    image = square(image, WORK_SIZE)[None]
    masks = image if pipeline["variant"] == "full image" else predict_masks(pipeline["unet"], image)
    inputs = classifier_inputs(image, masks, pipeline["variant"])
    probability = float(predict_tb(pipeline["classifiers"], inputs)[0])
    return {
        "tb_probability": round(probability, 4),
        "tb_suggestive": probability >= pipeline["threshold"],
        "threshold": pipeline["threshold"],
        "note": "Research project, not a diagnostic device."
    }
