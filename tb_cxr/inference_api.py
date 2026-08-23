import io
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from tb_cxr.models import build_classifier, build_unet
from tb_cxr.transforms import get_classification_transforms, get_segmentation_transforms

app = FastAPI(title="TB Screening Inference API")

device = torch.device("cpu")

# Threshold tuned on the validation set
TB_THRESHOLD = 0.2101

# Segmentation model
seg_model = build_unet(encoder_weights=None)
seg_model.load_state_dict(
    torch.load("outputs/checkpoints/unet_best.pt", map_location=device, weights_only=True)
)
seg_model.eval()

# Classifier model
cls_model = build_classifier()
cls_model.load_state_dict(
    torch.load("outputs/checkpoints/classifier_best.pt", map_location=device, weights_only=True)
)
cls_model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid or unreadable image file."})

    image_gray = np.array(image)

    seg_input = get_segmentation_transforms(train=False)(image=image_gray)["image"].unsqueeze(0)
    with torch.no_grad():
        mask = (torch.sigmoid(seg_model(seg_input)) > 0.5).squeeze().numpy().astype(np.uint8)
    
    mask = cv2.resize(
        mask, (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    masked = image_gray * mask
    masked_rgb = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)

    cls_input = get_classification_transforms(train=False)(image=masked_rgb)["image"].unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(cls_model(cls_input), dim=1).squeeze().numpy()

    tb_probability = float(probs[1])
    return {
        "tb_probability": tb_probability,
        "prediction": "TB-suggestive" if tb_probability >= TB_THRESHOLD else "Normal",
        "threshold_used": TB_THRESHOLD,
        "note": "Personal project — not a diagnostic tool.",
    }

@app.get("/health")
def health():
    return {"status": "ok"}