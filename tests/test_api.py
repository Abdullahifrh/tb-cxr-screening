import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tb_cxr import api
from tb_cxr.models import build_classifier, build_unet, save_pipeline

@pytest.fixture
def client(tmp_path, monkeypatch):
    unet_path, classifier_path = tmp_path / "unet.pt", tmp_path / "classifier.pt"
    save_pipeline(build_unet(pretrained=False), [build_classifier(pretrained=False)], "lung crop", 0.5, unet_path, classifier_path)
    monkeypatch.setattr(api, "UNET_PATH", unet_path)
    monkeypatch.setattr(api, "CLASSIFIER_PATH", classifier_path)
    with TestClient(api.app) as client:
        yield client

def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "variant": "lung crop", "threshold": 0.5}

def test_predict(client):
    _, png = cv2.imencode(".png", np.random.default_rng(0).integers(0, 255, (300, 260), np.uint8))
    body = client.post("/predict", files={"file": ("xray.png", png.tobytes(), "image/png")}).json()
    assert 0 <= body["tb_probability"] <= 1
    assert body["tb_suggestive"] == (body["tb_probability"] >= 0.5)

def test_full_image_variant_skips_segmentation(client, monkeypatch):
    monkeypatch.setitem(api.pipeline, "variant", "full image")
    monkeypatch.setattr(api, "predict_masks", lambda *_: pytest.fail("segmentation should not run"))
    _, png = cv2.imencode(".png", np.zeros((64, 64), np.uint8))
    assert client.post("/predict", files={"file": ("xray.png", png.tobytes(), "image/png")}).status_code == 200

def test_unreadable_upload_is_rejected(client):
    assert client.post("/predict", files={"file": ("xray.png", b"not an image", "image/png")}).status_code == 400
