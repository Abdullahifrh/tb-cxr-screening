import numpy as np
import torch
from tb_cxr.models import augment_images, build_classifier, build_unet, classification_loss, fit, load_pipeline, predict_tb, save_pipeline

def test_output_shapes():
    assert build_unet(pretrained=False)(torch.zeros(1, 1, 64, 64)).shape == (1, 1, 64, 64)
    assert build_classifier(pretrained=False)(torch.zeros(2, 1, 64, 64)).shape == (2, 1)

def test_fit_and_predict_probabilities():
    images = np.random.default_rng(0).integers(0, 255, (8, 64, 64), np.uint8)
    labels = np.array([0, 1] * 4)
    model = fit(build_classifier(pretrained=False), images, labels, classification_loss, augment_images, epochs=1, lr=1e-3, seed=0, desc="test")
    p = predict_tb([model], images)
    assert p.shape == (8,) and ((0 <= p) & (p <= 1)).all()

def test_saved_pipeline_predicts_the_same(tmp_path):
    images = np.random.default_rng(0).integers(0, 255, (2, 64, 64), np.uint8)
    classifiers = [build_classifier(pretrained=False).eval() for _ in range(2)]
    save_pipeline(build_unet(pretrained=False), classifiers, "lung crop", 0.3, tmp_path / "unet.pt", tmp_path / "classifier.pt")
    loaded = load_pipeline(tmp_path / "unet.pt", tmp_path / "classifier.pt")
    assert (loaded["variant"], loaded["threshold"]) == ("lung crop", 0.3)
    assert np.allclose(predict_tb(loaded["classifiers"], images), predict_tb(classifiers, images))
