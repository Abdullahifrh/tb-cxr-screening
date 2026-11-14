import math

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torchvision import models, tv_tensors
from torchvision.transforms import v2
from tqdm import tqdm

from tb_cxr.config import BATCH_SIZE, EPOCHS, FOLDS, LEARNING_RATE, SEED, SEG_EPOCHS, SEG_LEARNING_RATE, SEG_SIZE, WORK_SIZE

MEAN, STD = 0.449, 0.226
AUGMENT = v2.Compose([
    v2.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
])

def imagenet_weights() -> dict:
    """ImageNet ResNet18 weights with the first convolution summed over RGB, so the network takes one grey channel."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
    weights["conv1.weight"] = weights["conv1.weight"].sum(1, keepdim=True)
    return weights

def build_unet(pretrained: bool = True) -> nn.Module:
    model = smp.Unet("resnet18", encoder_weights=None, in_channels=1, classes=1)
    if pretrained:
        model.encoder.load_state_dict(imagenet_weights())
    return model

def build_classifier(pretrained: bool = True) -> nn.Module:
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        model.load_state_dict(imagenet_weights())
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def segmentation_loss(logits, masks):
    probs = torch.sigmoid(logits)
    dice = 2 * (probs * masks).sum((1, 2, 3)) / (probs.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, masks) + (1 - dice).mean()

def classification_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits.squeeze(1), labels)

def augment_images(images, labels):
    return torch.stack([AUGMENT(image) for image in images]), labels

def augment_pairs(images, masks):
    pairs = [AUGMENT(tv_tensors.Image(image), tv_tensors.Mask(mask)) for image, mask in zip(images, masks)]
    return torch.stack([image for image, _ in pairs]), torch.stack([mask for _, mask in pairs])

def normalize(images: torch.Tensor) -> torch.Tensor:
    return (images.float() / 255 - MEAN) / STD

def fit(model, images, targets, loss, augment, epochs, lr, seed, desc):
    """Train on uint8 images (N, H, W) with a fixed number of epochs, so no held-out data is used for stopping."""
    torch.manual_seed(seed)
    images = torch.from_numpy(images).unsqueeze(1)
    targets = torch.from_numpy(targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    with tqdm(total=epochs * math.ceil(len(images) / BATCH_SIZE), desc=desc, unit="batch") as bar:
        for epoch in range(epochs):
            for batch in torch.randperm(len(images)).split(BATCH_SIZE):
                x, y = augment(images[batch], targets[batch])
                optimizer.zero_grad()
                value = loss(model(normalize(x)), y.float())
                value.backward()
                optimizer.step()
                bar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{value.item():.3f}")
                bar.update()
    return model.eval()

@torch.no_grad()
def logits(model, images: np.ndarray) -> torch.Tensor:
    return torch.cat([model(normalize(torch.from_numpy(batch).unsqueeze(1))) for batch in np.array_split(images, max(1, len(images) // 64))])

def resize(images: np.ndarray, size: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    return np.stack([cv2.resize(image, (size, size), interpolation=interpolation) for image in images])

def train_unet(images: np.ndarray, masks: np.ndarray) -> nn.Module:
    masks = resize(masks, SEG_SIZE, cv2.INTER_NEAREST)[:, None]
    return fit(build_unet(), resize(images, SEG_SIZE), masks, segmentation_loss, augment_pairs, SEG_EPOCHS, SEG_LEARNING_RATE, SEED, "lung segmentation")

def predict_masks(model, images: np.ndarray) -> np.ndarray:
    probs = torch.sigmoid(logits(model, resize(images, SEG_SIZE)))
    return (F.interpolate(probs, size=WORK_SIZE, mode="bilinear")[:, 0] > 0.5).numpy().astype(np.uint8)

def predict_tb(classifiers: list, images: np.ndarray) -> np.ndarray:
    """Mean TB probability over the fold models."""
    return np.mean([torch.sigmoid(logits(model, images))[:, 0].numpy() for model in classifiers], axis=0)

def cross_validate(images: np.ndarray, labels: np.ndarray, strata: np.ndarray, name: str) -> tuple[np.ndarray, list]:
    """Out-of-fold TB probabilities for every image, and the fold models for predicting new images."""
    oof, classifiers = np.zeros(len(labels)), []
    for fold, (fit_idx, held_idx) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(images, strata)):
        desc = f"{name}, fold {fold + 1}/{FOLDS}"
        model = fit(build_classifier(), images[fit_idx], labels[fit_idx], classification_loss, augment_images, EPOCHS, LEARNING_RATE, SEED + fold, desc)
        oof[held_idx] = predict_tb([model], images[held_idx])
        classifiers.append(model)
    return oof, classifiers

def save_pipeline(unet, classifiers: list, variant: str, threshold: float, unet_path, classifier_path) -> None:
    unet_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unet.state_dict(), unet_path)
    torch.save({"variant": variant, "threshold": threshold, "models": [model.state_dict() for model in classifiers]}, classifier_path)

def load_pipeline(unet_path, classifier_path) -> dict:
    unet = build_unet(pretrained=False)
    unet.load_state_dict(torch.load(unet_path, weights_only=True))
    bundle = torch.load(classifier_path, weights_only=True)
    classifiers = []
    for state in bundle["models"]:
        classifiers.append(build_classifier(pretrained=False))
        classifiers[-1].load_state_dict(state)
    return {"unet": unet.eval(), "classifiers": [model.eval() for model in classifiers], "variant": bundle["variant"], "threshold": bundle["threshold"]}
