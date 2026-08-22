import segmentation_models_pytorch as smp
import torch.nn as nn
import torchvision.models as models

def build_classifier():
    """ResNet18 pretrained on ImageNet, fine-tuned for binary TB classification."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # binary: normal / TB
    return model

def set_backbone_trainable(model, trainable: bool):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable

def build_unet(encoder_name: str = "resnet18", encoder_weights: str | None = "imagenet"):
    """U-Net with a pretrained encoder for binary lung segmentation on grayscale CXR images."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,       # grayscale CXR
        classes=1,           # binary: lung vs. background
        activation=None,     # raw logits — sigmoid applied in the loss / at inference
    )