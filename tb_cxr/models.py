import segmentation_models_pytorch as smp

def build_unet(encoder_name: str = "resnet18", encoder_weights: str | None = "imagenet"):
    """U-Net with a pretrained encoder for binary lung segmentation on grayscale CXR images."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,       # grayscale CXR
        classes=1,           # binary: lung vs. background
        activation=None,     # raw logits — sigmoid applied in the loss / at inference
    )