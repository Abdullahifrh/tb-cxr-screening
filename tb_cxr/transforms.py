import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEG_IMAGE_SIZE = 512
CLS_IMAGE_SIZE = 224

def combine_lr_masks(left_mask_path: str | None, right_mask_path: str | None, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for path in (left_mask_path, right_mask_path):
        if path:
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                # corrupt/unreadable mask file — skip rather than crash the whole batch
                continue
            m = cv2.resize(m, (shape[1], shape[0]))
            mask = np.maximum(mask, (m > 127).astype(np.uint8))
    return mask

def get_segmentation_transforms(train: bool):
    if train:
        return A.Compose([
            A.Resize(SEG_IMAGE_SIZE, SEG_IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(SEG_IMAGE_SIZE, SEG_IMAGE_SIZE),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

def get_classification_transforms(train: bool):
    if train:
        return A.Compose([
            A.Resize(CLS_IMAGE_SIZE, CLS_IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=8, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(CLS_IMAGE_SIZE, CLS_IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])