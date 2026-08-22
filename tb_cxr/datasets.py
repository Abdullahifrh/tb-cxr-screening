import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tb_cxr.transforms import combine_lr_masks, get_segmentation_transforms

class LungSegmentationDataset(Dataset):
    """Montgomery-only. Returns (image_tensor, mask_tensor)."""

    def __init__(self, manifest: pd.DataFrame, transforms):
        self.df = manifest[manifest["dataset"] == "montgomery"].reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
        mask = combine_lr_masks(row.left_mask_path, row.right_mask_path, image.shape)
        augmented = self.transforms(image=image, mask=mask)
        image_t = augmented["image"]
        mask_t = augmented["mask"].unsqueeze(0).float()
        return image_t, mask_t

class TBClassificationDataset(Dataset):
    """Both datasets. Returns (image_tensor, label_tensor).

    If a segmentation_model is provided, applies it at inference time to mask
    out non-lung tissue before classification. The model is expected to
    already be on `device` and in eval mode.
    """

    def __init__(self, manifest: pd.DataFrame, transforms, segmentation_model=None, device="cpu"):
        self.df = manifest.reset_index(drop=True)
        self.transforms = transforms
        self.segmentation_model = segmentation_model
        self.device = device
        self._seg_transform = get_segmentation_transforms(train=False) if segmentation_model is not None else None

    def __len__(self):
        return len(self.df)

    def _apply_lung_mask(self, image_gray: np.ndarray) -> np.ndarray:
        seg_input = self._seg_transform(image=image_gray)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_mask = torch.sigmoid(self.segmentation_model(seg_input))
            pred_mask = (pred_mask > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (image_gray.shape[1], image_gray.shape[0]))
        return image_gray * pred_mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_gray = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
        if self.segmentation_model is not None:
            image_gray = self._apply_lung_mask(image_gray)
        image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        augmented = self.transforms(image=image_rgb)
        image_t = augmented["image"]
        label_t = torch.tensor(row.label, dtype=torch.long)
        return image_t, label_t