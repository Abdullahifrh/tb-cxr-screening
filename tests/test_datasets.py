from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from tb_cxr.datasets import LungSegmentationDataset, TBClassificationDataset
from tb_cxr.transforms import (
    combine_lr_masks,
    get_classification_transforms,
    get_segmentation_transforms,
)

MANIFEST_PATH = Path("data/processed/manifest.csv")

@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST_PATH.exists():
        pytest.skip("Manifest not built yet — run ingest.py first")
    return pd.read_csv(MANIFEST_PATH)

def test_segmentation_dataset_is_montgomery_only(manifest):
    ds = LungSegmentationDataset(manifest, get_segmentation_transforms(train=False))
    assert len(ds) == (manifest["dataset"] == "montgomery").sum()
    assert (ds.df["dataset"] == "montgomery").all()

def test_segmentation_dataset_item_shape(manifest):
    ds = LungSegmentationDataset(manifest, get_segmentation_transforms(train=False))
    image_t, mask_t = ds[0]
    assert image_t.shape == (1, 512, 512)
    assert mask_t.shape == (1, 512, 512)
    assert image_t.dtype == torch.float32
    assert mask_t.dtype == torch.float32

def test_segmentation_mask_is_binary(manifest):
    row = manifest[manifest["dataset"] == "montgomery"].iloc[0]
    image = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
    mask = combine_lr_masks(row.left_mask_path, row.right_mask_path, image.shape)
    assert set(np.unique(mask)) <= {0, 1}

def test_classification_dataset_covers_both_datasets(manifest):
    ds = TBClassificationDataset(manifest, get_classification_transforms(train=False))
    assert len(ds) == len(manifest)
    assert set(ds.df["dataset"].unique()) == {"montgomery", "shenzhen"}

def test_classification_dataset_item_shape(manifest):
    ds = TBClassificationDataset(manifest, get_classification_transforms(train=False))
    image_t, label_t = ds[0]
    assert image_t.shape == (3, 224, 224)
    assert image_t.dtype == torch.float32
    assert label_t.dtype == torch.long
    assert label_t.item() in (0, 1)