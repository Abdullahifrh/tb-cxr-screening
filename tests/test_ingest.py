import pandas as pd
from pathlib import Path
import pytest

MANIFEST_PATH = Path("data/processed/manifest.csv")

@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST_PATH.exists():
        pytest.skip("Manifest not built yet — run ingest.py first")
    return pd.read_csv(MANIFEST_PATH)

def test_no_duplicate_image_ids(manifest):
    assert manifest["image_id"].is_unique

def test_labels_are_binary(manifest):
    assert set(manifest["label"].unique()) <= {0, 1}

def test_all_image_paths_exist(manifest):
    missing = manifest[~manifest["image_path"].apply(lambda p: Path(p).exists())]
    assert missing.empty, f"{len(missing)} images referenced in manifest are missing on disk"

def test_montgomery_has_masks(manifest):
    mont = manifest[manifest["dataset"] == "montgomery"]
    assert mont["left_mask_path"].notna().all()
    assert mont["right_mask_path"].notna().all()

def test_expected_dataset_sizes(manifest):
    counts = manifest["dataset"].value_counts()
    assert counts.get("montgomery", 0) == 138
    assert counts.get("shenzhen", 0) == 662  # allow for known minor variance