import numpy as np
import pytest

from tb_cxr.config import INPUT_SIZE, WORK_SIZE
from tb_cxr.data import build_manifest, classifier_inputs, load_images, lung_box, parse_age, square

@pytest.mark.parametrize("reading, age", [
    ("male 34yrs\nRight PTB", 34),
    ("Male , 32yrs\nsecondary PTB", 32),
    ("male 42\nnormal", 42),
    ("male 16month\nnormal", 16 / 12),
    ("female 64days\nnormal", 64 / 365),
    ("Patient's Sex: F\nPatient's Age: 029Y\nnormal", 29),
])
def test_parse_age(reading, age):
    assert parse_age(reading) == pytest.approx(age)

def test_manifest_excludes_children_and_splits_adults(raw_dir):
    manifest = build_manifest()
    assert ((manifest.age < 15) == (manifest.split == "excluded")).all()
    assert (manifest.label == manifest.image_id.str[-1].astype(int)).all()
    assert set(manifest.loc[manifest.split == "test", "site"]) == {"montgomery", "shenzhen"}
    assert manifest.equals(build_manifest())

def test_load_images_marks_missing_masks_and_caches(raw_dir):
    manifest = build_manifest()
    images, masks, has_mask = load_images(manifest)
    assert images.shape == masks.shape == (len(manifest), WORK_SIZE, WORK_SIZE)
    assert set(manifest.image_id[~has_mask]) == {"CHNCXR_0009_1"}
    assert set(np.unique(masks)) == {0, 1}
    assert np.array_equal(load_images(manifest)[0], images)

def test_square_keeps_aspect_ratio():
    out = square(np.ones((100, 50), np.uint8), 100)
    assert out[:, :25].sum() == out[:, 75:].sum() == 0
    assert out[:, 25:75].all()

def test_lung_box_ignores_a_stray_blob():
    mask = np.zeros((100, 100), np.uint8)
    mask[30:70, 20:45] = mask[30:70, 55:80] = 1
    mask[2, 98] = 1
    top, _, _, right = lung_box(mask)
    assert top > 2 and right < 98

def test_lung_box_of_an_empty_mask_is_the_whole_image():
    assert lung_box(np.zeros((50, 40), np.uint8)) == (0, 50, 0, 40)

def test_classifier_inputs():
    images = np.full((2, WORK_SIZE, WORK_SIZE), 200, np.uint8)
    masks = np.zeros_like(images)
    masks[:, 100:400, 100:250] = masks[:, 100:400, 300:450] = 1
    for variant in ("full image", "lung mask", "lung crop"):
        assert classifier_inputs(images, masks, variant).shape == (2, INPUT_SIZE, INPUT_SIZE)
    assert classifier_inputs(images, masks, "lung mask")[:, 0, 0].sum() == 0
    with pytest.raises(ValueError):
        classifier_inputs(images, masks, "heart crop")
