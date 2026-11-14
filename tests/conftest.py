import cv2
import numpy as np
import pytest

from tb_cxr import data

@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """A miniature raw layout: ten X-rays per site, one child per site, and one Shenzhen image without a mask."""
    sites = {"montgomery": tmp_path / "MontgomerySet", "shenzhen": tmp_path / "ChinaSet"}
    mask_dir = tmp_path / "shenzhen_masks"
    rng = np.random.default_rng(0)
    lungs = np.zeros((80, 64), np.uint8)
    lungs[20:60, 8:28] = lungs[20:60, 36:56] = 255
    for site, root in sites.items():
        for folder in ("CXR_png", "ClinicalReadings", "ManualMask/leftMask", "ManualMask/rightMask"):
            (root / folder).mkdir(parents=True)
        mask_dir.mkdir(exist_ok=True)
        for i in range(10):
            image_id = f"{'MCUCXR' if site == 'montgomery' else 'CHNCXR'}_{i:04d}_{i % 2}"
            age = 10 if i == 0 else 40
            reading = f"Patient's Sex: F\nPatient's Age: {age:03d}Y\nnormal" if site == "montgomery" else f"male {age}yrs\nnormal"
            (root / "ClinicalReadings" / f"{image_id}.txt").write_text(reading)
            cv2.imwrite(str(root / "CXR_png" / f"{image_id}.png"), rng.integers(0, 255, (80, 64), np.uint8))
            if site == "montgomery":
                cv2.imwrite(str(root / "ManualMask/leftMask" / f"{image_id}.png"), np.where(np.arange(64) < 32, lungs, 0).astype(np.uint8))
                cv2.imwrite(str(root / "ManualMask/rightMask" / f"{image_id}.png"), np.where(np.arange(64) >= 32, lungs, 0).astype(np.uint8))
            elif i != 9:
                cv2.imwrite(str(mask_dir / f"{image_id}_mask.png"), lungs)
    monkeypatch.setattr(data, "SITES", sites)
    monkeypatch.setattr(data, "SHENZHEN_MASK_DIR", mask_dir)
    monkeypatch.setattr(data, "CACHE_PATH", tmp_path / "cache.npz")
    return tmp_path
