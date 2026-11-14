import re

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tb_cxr.config import CACHE_PATH, CROP_MARGIN, INPUT_SIZE, MIN_AGE, SEED, SHENZHEN_MASK_DIR, SITES, TEST_SIZE, WORK_SIZE

AGE = re.compile(r"(\d+) *([a-z]*)", re.IGNORECASE)

def parse_age(reading: str) -> float:
    """Age in years from a clinical reading such as 'male 34yrs', 'female 64days' or "Patient's Age: 029Y"."""
    number, unit = AGE.search(reading).groups()
    unit = unit.lower()
    if unit.startswith("day"):
        return int(number) / 365
    if unit.startswith("month"):
        return int(number) / 12
    return float(number)

def build_manifest() -> pd.DataFrame:
    """One row per X-ray. Children are excluded; adults are split into train and test, stratified by site and label."""
    df = pd.DataFrame(
        {"image_id": path.stem, "site": site, "label": int(path.stem[-1]), "age": round(parse_age(path.read_text(errors="replace")), 2)}
        for site, root in SITES.items()
        for path in sorted((root / "ClinicalReadings").glob("*.txt"))
    )
    adults = df[df.age >= MIN_AGE]
    _, test = train_test_split(adults.index, test_size=TEST_SIZE, stratify=adults.site + adults.label.astype(str), random_state=SEED)
    df["split"] = np.select([df.age < MIN_AGE, df.index.isin(test)], ["excluded", "test"], "train")
    return df

def read_gray(path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image

def mask_paths(site: str, image_id: str) -> list:
    if site == "montgomery":
        return [SITES[site] / "ManualMask" / side / f"{image_id}.png" for side in ("leftMask", "rightMask")]
    return [SHENZHEN_MASK_DIR / f"{image_id}_mask.png"]

def square(image: np.ndarray, size: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Pad to a centred square before resizing, so the anatomy keeps its aspect ratio."""
    h, w = image.shape
    side = max(h, w)
    padded = np.zeros((side, side), image.dtype)
    padded[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = image
    return cv2.resize(padded, (size, size), interpolation=interpolation)

def load_images(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Square images and ground-truth lung masks, cached because decoding the 3K to 5K originals is slow."""
    ids = manifest.image_id.to_numpy(dtype=str)
    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH)
        if np.array_equal(cache["ids"], ids):
            return cache["images"], cache["masks"], cache["has_mask"]
    images, masks, has_mask = [], [], []
    for row in tqdm(manifest.itertuples(), total=len(manifest), desc="reading X-rays", unit="image"):
        images.append(square(read_gray(SITES[row.site] / "CXR_png" / f"{row.image_id}.png"), WORK_SIZE))
        paths = mask_paths(row.site, row.image_id)
        has_mask.append(all(path.exists() for path in paths))
        mask = np.maximum.reduce([read_gray(path) for path in paths]) if has_mask[-1] else np.zeros((1, 1), np.uint8)
        masks.append(square((mask > 127).astype(np.uint8), WORK_SIZE, cv2.INTER_NEAREST))
    images, masks, has_mask = np.stack(images), np.stack(masks), np.array(has_mask)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, ids=ids, images=images, masks=masks, has_mask=has_mask)
    return images, masks, has_mask

def lung_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Box around the two largest mask regions plus a margin, so a stray blob cannot widen it; the whole image if empty."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n == 1:
        return 0, mask.shape[0], 0, mask.shape[1]
    lungs = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[-2:]
    rows, cols = np.nonzero(np.isin(labels, lungs))
    pad = round(CROP_MARGIN * mask.shape[0])
    return max(rows.min() - pad, 0), min(rows.max() + pad + 1, mask.shape[0]), max(cols.min() - pad, 0), min(cols.max() + pad + 1, mask.shape[1])

def classifier_inputs(images: np.ndarray, masks: np.ndarray, variant: str) -> np.ndarray:
    """The classifier's view of each X-ray: the whole image, only the pixels inside the lung mask, or a crop around the lungs."""
    def view(image, mask):
        if variant == "lung mask":
            image = image * mask
        elif variant == "lung crop":
            top, bottom, left, right = lung_box(mask)
            image = image[top:bottom, left:right]
        elif variant != "full image":
            raise ValueError(f"Unknown variant: {variant}")
        return square(image, INPUT_SIZE)
    return np.stack([view(image, mask) for image, mask in zip(images, masks)])
