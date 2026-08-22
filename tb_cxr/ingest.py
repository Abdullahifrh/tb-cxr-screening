import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

LABEL_PATTERN = re.compile(r"_(\d)$")
SHENZHEN_MASK_PATTERNS = ["{stem}_mask.png", "{stem}.png"]

@dataclass
class ImageRecord:
    image_id: str
    dataset: str  # "montgomery" or "shenzhen"
    image_path: str
    label: int  # 0 = normal, 1 = TB
    left_mask_path: str | None
    right_mask_path: str | None
    width: int
    height: int

def parse_label_from_filename(image_stem: str) -> int:
    match = LABEL_PATTERN.search(image_stem)
    if not match:
        raise ValueError(f"Could not parse binary TB label from stem: {image_stem}")
    return int(match.group(1))

def _read_image_size(img_path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    except (UnidentifiedImageError, OSError) as exc:
        warnings.warn(f"Skipping unreadable image {img_path}: {exc}")
        return None

def find_mask_pair(
    image_stem: str, left_mask_dir: Path, right_mask_dir: Path
) -> tuple[str | None, str | None]:
    left = left_mask_dir / f"{image_stem}.png"
    right = right_mask_dir / f"{image_stem}.png"
    return (
        str(left) if left.exists() else None,
        str(right) if right.exists() else None,
    )

def find_shenzhen_mask(image_stem: str, mask_root: Path) -> str | None:
    for pattern in SHENZHEN_MASK_PATTERNS:
        candidate = mask_root / pattern.format(stem=image_stem)
        if candidate.exists():
            return str(candidate)
    return None

def ingest_montgomery(root: Path) -> list[ImageRecord]:
    cxr_dir = root / "CXR_png"
    left_mask_dir = root / "ManualMask" / "leftMask"
    right_mask_dir = root / "ManualMask" / "rightMask"

    if not cxr_dir.is_dir():
        raise FileNotFoundError(
            f"Expected Montgomery images at {cxr_dir}, but directory was not found."
        )

    records = []
    for img_path in sorted(cxr_dir.glob("*.png")):
        size = _read_image_size(img_path)
        if size is None:
            continue
        width, height = size
        label = parse_label_from_filename(img_path.stem)
        left_mask, right_mask = find_mask_pair(
            img_path.stem, left_mask_dir, right_mask_dir
        )
        records.append(
            ImageRecord(
                image_id=img_path.stem,
                dataset="montgomery",
                image_path=str(img_path),
                label=label,
                left_mask_path=left_mask,
                right_mask_path=right_mask,
                width=width,
                height=height,
            )
        )
    return records

def ingest_shenzhen(root: Path, mask_root: Path | None = None) -> list[ImageRecord]:
    cxr_dir = root / "CXR_png"

    if not cxr_dir.is_dir():
        raise FileNotFoundError(
            f"Expected Shenzhen images at {cxr_dir}, but directory was not found."
        )

    if mask_root is not None and not mask_root.is_dir():
        warnings.warn(
            f"Shenzhen mask root {mask_root} not found — proceeding without masks."
        )
        mask_root = None

    records = []
    for img_path in sorted(cxr_dir.glob("*.png")):
        size = _read_image_size(img_path)
        if size is None:
            continue
        width, height = size
        label = parse_label_from_filename(img_path.stem)
        left_mask = (
            find_shenzhen_mask(img_path.stem, mask_root)
            if mask_root is not None
            else None
        )

        records.append(
            ImageRecord(
                image_id=img_path.stem,
                dataset="shenzhen",
                image_path=str(img_path),
                label=label,
                left_mask_path=left_mask,
                right_mask_path=None,
                width=width,
                height=height,
            )
        )
    return records

def build_manifest(
    montgomery_root: Path,
    shenzhen_root: Path,
    shenzhen_mask_root: Path | None,
    out_csv: Path,
) -> pd.DataFrame:
    records = ingest_montgomery(montgomery_root) + ingest_shenzhen(
        shenzhen_root, shenzhen_mask_root
    )

    if not records:
        raise RuntimeError("No records ingested. Check raw dataset directory paths.")

    df = pd.DataFrame([asdict(r) for r in records])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Successfully wrote {len(df)} records to {out_csv}\n")
    print("Dataset counts by label (0 = Normal, 1 = TB):")
    print(df.groupby(["dataset", "label"]).size())

    n_shenzhen = int((df["dataset"] == "shenzhen").sum())
    n_shenzhen_masks = int(
        df.loc[df["dataset"] == "shenzhen", "left_mask_path"].notna().sum()
    )
    print(f"\nShenzhen mask coverage: {n_shenzhen_masks} / {n_shenzhen}")

    return df

def make_splits(
    manifest: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split, jointly on (dataset, label) so both the class balance
    AND the Montgomery/Shenzhen ratio are preserved in every split."""
    strat_key = manifest[["dataset", "label"]].astype(str).agg("_".join, axis=1)
    trainval, test = train_test_split(
        manifest, test_size=test_size, stratify=strat_key, random_state=seed
    )
    strat_key_tv = trainval[["dataset", "label"]].astype(str).agg("_".join, axis=1)
    train, val = train_test_split(
        trainval,
        test_size=val_size / (1 - test_size),
        stratify=strat_key_tv,
        random_state=seed,
    )
    return train, val, test

def add_splits(
    manifest_csv: Path, out_csv: Path, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    train, val, test = make_splits(manifest, test_size=test_size, val_size=val_size, seed=seed)

    manifest = manifest.copy()
    manifest["split"] = ""
    manifest.loc[train.index, "split"] = "train"
    manifest.loc[val.index, "split"] = "val"
    manifest.loc[test.index, "split"] = "test"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_csv, index=False)

    print(f"Wrote {len(manifest)} records with split assignments to {out_csv}\n")
    print("Split sizes:")
    print(manifest["split"].value_counts())
    print("\nClass/dataset balance per split:")
    print(manifest.groupby(["split", "dataset", "label"]).size())

    return manifest

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the TB-CXR manifest, or add a stratified train/val/test split to an existing one."
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help=(
            "Skip ingestion; read the existing data/processed/manifest.csv, add a "
            "patient-level stratified train/val/test split column, and write "
            "data/processed/manifest_split.csv."
        ),
    )
    args = parser.parse_args()

    if args.split:
        add_splits(
            manifest_csv=Path("data/processed/manifest.csv"),
            out_csv=Path("data/processed/manifest_split.csv"),
        )
    else:
        build_manifest(
            montgomery_root=Path("data/raw/Montgomery/MontgomerySet"),
            shenzhen_root=Path("data/raw/ChinaSet_AllFiles/ChinaSet_AllFiles"),
            shenzhen_mask_root=Path("data/raw/shenzhen_masks/mask"),
            out_csv=Path("data/processed/manifest.csv"),
        )