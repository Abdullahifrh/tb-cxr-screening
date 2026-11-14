from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SITES = {
    "montgomery": RAW_DIR / "Montgomery" / "MontgomerySet",
    "shenzhen": RAW_DIR / "ChinaSet_AllFiles" / "ChinaSet_AllFiles",
}
SHENZHEN_MASK_DIR = RAW_DIR / "shenzhen_masks" / "mask"
MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.csv"
CACHE_PATH = ROOT / "data" / "interim" / "images.npz"
OUTPUT_DIR = ROOT / "outputs"
UNET_PATH = OUTPUT_DIR / "checkpoints" / "unet.pt"
CLASSIFIER_PATH = OUTPUT_DIR / "checkpoints" / "classifier.pt"

MIN_AGE = 15
TEST_SIZE = 0.2
WORK_SIZE = 512
SEG_SIZE = 256
INPUT_SIZE = 224
CROP_MARGIN = 0.05
VARIANTS = ["full image", "lung mask", "lung crop"]
MIN_AUC_GAIN = 0.01

FOLDS = 5
EPOCHS = 8
SEG_EPOCHS = 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
SEG_LEARNING_RATE = 3e-4
SEED = 42

WHO_SENSITIVITY = 0.90
WHO_SPECIFICITY = 0.70
