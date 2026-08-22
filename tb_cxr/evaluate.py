import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from tb_cxr.models import build_classifier, build_unet
from tb_cxr.datasets import TBClassificationDataset
from tb_cxr.transforms import get_classification_transforms

WHO_TPP_MIN_SENSITIVITY = 0.90
WHO_TPP_MIN_SPECIFICITY = 0.70

def compute_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)   # recall on the TB-positive class
    specificity = tn / (tn + fp)
    return sensitivity, specificity

def evaluate_against_who_tpp(y_true, y_probs, threshold=0.5):
    y_pred = (np.array(y_probs) >= threshold).astype(int)
    sensitivity, specificity = compute_sensitivity_specificity(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)

    print(f"Sensitivity: {sensitivity:.3f}  (WHO TPP minimum: {WHO_TPP_MIN_SENSITIVITY})")
    print(f"Specificity: {specificity:.3f}  (WHO TPP minimum: {WHO_TPP_MIN_SPECIFICITY})")
    print(f"ROC-AUC:     {auc:.3f}")
    print(f"Meets WHO TPP threshold at this operating point: "
          f"{sensitivity >= WHO_TPP_MIN_SENSITIVITY and specificity >= WHO_TPP_MIN_SPECIFICITY}")
    return {"sensitivity": sensitivity, "specificity": specificity, "auc": auc, "threshold": threshold}

def find_operating_point_for_target_sensitivity(y_true, y_probs, target_sensitivity=0.90):
    """WHO TPP is a sensitivity-first requirement (missing a TB case is worse
    than a false alarm), so report the specificity you'd get at the
    threshold that hits the required sensitivity — this is the honest,
    clinically-relevant way to report the trade-off, not just accuracy."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    valid = tpr >= target_sensitivity
    if not valid.any():
        return None
    idx = np.argmax(valid)  # first (highest) threshold that reaches target sensitivity
    return {"threshold": thresholds[idx], "sensitivity": tpr[idx], "specificity": 1 - fpr[idx]}

def save_confusion_matrix(y_true, y_pred, out_path="outputs/figures/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "TB"])
    disp.plot(cmap="Blues")
    plt.title("TB Classification — Confusion Matrix (Test Set)")
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_roc_curve(y_true, y_probs, out_path="outputs/figures/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.axhline(WHO_TPP_MIN_SENSITIVITY, color="red", linestyle="--", label="WHO TPP min. sensitivity")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.legend()
    plt.title("ROC Curve vs. WHO Target Product Profile")
    plt.savefig(out_path, dpi=150)
    plt.close()

def _load_models(config, device):
    seg_model = build_unet().to(device)
    seg_model.load_state_dict(torch.load(config["segmentation_checkpoint"], map_location=device))
    seg_model.eval()

    clf_model = build_classifier().to(device)
    clf_model.load_state_dict(torch.load(config["classifier_checkpoint"], map_location=device))
    clf_model.eval()
    return seg_model, clf_model

@torch.no_grad()
def _run_inference(df, seg_model, clf_model, device, batch_size, desc="Inference"):
    ds = TBClassificationDataset(df, get_classification_transforms(train=False), seg_model, device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    for images, labels in tqdm(loader, desc=desc):
        images = images.to(device)
        probs = torch.softmax(clf_model(images), dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)

if __name__ == "__main__":
    config = {
        "manifest_path": "data/processed/manifest_split.csv",
        "segmentation_checkpoint": "outputs/checkpoints/unet_best.pt",
        "classifier_checkpoint": "outputs/checkpoints/classifier_best.pt",
        "batch_size": 16,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(config["manifest_path"])
    seg_model, clf_model = _load_models(config, device)

    val_true, val_probs = _run_inference(
        manifest[manifest.split == "val"], seg_model, clf_model, device, config["batch_size"], desc="Val Set Inference"
    )
    test_true, test_probs = _run_inference(
        manifest[manifest.split == "test"], seg_model, clf_model, device, config["batch_size"], desc="Test Set Inference"
    )

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    print("\n=== Test set — default threshold (0.5) ===")
    evaluate_against_who_tpp(test_true, test_probs, threshold=0.5)

    print("\n=== Threshold tuned on validation set for >=90% sensitivity ===")
    tuned = find_operating_point_for_target_sensitivity(val_true, val_probs, target_sensitivity=0.90)
    if tuned is None:
        print("No threshold on the validation ROC curve reaches 90% sensitivity; skipping tuned-point test evaluation.")
    else:
        print(f"Tuned threshold (selected on val): {tuned['threshold']:.4f}")
        print("\n=== Test set — tuned threshold ===")
        evaluate_against_who_tpp(test_true, test_probs, threshold=tuned["threshold"])

    y_pred_default = (test_probs >= 0.5).astype(int)
    save_confusion_matrix(test_true, y_pred_default)
    save_roc_curve(test_true, test_probs)
    print("\nSaved outputs/figures/confusion_matrix.png and outputs/figures/roc_curve.png")