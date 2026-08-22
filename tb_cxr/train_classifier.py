import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from tb_cxr.models import build_classifier, set_backbone_trainable, build_unet
from tb_cxr.datasets import TBClassificationDataset
from tb_cxr.transforms import get_classification_transforms

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    manifest = pd.read_csv(config["manifest_path"])

    seg_model = build_unet().to(device)
    seg_model.load_state_dict(torch.load(config["segmentation_checkpoint"], map_location=device))
    seg_model.eval()

    train_ds = TBClassificationDataset(
        manifest[manifest.split == "train"], get_classification_transforms(True), seg_model, device
    )
    val_ds = TBClassificationDataset(
        manifest[manifest.split == "val"], get_classification_transforms(False), seg_model, device
    )
    # num_workers=0: the dataset runs a U-Net forward pass per __getitem__, and pickling
    # that model into worker processes on Windows/CPU is slow and unnecessary here.
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    model = build_classifier().to(device)
    set_backbone_trainable(model, False)  # freeze backbone before the frozen phase actually starts
    criterion = torch.nn.CrossEntropyLoss()  # train split is ~51/49 — no weighting needed
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    best_val_auc = 0.0
    for epoch in range(config["epochs"]):
        if epoch == config["freeze_epochs"]:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            print("  -> unfroze backbone, dropped LR to 1e-5")

        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} [train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                probs = torch.softmax(model(images), dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        val_auc = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch + 1}: val_AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "outputs/checkpoints/classifier_best.pt")
            print(f"  -> saved new best checkpoint (auc={val_auc:.4f})")

    print(f"\nBest val AUC: {best_val_auc:.4f}")
    return best_val_auc

if __name__ == "__main__":
    train({
        "manifest_path": "data/processed/manifest_split.csv",
        "segmentation_checkpoint": "outputs/checkpoints/unet_best.pt",
        "batch_size": 16,
        "epochs": 15,
        "freeze_epochs": 3
    })