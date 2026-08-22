import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from tb_cxr.models import build_unet
from tb_cxr.datasets import LungSegmentationDataset
from tb_cxr.transforms import get_segmentation_transforms
from tb_cxr.losses import DiceBCELoss, dice_coefficient, iou_score

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    manifest = pd.read_csv(config["manifest_path"])

    train_ds = LungSegmentationDataset(manifest[manifest.split == "train"], get_segmentation_transforms(train=True))
    val_ds = LungSegmentationDataset(manifest[manifest.split == "val"], get_segmentation_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    model = build_unet().to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    checkpoint_path = Path("outputs/checkpoints/unet_best.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0
    best_val_iou = 0.0
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += criterion(logits, masks).item() * images.size(0)
                val_dice += dice_coefficient(logits, masks).item() * images.size(0)
                val_iou += iou_score(logits, masks).item() * images.size(0)
        val_loss /= len(val_ds)
        val_dice /= len(val_ds)
        val_iou /= len(val_ds)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_dice={val_dice:.4f} val_iou={val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_iou = val_iou
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved new best checkpoint (dice={val_dice:.4f})")

    return best_val_dice, best_val_iou, val_ds

def save_segmentation_overlays(model, dataset, device, n: int = 6, out_path: str = "outputs/figures/segmentation_overlays.png"):
    model.eval()
    n = min(n, len(dataset))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        image, mask = dataset[i]
        with torch.no_grad():
            pred = torch.sigmoid(model(image.unsqueeze(0).to(device))).squeeze().cpu()
        axes[0, i].imshow(image.squeeze(), cmap="gray")
        axes[0, i].imshow(mask.squeeze(), alpha=0.3, cmap="Reds")
        axes[0, i].set_title("Ground truth")
        axes[0, i].axis("off")
        axes[1, i].imshow(image.squeeze(), cmap="gray")
        axes[1, i].imshow(pred > 0.5, alpha=0.3, cmap="Blues")
        axes[1, i].set_title("Prediction")
        axes[1, i].axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved overlay figure to {out_path}")

if __name__ == "__main__":
    config = {
        "manifest_path": "data/processed/manifest_split.csv",
        "batch_size": 8,
        "lr": 1e-4,
        "epochs": 30,
    }
    best_dice, best_iou, val_ds = train(config)
    print(f"\nBest val Dice: {best_dice:.4f} | Best val IoU: {best_iou:.4f}")

    # Reload the best checkpoint (not just whatever's in memory at the last epoch)
    # before generating the sanity-check figure.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/unet_best.pt", map_location=device))
    save_segmentation_overlays(model, val_ds, device)