import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve

from tb_cxr.config import VARIANTS, WHO_SENSITIVITY, WHO_SPECIFICITY
from tb_cxr.data import classifier_inputs, lung_box

COLORS = ["#2a78d6", "#eb6834"]
WHO_FILL = "#d6efd6"

def plot_inputs(images: np.ndarray, masks: np.ndarray, titles: list[str], path) -> None:
    """Each row: an X-ray with its predicted lungs and crop box, then what the classifier sees under each variant."""
    fig, axes = plt.subplots(len(images), 1 + len(VARIANTS), figsize=(3 * (1 + len(VARIANTS)), 3.2 * len(images)), squeeze=False)
    views = {variant: classifier_inputs(images, masks, variant) for variant in VARIANTS}
    for row, (image, mask, title) in enumerate(zip(images, masks, titles)):
        top, bottom, left, right = lung_box(mask)
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].contour(mask, levels=[0.5], colors=COLORS[0], linewidths=1)
        axes[row, 0].add_patch(Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor=COLORS[1], linewidth=1.5))
        axes[row, 0].set_title(title, fontsize=10)
        for col, variant in enumerate(VARIANTS, start=1):
            axes[row, col].imshow(views[variant][row], cmap="gray")
            axes[row, col].set_title(variant, fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_roc(curves: dict, path) -> None:
    """ROC curves against the WHO triage minimum, each with its operating point and 95% intervals."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.add_patch(Rectangle((0, WHO_SENSITIVITY), 1 - WHO_SPECIFICITY, 1 - WHO_SENSITIVITY, color=WHO_FILL, zorder=0))
    ax.text(0.01, 0.985, "WHO triage minimum", fontsize=9, va="top")
    for color, (name, (y, p, point)) in zip(COLORS, curves.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=name)
        sens, spec = point["sensitivity"], point["specificity"]
        x, x_ci = 1 - spec["value"], [1 - v for v in spec["ci95"][::-1]]
        ax.errorbar(x, sens["value"], xerr=[[x - x_ci[0]], [x_ci[1] - x]], yerr=[[sens["value"] - sens["ci95"][0]], [sens["ci95"][1] - sens["value"]]],
                    fmt="o", markersize=8, color=color, markeredgecolor="white", capsize=3, zorder=3)
    ax.plot([0, 1], [0, 1], color="0.75", linestyle=":", linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.005), xlabel="1 - specificity (normal X-rays flagged)", ylabel="sensitivity (TB X-rays flagged)")
    ax.grid(color="0.92")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, title="Dots: threshold fixed in advance", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
