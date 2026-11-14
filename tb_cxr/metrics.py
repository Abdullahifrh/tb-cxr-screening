import numpy as np
from sklearn.metrics import roc_auc_score

from tb_cxr.config import SEED, WHO_SENSITIVITY

def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson interval for a proportion k/n; it stays sensible near 0 and 1, unlike the normal approximation."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return [round(centre - half, 3), round(centre + half, 3)]

def rate(k: int, n: int) -> dict:
    return {"value": round(k / n, 3), "ci95": wilson(k, n), "count": f"{k}/{n}"}

def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, sensitivity: float) -> float:
    """Highest threshold that flags at least the given share of TB cases."""
    positives = np.sort(p[y == 1])[::-1]
    return float(positives[int(np.ceil(sensitivity * len(positives) - 1e-9)) - 1])

def operating_point(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    flagged = p >= threshold
    return {
        "threshold": round(threshold, 4),
        "sensitivity": rate(int((flagged & (y == 1)).sum()), int((y == 1).sum())),
        "specificity": rate(int((~flagged & (y == 0)).sum()), int((y == 0).sum())),
    }

def auc(y: np.ndarray, p: np.ndarray, n_boot: int = 2000) -> dict:
    """ROC AUC with a 95% bootstrap interval over images."""
    rng = np.random.default_rng(SEED)
    samples = (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
    boots = [roc_auc_score(y[i], p[i]) for i in samples if 0 < y[i].sum() < len(i)]
    return {"value": round(roc_auc_score(y, p), 3), "ci95": [round(float(q), 3) for q in np.percentile(boots, [2.5, 97.5])]}

def specificity_at_who_sensitivity(y: np.ndarray, p: np.ndarray) -> dict:
    """The best specificity these predictions allow while still flagging 90% of TB cases."""
    return operating_point(y, p, threshold_for_sensitivity(y, p, WHO_SENSITIVITY))["specificity"]

def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """AUC, the result at a threshold fixed in advance, and the best specificity the same data allows."""
    return {"auc": auc(y, p), "at_threshold": operating_point(y, p, threshold), "specificity_at_who_sensitivity": specificity_at_who_sensitivity(y, p)}

def dice(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Dice overlap per image."""
    overlap = (pred & true).sum((1, 2))
    return 2 * overlap / (pred.sum((1, 2)) + true.sum((1, 2)))
