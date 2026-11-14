import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from tb_cxr.config import CLASSIFIER_PATH, FOLDS, MANIFEST_PATH, MIN_AUC_GAIN, OUTPUT_DIR, SEED, UNET_PATH, VARIANTS, WHO_SENSITIVITY
from tb_cxr.data import build_manifest, classifier_inputs, load_images
from tb_cxr.metrics import auc, dice, evaluate, specificity_at_who_sensitivity, threshold_for_sensitivity
from tb_cxr.models import cross_validate, predict_masks, predict_tb, save_pipeline, train_unet
from tb_cxr.plots import plot_inputs, plot_roc

START = time.time()

def log(message: str) -> None:
    print(f"[{(time.time() - START) / 60:5.1f} min] {message}", flush=True)

def choose_variant(oof_auc: dict) -> str:
    """The best input variant, unless it needs segmentation and beats the full image by less than MIN_AUC_GAIN."""
    best = max(oof_auc, key=oof_auc.get)
    return best if oof_auc[best] - oof_auc["full image"] >= MIN_AUC_GAIN else "full image"

def age_and_hospital_reference(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Out-of-fold probabilities from age and hospital alone, the score any image model has to beat."""
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    return cross_val_predict(LogisticRegression(), features, labels, cv=folds, method="predict_proba")[:, 1]

def segmentation_summary(pred: np.ndarray, true: np.ndarray, sites: np.ndarray) -> dict:
    scores = dice(pred.astype(bool), true.astype(bool))
    return {
        site: {"images": int((sites == site).sum()), "median_dice": round(float(np.median(scores[sites == site])), 3), "worst_dice": round(float(scores[sites == site].min()), 3)}
        for site in np.unique(sites)
    }

def main() -> None:
    log("21 training runs: 1 lung segmentation, 3 inputs x 5 folds, 5 folds for the external test")
    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    cohort = manifest[manifest.split != "excluded"].reset_index(drop=True)
    images, true_masks, has_mask = load_images(cohort)
    y, sites = cohort.label.to_numpy(), cohort.site.to_numpy()
    strata = (cohort.site + cohort.label.astype(str)).to_numpy()
    train, test = (cohort.split == "train").to_numpy(), (cohort.split == "test").to_numpy()
    shenzhen = sites == "shenzhen"
    log(f"{len(cohort)} adult X-rays loaded, {train.sum()} train and {test.sum()} test")

    unet = train_unet(images[train & has_mask], true_masks[train & has_mask])
    masks = predict_masks(unet, images)
    scored = test & has_mask
    log("lung segmentation trained")

    inputs, oof, classifiers = {}, {}, {}
    for variant in VARIANTS:
        inputs[variant] = classifier_inputs(images, masks, variant)
        oof[variant], classifiers[variant] = cross_validate(inputs[variant][train], y[train], strata[train], variant)
        log(f"{variant} cross-validated")
    oof["age and hospital only"] = age_and_hospital_reference(np.column_stack([cohort.age, shenzhen])[train], y[train])
    compared = {v: {"auc": auc(y[train], p), "specificity_at_who_sensitivity": specificity_at_who_sensitivity(y[train], p)} for v, p in oof.items()}
    variant = choose_variant({v: compared[v]["auc"]["value"] for v in VARIANTS})
    log(f"out-of-fold AUC: { {v: compared[v]['auc']['value'] for v in VARIANTS} }, shipping {variant}")
    threshold = threshold_for_sensitivity(y[train], oof[variant], WHO_SENSITIVITY)
    test_p = predict_tb(classifiers[variant], inputs[variant][test])

    external_oof, external_classifiers = cross_validate(inputs[variant][shenzhen], y[shenzhen], strata[shenzhen], "external, Shenzhen only")
    external_threshold = threshold_for_sensitivity(y[shenzhen], external_oof, WHO_SENSITIVITY)
    external_p = predict_tb(external_classifiers, inputs[variant][~shenzhen])
    log("external test done")

    metrics = {
        "cohort": {
            "images": len(manifest),
            "excluded_under_15": int((manifest.split == "excluded").sum()),
            "split": {f"{split} {site}": {"images": len(g), "tb": int(g.label.sum())} for (split, site), g in cohort.groupby(["split", "site"])},
        },
        "segmentation_test": segmentation_summary(masks[scored], true_masks[scored], sites[scored]),
        "out_of_fold": compared,
        "shipped": {"variant": variant, "threshold": round(threshold, 4)},
        "test": evaluate(y[test], test_p, threshold) | {"auc_by_site": {s: auc(y[test & (sites == s)], test_p[sites[test] == s])["value"] for s in np.unique(sites)}},
        "external_shenzhen_to_montgomery": evaluate(y[~shenzhen], external_p, external_threshold),
    }
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_pipeline(unet, classifiers[variant], variant, threshold, UNET_PATH, CLASSIFIER_PATH)

    examples = [np.flatnonzero(test & (sites == s) & (y == 1))[0] for s in ("shenzhen", "montgomery")]
    plot_inputs(images[examples], masks[examples], [f"{sites[i].title()} test X-ray with TB" for i in examples], OUTPUT_DIR / "figures" / "classifier_inputs.png")
    plot_roc({
        "Test set, both hospitals": (y[test], test_p, metrics["test"]["at_threshold"]),
        "Montgomery, model trained on Shenzhen only": (y[~shenzhen], external_p, metrics["external_shenzhen_to_montgomery"]["at_threshold"]),
    }, OUTPUT_DIR / "figures" / "roc_vs_who.png")
    log(f"done: {variant}, test AUC {metrics['test']['auc']['value']}")

if __name__ == "__main__":
    main()
