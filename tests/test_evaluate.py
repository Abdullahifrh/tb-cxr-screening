import pytest
from tb_cxr.evaluate import compute_sensitivity_specificity, find_operating_point_for_target_sensitivity

def test_compute_sensitivity_specificity_mixed():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]  # 1 FP among negatives, 1 FN among positives
    sensitivity, specificity = compute_sensitivity_specificity(y_true, y_pred)
    assert sensitivity == pytest.approx(2 / 3)
    assert specificity == pytest.approx(0.5)

def test_compute_sensitivity_specificity_perfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    sensitivity, specificity = compute_sensitivity_specificity(y_true, y_pred)
    assert sensitivity == 1.0
    assert specificity == 1.0

def test_find_operating_point_reaches_target():
    y_true = [0, 0, 0, 1, 1, 1]
    y_probs = [0.1, 0.2, 0.3, 0.6, 0.8, 0.9]  # perfectly separable at 0.6
    result = find_operating_point_for_target_sensitivity(y_true, y_probs, target_sensitivity=1.0)
    assert result["threshold"] == pytest.approx(0.6)
    assert result["sensitivity"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(1.0)

def test_find_operating_point_unreachable_returns_none():
    y_true = [0, 0, 1, 1]
    y_probs = [0.9, 0.1, 0.2, 0.8]  # a positive scored lower than a negative
    result = find_operating_point_for_target_sensitivity(y_true, y_probs, target_sensitivity=1.01)
    assert result is None