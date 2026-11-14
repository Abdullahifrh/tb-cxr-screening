import numpy as np
from tb_cxr.metrics import auc, dice, operating_point, threshold_for_sensitivity, wilson

def test_wilson_interval():
    assert wilson(52, 59) == [0.775, 0.941]

def test_threshold_flags_the_required_share_of_tb_cases():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.65, 0.1])
    threshold = threshold_for_sensitivity(y, p, 0.75)
    point = operating_point(y, p, threshold)
    assert threshold == 0.7
    assert point["sensitivity"]["count"] == "3/4"
    assert point["specificity"]["count"] == "2/2"

def test_auc_of_a_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert auc(y, y + 0.1 * np.arange(6), n_boot=200) == {"value": 1.0, "ci95": [1.0, 1.0]}

def test_dice():
    mask = np.zeros((2, 4, 4), bool)
    mask[:, :2] = True
    assert dice(mask, mask).tolist() == [1.0, 1.0]
    assert dice(mask, ~mask).tolist() == [0.0, 0.0]
