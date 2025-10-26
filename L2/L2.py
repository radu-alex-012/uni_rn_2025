import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from timed_decorator.simple_timed import timed
from typing import Tuple

actual = np.array([
    1,1,1,1,0,0,1,0,0,0
])
predicted = np.array([
    1,1,1,0,1,0,1,1,0,0
])

big_size = 500000
big_actual = np.repeat(actual, big_size)
big_predicted = np.repeat(predicted, big_size)

@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_sklearn(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    return tp, fp, fn, tn

@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_numpy(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = np.sum(gt & pred)
    tn = np.sum(~gt & ~pred)
    fp = np.sum(~gt & pred)
    fn = np.sum(gt & ~pred)
    return tp, fp, fn, tn

assert tp_fp_fn_tn_sklearn(actual, predicted) == tp_fp_fn_tn_numpy(actual, predicted)

rez_1 = tp_fp_fn_tn_sklearn(big_actual, big_predicted)
rez_2 = tp_fp_fn_tn_numpy(big_actual, big_predicted)

assert rez_1 == rez_2