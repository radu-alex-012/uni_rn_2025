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
    # cf = np.column_stack((gt, pred))
    # tp = fp = fn = tn = 0
    # for e in cf:
    #     if e[0] == e[1]:
    #         if e[0] == 0:
    #             tn += 1
    #         else:
    #             tp += 1
    #     else:
    #         if e[0] == 0:
    #             fp += 1
    #         else:
    #             fn += 1
    an = np.sum(1 for v in gt if v == 0)
    ap = len(gt) - an
    tn = np.sum(1 for v1, v2 in zip (gt, pred) if v1 == v2 and v1 == 0)
    tp = np.sum(1 for v1, v2 in zip (gt, pred) if v1 == v2 and v1 == 1)
    fn = ap - tp
    fp = an - tn
    return tp, fp, fn, tn

assert tp_fp_fn_tn_sklearn(actual, predicted) == tp_fp_fn_tn_numpy(actual, predicted)

rez_1 = tp_fp_fn_tn_sklearn(big_actual, big_predicted)
rez_2 = tp_fp_fn_tn_numpy(big_actual, big_predicted)

assert rez_1 == rez_2