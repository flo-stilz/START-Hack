# Metric
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import r2_score
import torch
from numba import jit

@jit(nopython=True)
def concordance_index(y_true, y_preds):
    # compute CI
    # helper function
    h = lambda x: 1 if x > 0 else (0.5 if x == 0 else 0)
    pairs = np.concatenate((y_preds.reshape(-1,1),y_true.reshape(-1,1)),axis=1)
    order = (-pairs[:, 0]).argsort()
    pairs = pairs[order]

    rank = []
    for i in range(len(pairs) - 1):
        for j in range(i + 1, len(pairs)):
            diff = pairs[i, 1] - pairs[j, 1]
            val = h(diff)
            rank.append(val)

    CI = sum(rank) / len(rank)

    return CI


def area_under_PR(y_true: np.ndarray, y_preds: np.ndarray):
    # average precision and AUPR  are quite similar

    cutoff = 7

    y_true = pd.Series(y_true).apply(lambda x: 0 if x < 7 else 1).to_numpy()
    y_preds = pd.Series(y_preds).apply(lambda x: 0 if x < 7 else 1).to_numpy()

    precision, recall, thresholds = precision_recall_curve(y_true, y_preds)

    auc_precision_recall = auc(recall, precision)

    return auc_precision_recall


def compare_metrics(df: pd.DataFrame):
    pass


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(y_true, y_hat):
    r2 = r_squared_error(y_true, y_hat)
    r02 = squared_error_zero(y_true, y_hat)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


if __name__ == '__main__':
    true = np.random.rand(1000)
    preds = np.random.rand(1000)

    print(r2_score(true, preds))
    print(r_squared_error(true, preds))
    print(get_rm2(true,preds))
    print(concordance_index(true,preds))
