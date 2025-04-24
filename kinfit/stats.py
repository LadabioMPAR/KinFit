import numpy as np

def r_squared(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def aic(n, rss, k):
    return n * np.log(rss / n) + 2 * k

def bic(n, rss, k):
    return n * np.log(rss / n) + k * np.log(n)
