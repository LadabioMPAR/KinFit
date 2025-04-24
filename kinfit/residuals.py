import numpy as np

def calculate_residuals(y_true, y_pred):
    return np.array(y_true) - np.array(y_pred)
