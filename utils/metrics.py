import numpy as np
def MAE(pred, true):
    return np.mean(np.abs(true - pred))
def MSE(pred, true):
    return np.mean((true - pred) ** 2)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    return mae, mse