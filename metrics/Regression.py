import numpy as np


def explained_variance_score(Y_ture, Y_pred, multioutput: str = 'uniform_average'):
    ev = 1 - np.var(np.array(Y_ture) - np.array(Y_pred), axis=0) / np.var(Y_ture, axis=0)
    return np.sum(ev) if multioutput == 'uniform_average' else ev


def mean_absolute_error(Y_true, Y_pred, multioutput: str = 'uniform_average'):
    mae = np.sum(np.abs(np.array(Y_true) - np.array(Y_pred)) / len(Y_true), axis=0)
    return np.sum(mae) if multioutput == 'uniform_average' else mae


def mean_squared_error(Y_true, Y_pred, multioutput: str = 'uniform_average'):
    mse = np.sum(np.square(np.array(Y_true) - np.array(Y_pred)) / len(Y_true), axis=0)
    return np.sum(mse) if multioutput == 'uniform_average' else mse


def r2_score(Y_ture, Y_pred, multioutput: str = 'uniform_average'):
    r2 = 1 - np.sum(np.square(np.array(Y_ture) - np.array(Y_pred)), axis=0) / np.sum(
        np.square(np.array(Y_ture) - np.mean(Y_ture, axis=0)), axis=0)
    return np.sum(r2) if multioutput == 'uniform_average' else r2
