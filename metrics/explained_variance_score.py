import numpy as np


def explained_variance_score(Y_ture, Y_pred, multioutput='uniform_average'):
    ev = 1 - np.var(np.array(Y_ture) - np.array(Y_pred), axis=0) / np.var(Y_ture, axis=0)
    return np.sum(ev) if multioutput == 'uniform_average' else ev


if __name__ == '__main__':
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    print(explained_variance_score(y_true, y_pred, multioutput='raw_values'))
