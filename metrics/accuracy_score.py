import numpy as np


def accuracy_score(Y_true, Y_pred, sample_weight=None):
    assert len(Y_true) == len(Y_pred)
    n_samples = len(Y_true)
    sample_weight = np.array([1 / n_samples for _ in range(n_samples)]) if not sample_weight else sample_weight
    return np.sum((np.array(Y_true) == np.array(Y_pred)) * sample_weight)


if __name__ == '__main__':
    y_pred = [0, 2, 3]
    y_true = [0, 1, 3]
    print(accuracy_score(y_true, y_pred))
