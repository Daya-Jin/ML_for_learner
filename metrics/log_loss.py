import numpy as np


def log_loss(Y_true, Y_pred, eps=1e-15):
    Y_true = np.array(Y_true).astype(float)
    Y_pred = np.array(Y_pred).astype(float)

    # 交叉熵在0，1处无定义，需要做截断
    L = eps
    H = 1 - eps
    Y_true[Y_true > H] = H
    Y_true[Y_true < L] = L
    Y_pred[Y_pred > H] = H
    Y_pred[Y_pred < L] = L

    return -np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))


if __name__ == '__main__':
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 0]
    print(log_loss(y_true, y_pred))
