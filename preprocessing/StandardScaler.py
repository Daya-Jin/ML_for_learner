import numpy as np


class StandardScaler:
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None  # 均值
        self.var_ = None  # 方差

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0) if self.with_mean else 0
        self.var_ = np.var(X, axis=0) if self.with_std else 1

    def transform(self, X):
        if self.copy:
            X = np.copy(X)

        return (X - self.mean_) / np.sqrt(self.var_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    scaler = StandardScaler()
    print(scaler.mean_)
    print(scaler.transform(data))
    print(scaler.transform([[2, 2]]))
