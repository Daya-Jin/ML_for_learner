import numpy as np


class StandardScaler:
    def __init__(self, copy: bool = True, with_mean: bool = True, with_std: bool = True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None  # 均值
        self.var_ = None  # 方差

    def fit(self, X):
        X = np.array(X)  # 确保以下运算是基于numpy的
        self.mean_ = np.mean(X, axis=0) if self.with_mean else 0
        self.var_ = np.var(X, axis=0) if self.with_std else 1

    def transform(self, X):
        X = np.array(X)  # 确保以下运算是基于numpy的
        if self.copy:
            X = np.copy(X)

        return (X - self.mean_) / np.sqrt(self.var_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    data = [[0, 0],
            [0, 0],
            [1, 1],
            [1, 1]]
    scaler = StandardScaler()
    print(scaler.mean_)
    print(scaler.fit_transform(data))
    print(scaler.transform([[2, 2]]))
