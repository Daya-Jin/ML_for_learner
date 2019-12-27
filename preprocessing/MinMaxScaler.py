import numpy as np


class MinMaxScaler:
    def __init__(self, feature_range: tuple = (0, 1), copy: bool = True):
        '''

        :param feature_range: 缩放后的范围
        :param copy:
        '''
        self.feature_range = feature_range
        self._scale = None  # 缩放因子
        self.min_ = None
        self.data_min_ = None  # 数据中每一列的最小值
        self.data_max_ = None  # 数据中每一列的最大值

    def fit(self, X):
        self.data_min_, self.data_max_ = np.min(X, axis=0), np.max(X, axis=0)

        self._scale = (self.feature_range[1] - self.feature_range[0]) / (self.data_max_ - self.data_min_)  # 缩放因子
        self.min_ = self.feature_range[0] - self._scale * self.data_min_  # 调整偏差

    def transform(self, X):
        return self._scale * X + self.min_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    scaler = MinMaxScaler()
    print(scaler.fit_transform(data))
    print(scaler.transform([[2, 2]]))
