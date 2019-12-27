import numpy as np


class KFold:
    def __init__(self, n_splits: int = 3, shuffle: bool = False):
        self._K = n_splits
        self._shuffle = shuffle  # 洗牌标志
        self._test_size = None  # 测试集尺寸
        self._idx_arr = None  # 索引数组

    def get_n_splits(self):
        '''
        :return: 当前对象的fold数
        '''
        return self._K

    def split(self, X):
        n_samples = len(X)
        self._test_size = n_samples // self._K
        self._idx_arr = np.arange(n_samples)
        if self._shuffle:
            np.random.shuffle(self._idx_arr)

        return ((np.append(self._idx_arr[0:epoch * self._test_size], self._idx_arr[(epoch + 1) * self._test_size:]),
                 self._idx_arr[epoch * self._test_size:(epoch + 1) * self._test_size]) for epoch in range(self._K))


if __name__ == '__main__':
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    kf = KFold()
    for train_index, test_index in kf.split(X):
        print(train_index, test_index)
