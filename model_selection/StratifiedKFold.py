import numpy as np


class StratifiedKFold:
    def __init__(self, n_splits: int = 3, shuffle: bool = False):
        self._K = n_splits
        self._shuffle = shuffle  # 洗牌标志

    def get_n_splits(self):
        '''
        :return: 当前对象的fold数
        '''
        return self._K

    def _split_gen(self, y):
        '''
        生成器函数
        :param y: label array
        :return:
        '''
        n_samples = len(y)
        for epoch in range(self._K):
            # 空索引，等待扩展
            train_idx = np.array(list()).astype(int)
            test_idx = np.array(list()).astype(int)

            # 遍历y所有可能的取值并扩展索引
            for y_val in np.unique(y):
                idxs = np.arange(n_samples)[y == y_val]  # 当前类别下的全部索引
                test_size = len(idxs) // self._K
                train_idx = np.append(train_idx, np.append(
                    idxs[0:epoch * test_size], idxs[(epoch + 1) * test_size:]))
                test_idx = np.append(
                    test_idx, idxs[epoch * test_size:(epoch + 1) * test_size])
            yield (train_idx, test_idx)

    def split(self, X, y):
        assert len(X) == len(y)
        return self._split_gen(y)


if __name__ == '__main__':
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([0, 0, 1, 1])
    skf = StratifiedKFold(n_splits=2)

    for train_index, test_index in skf.split(X, y):
        print(train_index, test_index)
