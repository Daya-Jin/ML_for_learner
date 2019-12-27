import numpy as np


def train_test_split(*arrays, test_size: float = 0.25,
                     random_state: int = None, shuffle: bool = False):
    if not arrays or len(arrays) > 2:
        raise ValueError('params {} is illegal'.format(arrays))

    arr_1 = arrays[0]
    n_samples = len(arr_1)
    train_samples = int(n_samples * (1 - test_size))  # 训练样本数

    arr_2 = arrays[1] if len(arrays) > 1 else None
    if arr_2 is not None and n_samples != len(arr_2):
        raise ValueError('two arrays not equal')

    shuffle_idx = np.random.permutation(n_samples)
    if arr_2 is not None:
        return (arr_1[shuffle_idx][:train_samples], arr_1[shuffle_idx][train_samples:],
                arr_2[shuffle_idx][:train_samples], arr_2[shuffle_idx][train_samples:])
    else:
        return arr_1[shuffle_idx][:train_samples], arr_1[shuffle_idx][train_samples:]


if __name__ == '__main__':
    from datasets.dataset import load_boston

    data = load_boston()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
