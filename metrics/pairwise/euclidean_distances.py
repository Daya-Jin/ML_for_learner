import numpy as np


def euclidean_distances(X, Y=None, Y_norm_squared=None, X_norm_squared=None):
    X = np.array(X)
    Y = np.array(Y) if Y else X  # 若未指定Y则令其为X

    dist_mat = np.dot(X, Y.T)

    X_squared = np.sum(np.square(X), axis=1).reshape((dist_mat.shape[0], -1))
    Y_squared = np.sum(np.square(Y), axis=1).reshape((-1, dist_mat.shape[1]))
    squared_dist = X_squared - 2 * dist_mat + Y_squared
    squared_dist[squared_dist < 0] = 0  # 在某些数据下可能出现负数，需要做截断处理

    return np.sqrt(squared_dist)


if __name__ == '__main__':
    X = [[0, 1], [1, 1]]
    Y = [[0, 0]]
    print(euclidean_distances(X))
    print(euclidean_distances(X, Y))
