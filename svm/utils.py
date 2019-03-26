import numpy as np


def rbf(mat, array, sigma=1):
    '''
    RBF核变换函数，返回矩阵与向量的核矩阵
    :param mat:
    :param array:
    :param sigma:
    :return:
    '''
    n_samples, n_feature = mat.shape
    K_mat = np.zeros(n_samples)

    for i in range(n_samples):
        K_mat[i] = np.dot((mat[i, :] - array), (mat[i, :] - array).T)
    K_mat = np.exp(-K_mat / 2 / (sigma ** 2))
    return K_mat


def clip(x, L, H):
    '''
    截断函数
    :param x: 需要做截断处理的值
    :param L: 下界
    :param H: 上界
    :return: 截断值
    '''
    if x < L:
        x = L
    if x > H:
        x = H
    return x
