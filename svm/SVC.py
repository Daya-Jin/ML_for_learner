import numpy as np
from svm.utils import rbf, clip


class svm_data:
    '''
    训练时需要用到的辅助数据缓存
    '''

    def __init__(self, X_train):
        n_samples = len(X_train)
        self.err = np.zeros((n_samples, 2))  # 误差缓存
        self.K_mat = np.zeros((n_samples, n_samples))  # 核矩阵缓存
        for i in range(n_samples):
            self.K_mat[:, i] = rbf(X_train, X_train[i, :])


class SVC:
    def __init__(self, C: float = 1.0, tol: float = 0.001, max_iter: int = -1):
        # 训练参数
        self.C = C
        self.tol = tol
        self.max_iter = max_iter if max_iter > 0 else -1e5

        # 预测时需要用到的数据
        self.X_train = None
        self.Y_train = None
        self.lambdas = None
        self.b = 0

    def __compute_kth_err(self, k: int, DS):
        '''
        计算第k个训练样本的训练误差
        :param Y_train:
        :param k:
        :param DS:
        :return:
        '''
        y_pred = np.dot((self.lambdas * self.Y_train).T, DS.K_mat[k, :]) + self.b
        return y_pred - self.Y_train[k]

    def __update_kth_err(self, k: int, DS):
        '''
        更新误差缓存
        :param Y_train:
        :param k:
        :param errs:
        :return:
        '''
        err_k = self.__compute_kth_err(k, DS)
        DS.err[k] = [1, err_k]

    def __select_2nd_lambda(self, i, err_i, DS):
        '''
        选取第二个lambda
        :param DS:
        :param i: 第一个lambda的下标
        :param err_i:
        :return:
        '''
        max_delte_err = 0
        err_j = 0
        j = -1  # 默认j为最后一个

        DS.err[i] = [1, err_i]
        valid_err_idxs = np.nonzero(DS.err[:, 0])[0]  # 标记位非零的idxs

        if len(valid_err_idxs) > 1:  # 如果已有误差缓存
            for idx in valid_err_idxs:
                if idx == i:
                    continue
                else:
                    err_k = self.__compute_kth_err(idx, DS)
                    delta_err = abs(err_i - err_k)
                    if delta_err > max_delte_err:
                        j = idx
                        err_j = err_k
                        max_delte_err = delta_err
            return j, err_j

        else:  # 如果没有误差缓存，则只能随机选第二个lambda
            j = i
            n_samples = len(self.X_train)
            while j == i:
                j = np.random.randint(n_samples)
            return j, self.__compute_kth_err(j, DS)

    def __optimize_lambda(self, i, DS):
        '''
        成对优化lambda
        :param i:
        :param DS:
        :return: 是否做过优化
        '''
        err_i = self.__compute_kth_err(i, DS)
        if (self.Y_train[i] * err_i < -self.tol and self.lambdas[i] < self.C) or (
                self.Y_train[i] * err_i > self.tol and self.lambdas[i] > 0):
            j, err_j = self.__select_2nd_lambda(i, err_i, DS)  # 以优化方式选取第二个lambda
            lambda_i_pre, lambda_j_pre = self.lambdas[i].copy(
            ), self.lambdas[j].copy()  # 保存两lambda的旧值

            # 计算上下界
            if self.Y_train[i] != self.Y_train[j]:
                L = max(0, self.lambdas[j] - self.lambdas[i])
                H = min(self.C, self.C + self.lambdas[j] - self.lambdas[i])
            else:
                L = max(0, self.lambdas[j] + self.lambdas[i] - self.C)
                H = min(self.C, self.lambdas[j] + self.lambdas[i])
            if L == H:
                return 0

            eta = 2 * DS.K_mat[i, j] - DS.K_mat[i, i] - DS.K_mat[j, j]
            if eta >= 0:
                return 0

            # 优化lambda_j
            self.lambdas[j] -= self.Y_train[j] * (err_i - err_j) / eta
            self.lambdas[j] = clip(self.lambdas[j], L, H)
            self.__update_kth_err(j, DS)  # 更新缓存
            delta_lambda_j = self.lambdas[j] - lambda_j_pre
            if abs(delta_lambda_j) < 1e-5:  # 更新量太小则放弃
                return 0

            # 优化lambda_i
            self.lambdas[i] -= self.Y_train[i] * self.Y_train[j] * delta_lambda_j
            self.__update_kth_err(i, DS)  # 更新缓存
            delta_lambda_i = self.lambdas[i] - lambda_i_pre

            # 偏移量
            b_i = self.b - err_i - self.Y_train[i] * delta_lambda_i * DS.K_mat[i, i] - self.Y_train[
                j] * delta_lambda_j * \
                  DS.K_mat[i, j]
            b_j = self.b - err_j - self.Y_train[i] * delta_lambda_i * DS.K_mat[i, j] - self.Y_train[
                j] * delta_lambda_j * \
                  DS.K_mat[j, j]
            if 0 < self.lambdas[i] < self.C:
                self.b = b_i
            elif 0 < self.lambdas[j] < self.C:
                self.b = b_j
            else:
                self.b = (b_i + b_j) / 2
            return 1
        return 0

    def fit(self, X_train, Y_train):
        n_samples, n_feature = X_train.shape
        Y_train[Y_train == 0] = -1
        DS = svm_data(X_train)  # 生成缓存数据

        self.X_train = X_train
        self.Y_train = Y_train
        self.lambdas = np.zeros(n_samples)

        iter_cnt = 0
        full_set_flag = True  # 遍历所有lambda的标记位，第一次循环或找不到非边界样本时设为True
        lambda_changed_cnt = 0

        while iter_cnt < self.max_iter and (lambda_changed_cnt > 0 or full_set_flag):
            lambda_changed_cnt = 0
            if full_set_flag:
                for i in range(n_samples):  # 遍历选取第一个lambda
                    tmp = self.__optimize_lambda(i, DS)
                    lambda_changed_cnt += tmp
                iter_cnt += 1
            else:  # 存在非边界样本时
                non_bound_idxs = np.nonzero((self.lambdas > 0) * (self.lambdas < self.C))[0]  # 非边界样本，lambda大于0小于C
                for idx in non_bound_idxs:  # 在非边界样本中选取第一个lambda
                    tmp = self.__optimize_lambda(idx, DS)
                    lambda_changed_cnt += tmp
                iter_cnt += 1

            if full_set_flag:  # 如果上轮以遍历方式选取第一个lambda，则下次不使用遍历方式
                full_set_flag = False
            elif lambda_changed_cnt == 0:  # 如果一轮迭代没有做出优化，则下次尝试使用遍历方式
                full_set_flag = True
        del DS  # 释放缓存

    def predict(self, X_test):
        Y_pred = 0
        for i in range(len(self.X_train)):
            Y_pred += self.lambdas[i] * self.Y_train[i] * rbf(X_test, self.X_train[i, :])
        Y_pred += self.b
        Y_pred[Y_pred < 0] = 0
        Y_pred[Y_pred > 0] = 1
        return Y_pred


if __name__ == '__main__':
    from datasets.dataset import load_breast_cancer

    data = load_breast_cancer()
    X, Y = data.data, data.target
    del data

    from model_selection.train_test_split import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    svc = SVC(max_iter=50)
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    del svc
    print('acc:{}'.format(np.sum(Y_pred == Y_test) / len(Y_test)))

    from sklearn.svm import SVC

    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    print('sklearn acc:{}'.format(np.sum(Y_pred == Y_test) / len(Y_test)))
