import numpy as np


class LinearRegression:
    def __init__(self, lr=0.000001, mini_batch=None, batch_size=32, max_iter=2000):
        self.lr = lr
        self.mini_batch = mini_batch
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.W = None
        self.b = None

    def fit(self, X, Y):
        Y = Y.reshape((-1, 1))

        n_sample = X.shape[0]  # 样本数
        n_feature = X.shape[1]  # 特征数

        self.W = np.random.randn(n_feature).reshape((n_feature, 1))  # 权重
        self.b = 1  # 偏置

        assert Y.shape == (n_sample, 1)

        # 判断batch_size参数的有效性
        self.batch_size = self.batch_size if self.mini_batch else n_sample
        num_batch = n_sample // self.batch_size

        for epoch in range(self.max_iter):
            #### mini-batch ####
            for i in range(num_batch + 1):  # 考虑到不能整除的情况，多循环一次
                start_index = i * self.batch_size
                end_index = (i + 1) * self.batch_size
                if start_index < n_sample:
                    # 切片操作不会引发越界
                    X_batch = X[start_index:end_index + 1]
                    Y_batch = Y[start_index:end_index + 1]

                    n_batch = X_batch.shape[0]
                    Y_hat_batch = X_batch.dot(self.W) + self.b
                    dW = 2 * X_batch.T.dot(Y_hat_batch - Y_batch) / n_batch
                    db = 2 * np.sum(Y_hat_batch - Y_batch) / n_batch
                    assert dW.shape == self.W.shape

                    self.W -= self.lr * dW
                    self.b -= self.lr * db

    def predict(self, X):
        # 将矩阵压缩成向量，与原始输入Y保持一致
        return np.squeeze(np.dot(X, self.W) + self.b)


def RMSE(y_true, y_pred):
    return sum((y_true - y_pred) ** 2) ** 0.5 / len(y_true)


if __name__ == "__main__":
    from datasets.dataset import load_boston
    from model_selection.train_test_split import train_test_split

    data = load_boston()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    line_reg = LinearRegression(max_iter=2000)
    line_reg.fit(X_train, Y_train)

    Y_pred = line_reg.predict(X_test)
    rmse = RMSE(Y_test, Y_pred)
    print(rmse)
