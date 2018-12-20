import numpy as np


class LinearRegression:
    def __init__(self, lr=0.00001, batch_size=32, max_iter=1000):
        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.W = None
        self.b = None

    def fit(self, X, Y):
        X = X.copy()
        Y = Y.copy()

        n = X.shape[0]  # 样本数
        m = X.shape[1]  # 特征数
        assert Y.shape[0] == n  # 数据与标签应该相等
        Y = Y.reshape((n, 1))  # 标签，列向量

        self.W = np.random.rand(m).reshape((1, -1))  # 权重，行向量
        self.b = np.ones((1, 1))  # 偏置

        assert Y.shape == (n, 1)

        num_batch = n // self.batch_size

        for epoch in range(self.max_iter):
            for i in range(num_batch + 1):
                start_index = i * self.batch_size
                end_index = (i + 1) * self.batch_size
                if end_index <= n:
                    X_batch = X[start_index:end_index + 1]
                    Y_batch = Y[start_index:end_index + 1]
                else:
                    X_batch = X[start_index:]
                    Y_batch = Y[start_index:]

                Y_hat = X_batch.dot(self.W.T) + self.b
                dW = 2 * (Y_hat - Y_batch).T.dot(X_batch) / n
                db = 2 * (Y_hat - Y_batch).T.dot(np.ones((X_batch.shape[0], 1))) / n
                assert (dW.shape == self.W.shape) & (db.shape == self.b.shape)

                self.W = self.W - self.lr * dW
                self.b = self.b - self.lr * db

    def predict(self, X):
        X = X.copy()
        return np.squeeze(np.dot(X, self.W.T) + self.b)  # 将矩阵压缩成向量，与原始输入Y保持一致


if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    data = load_boston()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    line_reg = LinearRegression()
    line_reg.fit(X_train, Y_train)


    def RMSE(y_true, y_pred):
        return sum((y_true - y_pred) ** 2) ** 0.5


    Y_pred = line_reg.predict(X_test)
    rmse = RMSE(Y_test, Y_pred)
    print(rmse)
