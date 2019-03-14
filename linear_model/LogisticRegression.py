import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.0001, threshold=0.5, max_iter=2000):
        self.lr = lr
        self.threshold = threshold
        self.max_iter = max_iter
        self.W = None
        self.b = None

    def __sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def fit(self, X_train, Y_train):
        Y_train = Y_train.reshape((-1, 1))

        n_sample = X_train.shape[0]
        n_feature = X_train.shape[1]

        self.W = 0.001 * np.random.rand(n_feature).reshape((n_feature, 1))  # 权重
        self.b = 0  # 偏置

        for i in range(self.max_iter):
            Y_hat = self.__sigmoid(np.dot(X_train, self.W) + self.b)

            dW = X_train.T.dot(Y_hat - Y_train) / n_sample
            db = np.sum(Y_hat - Y_train) / n_sample

            self.W -= self.lr * dW
            self.b -= self.lr * db

            # if i % 200 == 200-1:
            #     Y_hat = self.__sigmoid(np.dot(X_train, self.W) + self.b)
            #     L = np.sum(-np.dot(Y_train.T, np.log(Y_hat)) - np.dot((1 - Y_train).T, np.log(1 - Y_hat))) / n_sample
            #     print(L, end=' ')

    def predict(self, X_test):
        return np.squeeze(np.where(self.__sigmoid(np.dot(X_test, self.W) + self.b) > self.threshold, 1, 0))


def ACC(Y_true,Y_pred):
    return np.sum(Y_true==Y_pred)/len(Y_true)

if __name__=='__main__':
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    lr_model=LogisticRegression()
    lr_model.fit(X_train,Y_train)

    Y_pred=lr_model.predict(X_test)

    print('ACC:{}'.format(ACC(Y_test,Y_pred)))