import numpy as np
from tree.DecisionTreeClassifier import DecisionTreeClassifier


class AdaBoostClassifier:
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.estimators_ = [DecisionTreeClassifier(max_depth=1) for _ in range(self.n_estimators)]  # 默认基模型为树
        self.estimator_weights_ = [None] * self.n_estimators

    def __update_w(self, w, Y_true, Y_pred):
        weight_err = np.sum(w * (Y_true != Y_pred)) / np.sum(w)  # 加权训练误差
        alpha = np.log(1 / weight_err - 1)    # 根据加权误差计算模型权重
        w = w * np.exp(alpha * (Y_true != Y_pred))
        w = w / np.sum(w)  # 归一化

        return w, alpha

    def fit(self, X_train, Y_train):
        Y_train[Y_train == 0] = -1  # 0转-1
        n_samples, n_features = X_train.shape
        w = np.array([1 / n_samples] * n_samples)  # 初始样本权重
        for i in range(self.n_estimators):
            self.estimators_[i].fit(X_train, Y_train, sample_weight=w)
            cur_pred = self.estimators_[i].predict(X_train)
            w, self.estimator_weights_[i] = self.__update_w(w, Y_train, cur_pred)

    def predict(self, X_test):
        for i in range(self.n_estimators):
            if i == 0:
                Y_pred = self.estimator_weights_[i] * self.estimators_[i].predict(X_test)
            else:
                Y_pred = np.c_[Y_pred, self.estimator_weights_[i] * self.estimators_[i].predict(X_test)]
        Y_pred = np.sign(np.sum(Y_pred, axis=1))
        Y_pred[Y_pred == -1] = 0  # -1转0
        return Y_pred


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, Y = data.data, data.target
    del data

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    ada_clf = AdaBoostClassifier()
    ada_clf.fit(X_train, Y_train)
    Y_pred = ada_clf.predict(X_test)
    print('ada acc:{}'.format(np.sum(Y_pred == Y_test) / len(Y_test)))
