import numpy as np


class LinearDiscriminantAnalysis:
    def __init__(self, n_components: int = 2):
        '''

        :param n_components: 降维的维数
        '''
        self.n_components = n_components
        self.means_ = None  # 类均值
        self.xbar_ = None  # 全局均值
        self.classes_ = None  # 类别数组
        self.top_eig_vec = None  # 前n个特征向量

    def fit(self, X_train, Y_train):
        n_samples, n_features = X_train.shape
        self.classes_ = np.unique(Y_train)

        self.xbar_ = np.mean(X_train, axis=0)  # 数据均值，向量

        self.means_ = list()  # 类均值向量，(K,n_feature)
        for k in self.classes_:
            self.means_.append(np.mean(X_train[Y_train == k], axis=0))
        self.means_ = np.array(self.means_)

        n_k = list()  # 类别计数

        S_w = np.zeros((n_features, n_features))  # 类内散度矩阵
        for k in self.classes_:
            n_k.append(len(X_train[Y_train == k]))
            tmp = X_train[Y_train == k] - self.means_[k]
            S_w += np.dot(tmp.T, tmp)

        S_b = np.dot(n_k * (self.means_ - self.xbar_).T, (self.means_ - self.xbar_))  # 类间散度矩阵

        eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(S_w), S_b))  # 注意特征值与特征向量可能出现复数情况
        top_idx = np.argsort(eigval)[::-1]  # 特征值的排序索引
        self.top_eig_vec = eigvec[:, top_idx[:self.n_components]]  # 取前n个特征向量

    def transform(self, X_train):
        return np.dot(X_train, self.top_eig_vec.real)  # 只取特征向量的实部做运算

    def fit_transform(self, X_train, Y_train):
        self.fit(X_train, Y_train)
        return self.transform(X_train)


if __name__ == '__main__':
    from datasets.dataset import load_wine

    data = load_wine()
    X, Y = data.data, data.target

    from preprocessing.StandardScaler import StandardScaler

    X = StandardScaler().fit_transform(X)

    from model_selection.train_test_split import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_trans = lda.fit_transform(X_train, Y_train)

    import matplotlib.pyplot as plt

    plt.scatter(X_trans[:, 0], X_trans[:, 1], c=Y_train)
    plt.show()

    # 对比scikit-learn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_trans = lda.fit_transform(X_train, Y_train)
    plt.clf()
    plt.scatter(X_trans[:, 0], X_trans[:, 1], c=Y_train)
    plt.show()
