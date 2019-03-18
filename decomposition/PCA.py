class PCA:
    def __init__(self, n_components=None):
        self.n_components_ = n_components
        self.top_vec = None
        self.covar = None

    def fit(self, X):
        # 数据信息
        n_sample, n_feature = X.shape
        if not self.n_components_:    # 如果未设置维数则设一个较大值
            self.n_components_ = min(n_sample, n_feature) - 1

        self.covar = np.cov(X.T)
        eigval, eigvec = np.linalg.eig(self.covar)
        top_idx = np.argsort(eigval)[::-1]
        self.top_vec = eigvec[:, top_idx[:self.n_components_]]

    def transform(self, X):
        return X.dot(self.top_vec)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_wine

    data = load_wine()
    X = data.data
    Y = data.target

    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()
