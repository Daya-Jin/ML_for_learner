import numpy as np


class TruncatedSVD:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.Sigma = None
        self.U = None
        self.VT = None

    def fit(self, X):
        self.U, self.Sigma, self.VT = np.linalg.svd(X)
        self.VT = self.VT.T  # 此句为修复语句，但不明白原因

        top_idx = np.argsort(self.Sigma)[::-1]
        self.Sigma = np.diag(self.Sigma)
        self.U = self.U[top_idx[:self.n_components], :]
        self.VT = self.VT[:, top_idx[:self.n_components]]

    def transform(self, X):
        return np.dot(X, self.VT)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from datasets.dataset import load_wine

    data = load_wine()
    X = data.data
    Y = data.target

    from preprocessing.StandardScaler import StandardScaler

    X = StandardScaler().fit_transform(X)
    svd = TruncatedSVD()
    X_trans = svd.fit_transform(X)

    plt.scatter(X_trans[:, 0], X_trans[:, 1], c=Y)
    plt.show()

    del svd, X_trans
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=2)
    X_trans = svd.fit_transform(X)

    plt.scatter(X_trans[:, 0], X_trans[:, 1], c=Y)
    plt.show()
