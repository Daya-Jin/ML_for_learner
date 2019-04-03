import numpy as np


class TruncatedSVD:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.Sigma = None
        self.U = None
        self.VT = None

    def fit(self, X):
        self.U, self.Sigma, self.VT = np.linalg.svd(X)
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
    from sklearn.datasets import make_circles
    import matplotlib.pyplot as plt

    X, Y = make_circles(factor=0.5, random_state=0, noise=0.05)

    svd = TruncatedSVD()
    X_trans = svd.fit_transform(X)

    fig, axs = plt.subplots(1, 2, figsize=(5, 5))
    axs[0].scatter(X[:, 0], X[:, 1], c=Y)
    axs[1].scatter(X_trans[:, 0], X_trans[:, 1], c=Y)

    plt.show()
