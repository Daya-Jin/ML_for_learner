import numpy as np


class KMeans:
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++', max_iter: int = 300, tol: float = 0.0001):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol

        self.cluster_centers_ = None
        self.dist = None
        self.labels_ = None

    def __gen_center(self, X_train):
        n_sample, n_feature = X_train.shape

        if self.init == 'random':
            # 为了在数据范围内产生随机质心，首先计算各特征的统计量
            f_mean = np.mean(X_train, axis=0)
            f_std = np.std(X_train, axis=0)
            self.cluster_centers_ = f_mean + np.random.randn(self.n_clusters, n_feature) * f_std

        elif self.init == 'k-means++':
            # 第一个质心随机选
            idx = np.random.randint(0, n_sample)
            self.cluster_centers_ = [X_train[idx, :]]

            # 选出后面k-1个质心
            for i in range(1, self.n_clusters):
                dist = np.zeros((n_sample, len(self.cluster_centers_)))  # 各样本到质心的距离矩阵
                for cent_idx in range(len(self.cluster_centers_)):
                    dist[:, cent_idx] = np.linalg.norm(
                        X_train - self.cluster_centers_[cent_idx], axis=1)

                dist = np.min(dist, axis=1)  # 所有样本离各质心距离的最小值
                p = dist / np.sum(dist)  # 归一化后的最小距离当做概率进行下一个质心的选取，这里没有计算平方

                next_cent_idx = np.random.choice(n_sample, p=p)
                self.cluster_centers_.append(X_train[next_cent_idx])
            self.cluster_centers_ = np.array(self.cluster_centers_)

    def fit(self, X_train):
        n_sample, n_feature = X_train.shape

        self.__gen_center(X_train)
        self.dist = np.zeros((n_sample, self.n_clusters))

        cent_pre = np.zeros(self.cluster_centers_.shape)
        cent_move = np.linalg.norm(self.cluster_centers_ - cent_pre)

        epoch = 0
        from copy import deepcopy
        while epoch < self.max_iter and cent_move > self.tol:
            epoch += 1

            # 首先计算每个样本离每个质心的距离
            for i in range(self.n_clusters):
                self.dist[:, i] = np.linalg.norm(X_train - self.cluster_centers_[i], axis=1)

            # 样本对应的类别为距离最近的质心
            self.labels_ = np.argmin(self.dist, axis=1)

            cent_pre = deepcopy(self.cluster_centers_)

            # 计算每个类别下的均值坐标，更新质心
            for i in range(self.n_clusters):
                self.cluster_centers_[i] = np.mean(X_train[self.labels_ == i], axis=0)

            cent_move = np.linalg.norm(self.cluster_centers_ - cent_pre)

    def predict(self, X_test):
        n_sample = X_test.shape[0]
        dist_test = np.zeros((n_sample, self.n_clusters))

        for i in range(self.n_clusters):
            dist_test[:, i] = np.linalg.norm(X_test - self.cluster_centers_[i], axis=1)
        clus_pred = np.argmin(dist_test, axis=1)

        return clus_pred


if __name__ == '__main__':
    import numpy as np

    data_1 = np.random.randn(200, 2) + [1, 1]
    data_2 = np.random.randn(200, 2) + [4, 4]
    data_3 = np.random.randn(200, 2) + [7, 1]
    data = np.concatenate((data_1, data_2, data_3), axis=0)

    from sklearn.model_selection import train_test_split

    X_train, X_test = train_test_split(data, test_size=0.2)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)

    import matplotlib.pyplot as plt

    plt.clf()
    plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.5, c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', c='k')
    plt.show()

    clus_pred = kmeans.predict(X_test)
    plt.clf()
    plt.scatter(X_test[:, 0], X_test[:, 1], alpha=0.5, c=clus_pred)
    plt.show()
