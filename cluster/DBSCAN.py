import numpy as np
from metrics.pairwise.euclidean_distances import euclidean_distances


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        '''
        :param eps: 邻域距离
        :param min_samples: 形成类簇所需的最小样本数
        '''
        self.eps = eps
        self.min_samples = min_samples
        self.core_sample_indices_ = list()
        self.components_ = None
        self.labels_ = None

    def fit(self, X):
        n_samples = len(X)

        dist_mat = euclidean_distances(X)  # pair-wise距离矩阵

        density_arr = np.array([np.sum(dist <= self.eps) for dist in dist_mat])  # 密度数组

        visited_arr = [False for _ in range(n_samples)]  # 访问标记数组

        k = -1  # 初始类别
        self.labels_ = np.array([-1 for _ in range(n_samples)])

        for sample_idx in range(n_samples):
            if visited_arr[sample_idx]:  # 跳过已访问样本
                continue

            visited_arr[sample_idx] = True

            # 跳过噪声样本与边界样本
            if density_arr[sample_idx] == 1 or density_arr[sample_idx] < self.min_samples:
                continue

            else:
                cores = [idx for idx in range(n_samples) if
                         dist_mat[idx, sample_idx] <= self.eps and density_arr[idx] >= self.min_samples]
                k += 1
                self.labels_[sample_idx] = k
                self.core_sample_indices_.append(sample_idx)

                while cores:
                    cur_core = cores.pop(0)
                    if not visited_arr[cur_core]:
                        self.core_sample_indices_.append(cur_core)
                        visited_arr[cur_core] = True
                        self.labels_[cur_core] = k

                        neighbors = [idx for idx in range(n_samples) if dist_mat[idx, cur_core] <= self.eps]
                        neighbor_cores = [idx for idx in neighbors if
                                          idx not in cores and density_arr[idx] >= self.min_samples]
                        neighbor_boards = [idx for idx in neighbors if density_arr[idx] < self.min_samples]

                        cores.extend(neighbor_cores)

                        for idx in neighbor_boards:
                            if self.labels_[idx] == -1:
                                self.labels_[idx] = k

        # 更新类属性
        self.core_sample_indices_ = np.sort(np.array(self.core_sample_indices_))
        self.components_ = X[self.core_sample_indices_]

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs
    from preprocessing.StandardScaler import StandardScaler

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, Y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                      random_state=0)
    X = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.3, min_samples=10)
    db.fit(X)

    import matplotlib.pyplot as plt

    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=db.labels_)
    plt.show()

    # 对比sklearn
    del db
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.3, min_samples=10)
    db.fit(X)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=db.labels_)
    plt.show()
