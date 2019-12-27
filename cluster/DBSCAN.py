import numpy as np
from scipy.spatial import KDTree


class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
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

        kd_tree = KDTree(X)  # 构造KD树

        density_arr = np.array([len(kd_tree.query_ball_point(x, self.eps)) for x in X])  # 密度数组

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

            # 核心对象
            else:
                # 找出邻域中的所有核心对象，包括自身
                cores = [idx for idx in kd_tree.query_ball_point(X[sample_idx], self.eps) if
                         density_arr[idx] >= self.min_samples]
                k += 1
                self.labels_[sample_idx] = k
                self.core_sample_indices_.append(sample_idx)

                while cores:
                    cur_core = cores.pop(0)
                    if not visited_arr[cur_core]:
                        self.core_sample_indices_.append(cur_core)
                        visited_arr[cur_core] = True
                        self.labels_[cur_core] = k

                        neighbors = kd_tree.query_ball_point(X[cur_core], self.eps)
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
