import numpy as np

# 欧氏距离
def E_dist(a:list,b:list):
    a=np.array(a)
    b=np.array(b)
    return np.linalg.norm(a-b)

from collections import Counter


class KNN:
    def __init__(self, n_neighbors=5, metric=E_dist):
        self.X_train = None
        self.Y_train = None
        self.k = n_neighbors
        self.metric = metric

    def fit(self, X_train, Y_train):
        # 模型不改变输入数据，所以这里等号赋值没有问题
        self.X_train = X_train
        self.Y_train = Y_train

    def __get_nb_of_one(self, x_test):
        dists = []

        for idx in range(len(self.X_train)):
            cur_dist = self.metric(x_test, self.X_train[idx])
            dists.append((self.Y_train[idx], cur_dist))    # 首位为标签，末位为距离
        dists.sort(key=lambda x: x[1])    # 按照距离排序

        return dists[:self.k]

    def __vote(self, neighbors):
        counter = Counter()
        for idx in range(len(neighbors)):
            dist = neighbors[idx][1]
            label = neighbors[idx][0]
            counter[label] += 1/(dist+1)    # 首位(标签)计数，权重为距离的倒数
        return counter.most_common(1)[0][0]

    def predict(self, X_test):
        Y_pred = []
        for x_test in X_test:
            neighbors = self.__get_nb_of_one(x_test)
            Y_pred.append(self.__vote(neighbors))
        return np.array(Y_pred)