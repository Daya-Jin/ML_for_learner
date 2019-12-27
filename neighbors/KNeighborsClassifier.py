from collections import Counter
from scipy.spatial import KDTree


class KNN:
    def __init__(self, n_neighbors: int = 5):
        self.Y_train = None
        self.k = n_neighbors
        self.kd_tree = None

    def fit(self, X_train, Y_train):
        self.Y_train = Y_train
        self.kd_tree = KDTree(X_train)

    def __get_nb_of_one(self, x_test):
        dists, idxs = self.kd_tree.query(x_test, self.k)
        return zip(self.Y_train[idxs], dists)

    def __vote(self, neighbors):
        counter = Counter()
        for label, dist in neighbors:
            counter[label] += 1 / (dist + 1)  # 首位(标签)计数，权重为距离的倒数
        return counter.most_common(1)[0][0]

    def predict(self, X_test):
        Y_pred = []
        for x_test in X_test:
            neighbors = self.__get_nb_of_one(x_test)
            Y_pred.append(self.__vote(neighbors))
        return np.array(Y_pred)


if __name__ == "__main__":
    import numpy as np
    from datasets.dataset import load_breast_cancer
    from model_selection.train_test_split import train_test_split

    data = load_breast_cancer()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    knn = KNN()
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    print('acc:{}'.format(np.sum(Y_pred == Y_test) / len(Y_test)))

    del knn, Y_pred
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    print('sklearn acc:{}'.format(np.sum(Y_pred == Y_test) / len(Y_test)))
