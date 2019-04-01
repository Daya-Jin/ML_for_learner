import numpy as np
from tree.DecisionTreeClassifier import DecisionTreeClassifier
from scipy import stats  # 用于求众数


class RandomForestClassifier:
    def __init__(self, n_estimators=5, min_samples_split=5, min_samples_leaf=5, min_impurity_decrease=0.0):
        '''
        :param n_estimators: 子树的数量
        :param min_samples_split: 最小分割样本数，用于传递给子CART树
        :param min_samples_leaf: 最小叶节点样本数，用于传递给子CART树
        :param min_impurity_decrease: 最小增益，用于传递给子CART树
        '''
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.estimators_ = list()

    def __RandomPatches(self, data):
        '''
        实现RandomPatches
        :param data: 用于抽样的数据
        :return: 随机抽样得到的子数据
        '''
        n_samples, n_features = data.shape
        n_features -= 1
        sub_data = np.copy(data)

        random_f_idx = np.random.choice(
            n_features, size=int(np.sqrt(n_features)), replace=False)
        mask_f_idx = [i for i in range(n_features) if i not in random_f_idx]  # 未抽到的特征idx

        random_data_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        sub_data = sub_data[random_data_idx]
        sub_data[:, mask_f_idx] = 0  # 未抽到的特征列全部置零
        return sub_data

    def __RF_Clf(self, data):
        '''
        串行生成随机森林
        '''
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf,
                                          min_impurity_decrease=self.min_impurity_decrease)
            sub_data = self.__RandomPatches(data)
            tree.fit(sub_data[:, :-1], sub_data[:, -1])
            self.estimators_.append(tree)

    def fit(self, X_train, Y_train):
        data = np.c_[X_train, Y_train]
        self.__RF_Clf(data)
        del data

    def predict(self, X_test):
        raw_pred = np.array([tree.predict(X_test) for tree in self.estimators_]).T
        return np.array([stats.mode(y_pred)[0][0] for y_pred in raw_pred])


if __name__ == '__main__':
    from datasets.dataset import load_breast_cancer

    data = load_breast_cancer()
    X, Y = data.data, data.target
    del data

    from model_selection.train_test_split import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, Y_train)
    Y_pred = rf_clf.predict(X_test)

    print('rf acc:{}'.format(np.sum(Y_pred == Y_test) / len(Y_test)))
