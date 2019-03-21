import numpy as np
from scipy import stats

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=5, min_samples_leaf=5, min_impurity_decrease=0.0):
        '''
        :param min_samples_split: 分裂所需的最小样本数
        :param min_samples_leaf: 叶节点中的最小样本数
        :param min_impurity_decrease: 分裂需要满足的最小增益
        '''
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def __Gini(self, data, y_idx=-1):
        '''
        :param data:
        :param y_idx: 目标值在data中的列索引
        :return: y的Gini
        '''
        K = np.unique(data[:, y_idx])
        n_sample = len(data)
        gini_idx = 1 - \
                   np.sum([np.square(len(data[data[:, y_idx] == k]) / n_sample) for k in K])

        return gini_idx

    def __BinSplitData(self, data, f_idx, f_val):
        '''
        划分数据集
        :param data:
        :param f_idx: 特征的列索引
        :param f_val: 特征的取值
        :return: 分割后的左右数据子集
        '''
        data_left = data[data[:, f_idx] <= f_val]
        data_right = data[data[:, f_idx] > f_val]
        return data_left, data_right

    def __Test(self, data):
        '''
        对数据做test，找到最佳分割特征与特征值
        :param data:
        :return: best_f_idx, best_f_val，前者为空时代表无法分裂
        '''
        n_sample, n_feature = data.shape

        # 数据量小于阈值则直接返回叶节点，数据已纯净也返回叶节点
        if n_sample < self.__min_samples_split or len(np.unique(data[:, -1])) == 1:
            return None, stats.mode(data[:, -1])[0][0]

        Gini_before = self.__Gini(data)  # 分裂前的Gini
        best_gain = 0
        best_f_idx = None
        best_f_val = stats.mode(data[:, -1])[0][0]  # 默认分割值设为目标众数，当找不到分割点时返回该值作为叶节点

        # 遍历所有特征与特征值
        for f_idx in range(n_feature - 1):
            for f_val in np.unique(data[:, f_idx]):
                data_left, data_right = self.__BinSplitData(data, f_idx, f_val)  # 二分数据

                # 分割后的分支样本数小于阈值则放弃分裂
                if len(data_left) < self.__min_samples_leaf or len(data_right) < self.__min_samples_leaf:
                    continue

                # 分割后的加权Gini
                Gini_after = len(data_left) / n_sample * self.__Gini(data_left) + \
                             len(data_right) / n_sample * self.__Gini(data_right)
                gain = Gini_before - Gini_after  # Gini的减小量为增益

                # 分裂后的增益小于阈值或小于目前最大增益则放弃分裂
                if gain < self.__min_impurity_decrease or gain < best_gain:
                    continue
                else:
                    # 否则更新最大增益与最佳分裂位置
                    best_gain = gain
                    best_f_idx, best_f_val = f_idx, f_val

        # 返回一个最佳分割特征与最佳分割点，当无法分裂时best_f_idx为空
        return best_f_idx, best_f_val

    def __CART(self, data):
        '''
        生成CART树
        :param data: 用于生成树的数据集，包含X，Y
        :return: CART树
        '''
        # 首先是做test，数据集的质量由Test函数来保证并提供反馈
        best_f_idx, best_f_val = self.__Test(data)

        if best_f_idx is None:  # f_idx为空表示需要生成叶节点
            return best_f_val

        tree = dict()
        tree['cut_f'] = best_f_idx
        tree['cut_val'] = best_f_val

        data_left, data_right = self.__BinSplitData(data, best_f_idx, best_f_val)
        tree['left'] = self.__CART(data_left)
        tree['right'] = self.__CART(data_right)

        return tree

    def __predict_one(self, x_test, tree):
        '''
        预测单个样本
        :param x_test: 单个样本向量
        :param tree: 训练生成树
        :return: 单个预测值
        '''
        if isinstance(tree, dict):  # 非叶节点才做左右判断
            cut_f_idx, cut_val = tree['cut_f'], tree['cut_val']
            sub_tree = tree['left'] if x_test[cut_f_idx] <= cut_val else tree['right']
            return self.__predict_one(x_test, sub_tree)
        else:  # 叶节点则直接返回值
            return tree

    def fit(self, X_train, Y_train):
        data = np.c_[X_train, Y_train]  # 拼接特征与目标，便于操作
        self.tree = self.__CART(data)  # 生成CART树即完成训练

    def predict(self, X_test):
        return [self.__predict_one(x_test, self.tree) for x_test in X_test]


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, Y = data.data, data.target
    del data

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, Y_train)
    Y_pred = tree_clf.predict(X_test)
    print('acc_before_tuning:{}'.format(np.sum(Y_pred == Y_test) / len(Y_test)))

    tree_clf = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=2)
    tree_clf.fit(X_train, Y_train)
    Y_pred = tree_clf.predict(X_test)
    print('acc_after_tuning:{}'.format(np.sum(Y_pred == Y_test) / len(Y_test)))
