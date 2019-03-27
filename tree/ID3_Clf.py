import pandas as pd
import numpy as np

class ID3:
    def __init__(self):
        self.tree = None
        self.dataset = None

    def __entropy(self, feature):
        uni_val, cnt = np.unique(feature, return_counts=True)  # 返回独特值与计数
        # 熵的计算
        H = np.sum([(-cnt[i] / np.sum(cnt)) * np.log2(cnt[i] / np.sum(cnt))
                    for i in range(len(uni_val))])
        return H

    def __InfoGain(self, dataset, f_test_col, Y_col=-1):
        entropy_before = self.__entropy(dataset.iloc[:, Y_col])  # 分割前的熵

        uni_val, cnt = np.unique(dataset.iloc[:, f_test_col], return_counts=True)  # 计算分割特征的独特值与计数
        entropy_cond = np.sum([(cnt[i] / np.sum(cnt)) * self.__entropy(dataset.where(dataset.iloc[:, f_test_col]
                                                                                     == uni_val[i]).dropna().iloc[:,
                                                                       Y_col])
                               for i in range(len(uni_val))])
        return entropy_before - entropy_cond

    def __gen_tree(self, dataset, org_dataset, f_cols, Y_col=-1, p_node_cls=None):
        '''
        dataset: 用于分割的数据
        org_dataset: 最原始的数据，全部数据
        f_cols: 备选特征
        '''
        # 如果数据中的Y已经纯净了，则返回Y的取值
        if len(np.unique(dataset.iloc[:, Y_col])) <= 1:
            return np.unique(dataset.iloc[:, Y_col])[0]

        # 如果传入数据为空(对应空叶节点)，则返回原始数据中数量较多的label值
        elif len(dataset) == 0:
            uni_cls, cnt = np.unique(
                org_dataset.iloc[:, Y_col], return_counts=True)
            return uni_cls[np.argmax(cnt)]

        # 如果没有特征可用于划分，则返回父节点中数量较多的label值
        # 由于初始传入的是Index类型，所以这里不能用if not
        elif len(f_cols) == 0:
            return p_node_cls

        # 否则进行分裂
        else:
            # 得到当前节点中数量最多的label，递归时会赋给下层函数的p_node_cls
            cur_uni_cls, cnt = np.unique(
                dataset.iloc[:, Y_col], return_counts=True)
            cur_node_cls = cur_uni_cls[np.argmax(cnt)]
            del cur_uni_cls, cnt

            # 根据信息增益选出最佳分裂特征
            gains = [self.__InfoGain(dataset, f_col, Y_col) for f_col in f_cols]
            best_f = f_cols[np.argmax(gains)]

            # 更新备选特征
            f_cols = [col for col in f_cols if col != best_f]

            # 按最佳特征的不同取值，划分数据集并递归
            tree = {best_f: {}}
            for val in np.unique(dataset.iloc[:, best_f]):  # ID3对每一个取值都划分数据集
                sub_data = dataset.where(dataset.iloc[:, best_f] == val).dropna()
                sub_tree = self.__gen_tree(sub_data, dataset, f_cols, Y_col, cur_node_cls)
                tree[best_f][val] = sub_tree  # 分裂特征的某一取值，对应一颗子树或叶节点

            return tree

    def fit(self, X_train, Y_train):
        dataset = np.c_[X_train, Y_train]
        self.dataset = pd.DataFrame(dataset, columns=list(range(dataset.shape[1])))
        self.tree = self.__gen_tree(self.dataset, self.dataset, list(range(self.dataset.shape[1] - 1)))

    def __predict_one(self, x_test, tree, default=-1):
        '''
        query：一个测试样本，字典形式，{f:val,f:val,...}
        tree：训练生成树
        default：查找失败时返回的默认类别
        '''
        for feature in list(x_test.keys()):
            if feature in list(tree.keys()):  # 如果该特征与根节点的划分特征相同
                try:
                    sub_tree = tree[feature][x_test[feature]]  # 根据特征的取值来获取左右分支

                    if isinstance(sub_tree, dict):  # 判断是否还有子树
                        return self.__predict_one(x_test, tree=sub_tree)  # 有则继续查找
                    else:
                        return sub_tree  # 是叶节点则返回结果
                except:  # 没有查到则说明是未见过的情况，只能返回default
                    return default

    def predict(self, X_test):
        X_test = pd.DataFrame(X_test, columns=list(range(X_test.shape[1]))).to_dict(orient='record')
        Y_pred = list()
        for item in X_test:
            Y_pred.append(self.__predict_one(item, tree=self.tree))
        return Y_pred


def load_zoo():
    '''
    返回一个sklearn-like的数据集
    '''
    from collections import namedtuple
    df = pd.read_csv('../utils/dataset/UCI_Zoo_Data_Set/zoo.data.csv', header=None)
    df = df.drop([0], axis=1)  # 首列是animal_name，丢弃

    dataClass = namedtuple('data', ['data', 'target'])
    dataClass.data = df.iloc[:, :-1].values
    dataClass.target = df.iloc[:, -1].values

    return dataClass


if __name__ == '__main__':
    from model_selection.train_test_split import train_test_split

    data = load_zoo()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    id3_tree = ID3()
    id3_tree.fit(X_train, Y_train)

    Y_pred = id3_tree.predict(X_test)
    print('acc:{}'.format(np.sum(np.array(Y_test) == np.array(Y_pred)) / len(Y_test)))
