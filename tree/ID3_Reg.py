import numpy as np
import pandas as pd

class RegTree:
    def __init__(self):
        self.tree=None
        self.dataset=None

    def __Var(self,data, f_test_col, Y_col=-1):
        f_uni_val = np.unique(data.iloc[:, f_test_col])

        # 对每一个可能的特征值做分裂测试并记录分裂后的加权方差
        f_var = 0
        for val in f_uni_val:
            # 把该特征等于某特定值的子集取出来
            cutset = data[data.iloc[:, f_test_col] == val].reset_index()
            # 加权方差
            cur_var = (len(cutset) / len(data)) * np.var(cutset.iloc[:, Y_col], ddof=1)
            f_var += cur_var

        return f_var

    def __gen_tree(self,data, org_dataset, f_cols, min_instances=5, Y_col=-1, p_node_mean=None):
        '''

        '''
        # 如果数据量小于最小分割量
        if len(data) <= int(min_instances):
            return np.mean(data.iloc[:, Y_col])

        # 数据为空，返回父节点数据中的目标均值
        elif len(data) == 0:
            return np.mean(org_dataset.iloc[:, Y_col])

        # 无特征可分，返回父节点均值
        elif len(f_cols) == 0:
            return p_node_mean

        else:
            # 当前节点的均值，会被传递给下层函数作为p_node_mean
            p_node_mean = np.mean(data.iloc[:, Y_col])

            # 找出最佳(方差最低)分裂特征
            f_vars = [self.__Var(data, f) for f in f_cols]
            best_f_idx = np.argmin(f_vars)
            best_f = f_cols[best_f_idx]

            tree = {best_f: {}}

            # 移除已分裂的特征
            features = [f for f in f_cols if f != best_f]

            # 以最佳特征的每一个取值划分数据并生成子树
            for val in np.unique(data.loc[:, best_f]):
                subset = data.where(data.loc[:, best_f] == val).dropna()
                tree[best_f][val] = self.__gen_tree(
                    subset, data, features, min_instances, Y_col, p_node_mean)

            return tree

    def fit(self, X_train, Y_train):
        dataset = np.c_[X_train, Y_train]
        self.dataset = pd.DataFrame(dataset, columns=list(range(dataset.shape[1])))
        self.tree = self.__gen_tree(self.dataset, self.dataset, list(range(self.dataset.shape[1] - 1)))

    def __predict_one(self, x_test, tree, default=0):
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
        return np.array(Y_pred)

def load_bike():
    '''
    返回一个sklearn-like的数据集
    '''
    from collections import namedtuple
    df = pd.read_csv('../utils/dataset/Bike_Sharing_Dataset/day.csv',usecols=['season','holiday','weekday','workingday','weathersit','cnt'])

    dataClass = namedtuple('data', ['data', 'target'])
    dataClass.data = df.iloc[:, :-1].values
    dataClass.target = df.iloc[:, -1].values

    return dataClass

if __name__ == '__main__':
    from model_selection.train_test_split import train_test_split

    data = load_bike()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    reg_tree=RegTree()
    reg_tree.fit(X_train, Y_train)

    Y_pred=reg_tree.predict(X_test)

    print('RMSE:{}'.format(np.sqrt(np.sum(np.square(Y_test-Y_pred))/len(Y_test))))