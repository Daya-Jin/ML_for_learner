import pandas as pd
from collections import namedtuple

data_structure = namedtuple('data_structure', ['data', 'target'])


# 路径问题，注意在外部引用并执行load函数时，相对路径的当前路径是以__main__为准的
# 这里的路径写法只能用于算法模块的测试逻辑，实际是存在BUG的
# 一种解决方法是使用绝对路径，但是不具有通用性，待解决


def load_boston():
    df = pd.read_csv('../datasets/housing/housing.data', header=None)
    data = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    return data_structure(data, target)


def load_iris():
    df = pd.read_csv('../datasets/iris/bezdekIris.data', header=None)
    data = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    return data_structure(data, target)


def load_diabetes():
    df = pd.read_csv('../datasets/diabetes/diabetes.tab.txt', sep='\t')
    data = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    return data_structure(data, target)


def load_digits():
    df = pd.read_csv('../datasets/optdigits/optdigits.tes', header=None)
    data = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    return data_structure(data, target)


def load_wine():
    df = pd.read_csv('../datasets/wine/wine.data', header=None)
    data = df.iloc[:, 1:].values
    target = df.iloc[:, 0].values - 1
    return data_structure(data, target)


def load_breast_cancer():
    df = pd.read_csv('../datasets/breast-cancer-wisconsin/wdbc.data', header=None)
    data = df.iloc[:, 2:].values
    target = df.iloc[:, 1].values
    return data_structure(data, target)


if __name__ == '__main__':
    data = load_breast_cancer()
    X = data.data
    Y = data.target
