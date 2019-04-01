import pandas as pd
from collections import namedtuple
import os

data_structure = namedtuple('data_structure', ['data', 'target'])


# 路径问题，注意在外部引用并执行load函数时，相对路径的当前路径是以__main__为准的
# 这里的路径写法只能用于算法模块的测试逻辑，实际是存在BUG的
# 一种解决方法是使用绝对路径，但是不具有通用性，待解决


def load_boston():
    path = os.path.join(os.path.dirname(__file__), 'housing', 'housing.data')
    df = pd.read_csv(path, header=None)
    data = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    return data_structure(data, target)


def load_iris():
    path = os.path.join(os.path.dirname(__file__), 'iris', 'bezdekIris.data')
    df = pd.read_csv(path, header=None)
    data = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    return data_structure(data, target)


def load_diabetes():
    path = os.path.join(os.path.dirname(__file__), 'diabetes', 'diabetes.tab.txt')
    df = pd.read_csv(path, sep='\t')
    data = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    return data_structure(data, target)


def load_digits():
    path = os.path.join(os.path.dirname(__file__), 'optdigits', 'optdigits.tes')
    df = pd.read_csv(path, header=None)
    data = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    return data_structure(data, target)


def load_wine():
    path = os.path.join(os.path.dirname(__file__), 'wine', 'wine.data')
    df = pd.read_csv(path, header=None)
    data = df.iloc[:, 1:].values
    target = df.iloc[:, 0].values - 1
    return data_structure(data, target)


def load_breast_cancer():
    path = os.path.join(os.path.dirname(__file__), 'breast-cancer-wisconsin', 'wdbc.data')
    df = pd.read_csv(path, header=None)
    data = df.iloc[:, 2:].values
    target = df.iloc[:, 1].values
    return data_structure(data, target)


if __name__ == '__main__':
    data = load_breast_cancer()
    X = data.data
    Y = data.target
