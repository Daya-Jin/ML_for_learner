import pandas as pd
from collections import namedtuple
import os

data_structure = namedtuple('data_structure', ['data', 'target'])


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
