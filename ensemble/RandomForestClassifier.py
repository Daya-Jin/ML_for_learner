if __name__=='__main__':
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, Y = data.data, data.target

    from tree.DecisionTreeClassifier import DecisionTreeClassifier
    for _ in range(3):
        tree=DecisionTreeClassifier()
        tree.fit(X,Y)
        print('import!')