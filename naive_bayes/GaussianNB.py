class GaussianNB:
    def __init__(self):
        self.Y_prob = None
        self.mean = None
        self.std = None

    # 计算类分布概率
    def __cls_prob(self, Y_train):
        uni_val, cnt = np.unique(Y, return_counts=True)
        self.Y_prob = np.array([cnt[idx]/np.sum(cnt)
                                for idx in range(len(uni_val))])

    # 计算各特征在各类别下的统计值
    def __mean_std(self, X_train, Y_train):
        uni_cls = np.unique(Y_train)
        m_feature = X_train.shape[1]

        self.mean = np.array(
            [X_train[np.where(Y_train == cls)].mean(axis=0) for cls in uni_cls])
        self.std = np.array(
            [X_train[np.where(Y_train == cls)].std(axis=0) for cls in uni_cls])

    # 训练函数，实质就是计算训练数据中的一些统计量
    def fit(self, X_train, Y_train):
        self.__cls_prob(Y_train)
        self.__mean_std(X_train, Y_train)

    def __post_prob(self, x_test):
        return np.log2(1/(np.sqrt(2*np.pi)*self.std)*np.exp(-np.square(x_test-self.mean)/(2*np.square(self.std)))).sum(axis=1)

    def predict(self, X_test):
        return np.array([np.argmax(self.__post_prob(item)+np.log2(self.Y_prob)) for item in X_test])

if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    Y_pred = gnb.predict(X_test)
    print('acc:{}'.format(np.sum(Y_pred == Y_test)/len(Y_test)))