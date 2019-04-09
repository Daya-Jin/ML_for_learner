import numpy as np
import re
from collections import Counter


class CountVectorizer:
    def __init__(self):
        self.tokens = None
        self.vocabulary_ = None

    def _clean(self, s):
        s = s.lower()  # 小写化
        s = re.sub(r"[^a-zA-Z0-9]", " ", s)  # 标点转空格
        return s

    def fit(self, raw_documents):
        self.tokens = set()
        for s in raw_documents:
            self.tokens.update(self._clean(s).split())
        self.tokens = np.array(sorted(list(self.tokens)))
        self.vocabulary_ = dict(zip(self.tokens, [idx for idx in range(len(self.tokens))]))

    def _string_vectorizer(self, s):
        '''
        单个字串向量化
        :param s:
        :return:
        '''
        s = self._clean(s)
        vec = [0 for _ in range(len(self.tokens))]
        cnts = dict(Counter(s.split()))
        for idx in range(len(self.tokens)):
            vec[idx] = cnts.get(self.tokens[idx], 0)
        return vec

    def transform(self, raw_documents):
        return np.array(list(map(self._string_vectorizer, raw_documents)))

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def get_feature_names(self):
        return list(self.tokens)


if __name__ == '__main__':
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.vocabulary_)
    print(vectorizer.get_feature_names())
    print(X)
