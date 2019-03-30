import numpy as np


def roc_auc_score(Y_true, Y_scores):
    n_pos = np.sum(Y_true)
    n_neg = len(Y_true) - n_pos
    denominator = n_pos * n_neg

    rank_bins = 100
    bin_width = 1 / rank_bins

    # 某一rank值下对应的正负样本数
    pos_ranking = [0 for _ in range(rank_bins)]
    neg_ranking = [0 for _ in range(rank_bins)]

    for idx, label in enumerate(Y_true):
        rank = int(Y_scores[idx] / bin_width)  # 计算概率值对应的rank
        if label == 1:
            pos_ranking[rank] += 1
        else:
            neg_ranking[rank] += 1

    acc_neg = 0  # 由低往高开始计数，那么可以与正样本组合的负样本数是不断累加的
    legal_pair = 0  # 满足P_pos>P_neg的正负样本对
    for rank in range(rank_bins):
        legal_pair += (pos_ranking[rank] * acc_neg + pos_ranking[rank] * neg_ranking[rank] * 0.5)
        acc_neg += neg_ranking[rank]

    return legal_pair / denominator


if __name__ == '__main__':
    Y_true = np.array([0, 0, 1, 1])
    Y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    print('AUC:{}'.format(roc_auc_score(Y_true, Y_scores)))

    from sklearn.metrics import roc_auc_score

    print('sklearn AUC:{}'.format(roc_auc_score(Y_true, Y_scores)))
