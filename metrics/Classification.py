import numpy as np


def accuracy_score(Y_true, Y_pred, sample_weight=None):
    assert len(Y_true) == len(Y_pred)
    n_samples = len(Y_true)
    sample_weight = np.array([1 / n_samples for _ in range(n_samples)]) if not sample_weight else sample_weight
    return np.sum((np.array(Y_true) == np.array(Y_pred)) * sample_weight)


def f1_score(Y_true, Y_pred, average: str = None):
    '''
    :param Y_true:
    :param Y_pred:
    :param average: 均化方式，可选参数'micro'，'macro'，'weighted'
    该参数置空时返回所有类别的F1分数
    :return:
    '''
    uni_labels, label_weight = np.unique(Y_true, return_counts=True)
    label_weight = label_weight / len(Y_true)  # 类分布概率，用作标签权重
    total_TP = total_FP = total_FN = 0  # 用于计算micro f1的总计数

    f1_scores = list()
    for label in uni_labels:
        TP = np.sum((np.array(Y_pred) == label) * (np.array(Y_true) == label))
        FP = np.sum((np.array(Y_pred) == label) * (np.array(Y_true) != label))
        FN = np.sum((np.array(Y_pred) != label) * (np.array(Y_true) == label))

        total_TP += TP
        total_FP += FP
        total_FN += FN

        cur_precision = TP / (TP + FP)
        cur_recall = TP / (TP + FN)

        cur_f1 = 0 if cur_precision == 0 or cur_recall == 0 else 2 * cur_precision * cur_recall / (
                cur_precision + cur_recall)
        f1_scores.append(cur_f1)
    f1_scores = np.array(f1_scores)

    if average == 'micro':
        precision = total_TP / (total_TP + total_FP)
        recall = total_TP / (total_TP + total_FN)
        return 0 if precision == 0 or recall == 0 else 2 * precision * recall / (precision + recall)
    elif average == 'macro':
        return np.sum(f1_scores) / len(uni_labels)
    elif average == 'weighted':
        return np.sum(f1_scores * label_weight)
    else:
        return f1_scores


def log_loss(Y_true, Y_pred, eps: float = 1e-15):
    '''
    交叉熵计算函数
    :param Y_true:
    :param Y_pred:
    :param eps: 极小值，用于截断{0,1}值
    :return:
    '''
    Y_true = np.array(Y_true).astype(float)
    Y_pred = np.array(Y_pred).astype(float)

    # 交叉熵在0，1处无定义，需要做截断
    L = eps
    H = 1 - eps
    Y_true[Y_true > H] = H
    Y_true[Y_true < L] = L
    Y_pred[Y_pred > H] = H
    Y_pred[Y_pred < L] = L

    return -np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))


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
