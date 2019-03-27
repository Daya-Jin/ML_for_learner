import numpy as np


def f1_score(Y_true, Y_pred, average=None):
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


if __name__ == '__main__':
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    print(f1_score(y_true, y_pred, average='weighted'))
