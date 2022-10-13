from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt


def onine_scores_UEA(y_hat, test_labels):
    time = [i+1 for i in range(len(y_hat))]
    tprs = []
    precisions = []
    fprs = []
    f1s = []
    for i in range(len(y_hat)):
        r= score_pr(y_hat[:i+1], test_labels[:i+1])
        tprs.append(r[4])
        precisions.append(r[5])
        fprs.append(r[-2])
        f1s.append(r[-1])
    # plt.tick_params(labelsize=15)

    ax1 = plt.subplot()
    ax1.tick_params(labelsize=15)
    l1, = ax1.plot(time, tprs, color='red')
    ax2 = ax1.twinx()
    ax2.tick_params(labelsize=15)
    l2, = ax2.plot(time, precisions, color='green')

    plt.show()
    plt.legend([l1, l2], ["TPR/Recall", "Precision"])

    ax1.set_xlabel('days since first aftershock', fontsize=15)
    ax1.set_ylabel('TPR', fontsize=15)
    ax2.set_ylabel("Precision", fontsize=15)

    return (tprs[-1], fprs[-1], precisions[-1]), time, f1s, tprs, fprs

def score_tpr_fpr2(predict_labels, true_labels, print_results=False):
    tp = np.where((predict_labels == true_labels) & (true_labels == 1), 1, 0)
    tp = tp.sum()
    tn = np.where((predict_labels == true_labels) & (true_labels == 0), 1, 0)
    tn = tn.sum()
    fn = np.where((predict_labels != true_labels) & (true_labels == 1), 1, 0)
    fn = fn.sum()
    fp = np.where((predict_labels != true_labels) & (true_labels == 0), 1, 0)
    fp = fp.sum()

    if print_results:
        print(tp, tn, fn, fp)
    tpr = tp / (tp + fn)
    if fp == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)
    return (tp, tn, fp, fn, tpr, fpr)

def score_tpr_fpr3(predict_labels, true_labels, true_labels_sum, print_results=False):
    # tmp = (predict_labels == true_labels)

    # tmp = np.logical_xor(predict_labels, true_labels_reverse)
    # tp = np.where((tmp & true_labels_eq1_condition), 1, 0)
    # tp = tp.sum()
    n = len(true_labels)
    tp = (predict_labels & true_labels).sum()
    tn = n - (predict_labels | true_labels).sum()


    fn = true_labels_sum - tp
    fp = n - true_labels_sum - tn
    # if fn2 != fn or fp2 != fp:
    #     raise RuntimeError
    # if print_results:
    #     print(tp, tn, fn, fp)
    if tp == 0:
        tpr = 0
    else:
        tpr = tp / (tp + fn)
    if fp == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)
    return (tp, tn, fp, fn, tpr, fpr)

def score_pr(predict_labels, true_labels, print_results=False):
    tp = np.where((predict_labels == true_labels) & (true_labels == 1), 1, 0)
    tp = tp.sum()
    tn = np.where((predict_labels == true_labels) & (true_labels == 0), 1, 0)
    tn = tn.sum()
    fn = np.where((predict_labels != true_labels) & (true_labels == 1), 1, 0)
    fn = fn.sum()
    fp = np.where((predict_labels != true_labels) & (true_labels == 0), 1, 0)
    fp = fp.sum()
    if print_results:
        print(tp, tn, fn, fp)
    # tpr = tp / (tp + fn)
    recall = tp / (tp + fn)
    if tp == 0:
        precision = 0
    else:
        precision = tp/(tp+fp)
    if fp == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2*precision*recall)/ (precision+recall)
    return (tp, tn, fp, fn, recall, precision, fpr, f1)

def score_tpr_fpr(predict_labels, true_labels):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(predict_labels)):
        true_label = true_labels[i]
        predict_label = predict_labels[i]

        if true_label == 1:
            if predict_label == 0:
                fn += 1
            elif predict_label == 1:
                tp += 1
        elif true_label == 0:
            if predict_label == 0:
                tn += 1
            elif predict_label == 1:
                fp += 1

    tpr = tp / (tp + fn)
    if fp == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)
    return (tp, tn, fp, fn, tpr, fpr)

def score_auc(predicts, true_labels, reverse=False):
    predicts_sort = sorted(predicts)
    thresholds = []

    thresholds.append(predicts_sort[0] - 1)
    for i in range(1, len(predicts_sort) - 1):
        thresholds.append(predicts_sort[i-1] + (predicts_sort[i] - predicts_sort[i-1])/2)
    thresholds.append(predicts_sort[-1] + 1)
    tprs = []
    fprs = []
    for cur_threshold in thresholds:
        cur_y_hat = get_labels(predicts, cur_threshold, reverse)
        cur_score = score_tpr_fpr(cur_y_hat, true_labels)
        tprs.append(cur_score[4])
        fprs.append(cur_score[5])
    final_auc = metrics.auc(fprs, tprs)
    roc_display = metrics.RocCurveDisplay(fpr=fprs, tpr=tprs, roc_auc=final_auc)
    return final_auc, roc_display

def get_labels(af_probs, threshold, reverse):
    # if < thresholds then return 0, >= threshold return 1
    y_hat = []
    y_hat_reverse = []
    for cur_af_prob in af_probs:
        if cur_af_prob >= threshold:
            y_hat.append(1)
            y_hat_reverse.append(0)
        else:
            y_hat.append(0)
            y_hat_reverse.append(1)
    if reverse:
        return y_hat_reverse
    else:
        return y_hat

