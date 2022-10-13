import numpy as np
import torch


'''
Difference to ExpDT is the threshold selection rule
'''


class ATDTLOO:
    def __init__(self, kp, kn, target_FPR):
        self.kp = kp
        self.kn= kn
        self.target_FPR = target_FPR
        self.p_threshold = None
        self.n_threshold = None

    def __str__(self):
        return f"Two level decision decision tree: k_af:{self.kp}, k_non_af:{self.kn}, target_FPR={self.target_FPR}"

    @staticmethod
    def gen_thresholds(nn_pair_dist_list):
        thresholds_sort = np.sort(nn_pair_dist_list)
        thresholds = []
        thresholds.append(0)
        for i in range(len(thresholds_sort) - 1):
            mid = thresholds_sort[i] + (thresholds_sort[i + 1] - thresholds_sort[i]) / 2
            thresholds.append(mid)
        thresholds.append(thresholds_sort[-1] + 0.01)
        return np.array(thresholds)

    @staticmethod
    def gen_thresholds_v2(nn_pair_dist_list):
        thresholds_sort = np.sort(np.array(nn_pair_dist_list))
        thresholds = []
        thresholds.append(thresholds_sort[0] - 0.01)
        for i in range(len(thresholds_sort) - 1):
            mid = thresholds_sort[i] + (thresholds_sort[i + 1] - thresholds_sort[i]) / 2
            thresholds.append(mid)
        thresholds.append(thresholds_sort[-1] + 0.01)
        return np.array(thresholds)

    def fit(self, neighbor2, loo_labels_all, gpu_num):

        p_nn_dist_all, test_p_nnd = neighbor2.get_loo_p_nnd(self.kp)
        n_nn_dist_all, test_n_nnd = neighbor2.get_loo_n_nnd(self.kn)

        p_thresholds_all = np.array([ATDTLOO.gen_thresholds_v2(x) for x in p_nn_dist_all])
        n_thresholds_all = np.array([ATDTLOO.gen_thresholds_v2(x) for x in n_nn_dist_all])

        device = torch.device(f"cuda:{gpu_num}")
        y_hat = self.loo_predict_torch(p_nn_dist_all, n_nn_dist_all, p_thresholds_all, n_thresholds_all, loo_labels_all,
                                       test_p_nnd, test_n_nnd, self.target_FPR, device=device)

        return y_hat


    @staticmethod
    def loo_predict_torch(p_nn_dist_all, n_nn_dist_all, p_thresholds_all, n_thresholds_all, labels_all, test_p_nn_dist_all,
                          test_n_nn_dist_all, target_FPR, device=None):

        p_nn_dist_all_torch = torch.tensor(p_nn_dist_all, dtype=torch.float32, device=device)
        n_nn_dist_all_torch = torch.tensor(n_nn_dist_all, dtype=torch.float32, device=device)
        p_thresholds_all_torch = torch.tensor(p_thresholds_all, dtype=torch.float32, device=device)
        n_thresholds_all_torch = torch.tensor(n_thresholds_all, dtype=torch.float32, device=device)
        labels_all_torch = torch.tensor(labels_all, dtype=torch.float32, device=device)
        labels_all_sum_torch = torch.sum(labels_all_torch, dim=1)
        labels_reverse_all_sum_torch = len(labels_all[0]) - labels_all_sum_torch
        test_p_nn_dist_all_torch = torch.tensor(test_p_nn_dist_all, dtype=torch.float32, device=device)
        test_n_nn_dist_all_torch = torch.tensor(test_n_nn_dist_all, dtype=torch.float32, device=device)

        y_hat = torch.zeros(len(test_p_nn_dist_all))
        for i in range(len(test_p_nn_dist_all)):
            p_threshold, n_threshold = ATDTLOO.tune_threshold_torch(p_thresholds_all_torch[i, :], n_thresholds_all_torch[i, :],
                                                p_nn_dist_all_torch[i, :], n_nn_dist_all_torch[i, :],
                                                labels_all_torch[i, :], labels_all_sum_torch[i],
                                                                    labels_reverse_all_sum_torch[i], target_FPR, device)
            if test_p_nn_dist_all_torch[i] <= p_threshold and test_n_nn_dist_all_torch[i] >= n_threshold:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        return y_hat.cpu().detach().numpy()



    @staticmethod
    def tune_threshold_torch(p_thresholds, n_thresholds, p_nn_dist, n_nn_dist, train_labels, train_labels_sum,
                             train_inverse_label_sum, target_FPR, device):

        tpr_m, fpr_m = ATDTLOO._gen_matrix_torch(p_thresholds, n_thresholds, p_nn_dist, n_nn_dist, train_labels,
                                                 train_labels_sum, train_inverse_label_sum)
        # tmp = np.abs(fpr_m - target_FPR)
        tmp = torch.abs(torch.sub(fpr_m, target_FPR))
        # bsf_i = np.argmin(tmp, axis=
        min_tmp = torch.min(tmp)
        bsf_i = torch.argmin(tmp)
        r = int(bsf_i / tmp.size(dim=1))
        c = bsf_i % tmp.size(dim=1)

        fpr_cloest = fpr_m[r, c]
        # tprs = tpr_m[fpr_m == fpr_cloest]
        min_dummy = torch.tensor(-1.0, dtype=torch.float32, device=device)
        tprs = torch.where(fpr_m == fpr_cloest, tpr_m, min_dummy)
        tpr_max = torch.max(tprs)

        # bsf_m = np.where((tpr_m == tpr_max) & (fpr_m == fpr_cloest))
        condition = torch.where((tpr_m == tpr_max) & (fpr_m == fpr_cloest), 1, 0)
        bsf_m = torch.nonzero(condition, as_tuple=True)

        # r_index = bsf_m[0].cpu().detach().numpy()
        # c_index = bsf_m[1].cpu().detach().numpy()
        r_index = bsf_m[0]
        c_index = bsf_m[1]

        select_p_thresholds = torch.index_select(p_thresholds, 0, r_index)
        select_n_thresholds = torch.index_select(n_thresholds, 0, c_index)

        p_threshold = select_p_thresholds.mean()
        n_threshold = select_n_thresholds.mean()

        # return af_threshold, non_af_threshold, r_index, c_index, tpr_m.cpu().detach().numpy(), fpr_m.cpu().detach().numpy(), tpr_max.cpu().detach().numpy(), fpr_cloest.cpu().detach().numpy()
        return p_threshold, n_threshold


    @staticmethod
    def _gen_matrix_torch(p_thresholds, n_thresholds, p_nn_dist, n_nn_dist, train_labels, train_labels_sum, train_inverse_label_sum):

        train_labels_inverse = 1. - train_labels
        n_result = n_nn_dist.view(1, -1) - n_thresholds.view(-1, 1)
        n_result = torch.where(n_result >= 0, 1., 0.)
        n_result = torch.transpose(n_result, 0, 1)

        p_result = p_nn_dist.view(1, -1) - p_thresholds.view(-1, 1)
        p_result = torch.where(p_result <= 0, 1., 0.)

        train_labels = torch.mul(train_labels, p_result)
        tpr = torch.mm(train_labels, n_result)
        tpr.div_(train_labels_sum)

        train_labels_inverse = torch.mul(train_labels_inverse, p_result)
        fpr = torch.mm(train_labels_inverse, n_result)
        fpr.div_(train_inverse_label_sum)


        # return tpr.cpu().detach().numpy(), fpr.cpu().detach().numpy()
        return tpr, fpr
        # return train_labels, train_labels_inverse

