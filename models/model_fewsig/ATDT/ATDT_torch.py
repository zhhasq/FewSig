import copy
import time

import numpy as np
import torch

from models.model_fewsig.ATDT.ATDT_LOO_torch import ATDTLOO
from models.model_fewsig.ATDT.Neighbors_fast import NeighborsFast
from models.model_fewsig.ATDT.Neighbors_fast2 import NeighborsFast2
from utils import scores
from utils.log_utils import log

'''
Difference to ExpDT is the threshold selection rule
'''
class ATDT:
    def __init__(self, target_fpr, gpu_count, log_path=None, init_gpu_id=0):
        self.kp = None
        self.kn = None
        self.p_threshold = None
        self.n_threshold = None
        self.pending_kp = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        self.pending_kn = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        # if the model is re-fit k_retrain_threshold times, then the k_af, k_non_af need to re-tuned
        self.k_retrain_threshold = 3
        self.k_retrain_counter = -1
        self.fit_counter = 0
        self.log_path = log_path

        self.target_FPR = target_fpr
        self.gpu_count = gpu_count
        self.init_gpu_id = init_gpu_id

    def __str__(self):
        return f"Two level decision decision tree: kp:{self.kp}, kn:{self.kn}, target_FPR={self.target_FPR}"


    def gen_thresholds(self, nn_pair_dist_list):
        thresholds_sort = np.sort(np.array(nn_pair_dist_list))
        thresholds = []
        thresholds.append(0)
        for i in range(len(thresholds_sort) - 1):
            mid = thresholds_sort[i] + (thresholds_sort[i + 1] - thresholds_sort[i]) / 2
            thresholds.append(mid)
        thresholds.append(thresholds_sort[-1] + 0.01)
        return np.array(thresholds)

    def gen_thresholds_v2(self, nn_pair_dist_list):
        thresholds_sort = np.sort(np.array(nn_pair_dist_list))
        thresholds = []
        thresholds.append(thresholds_sort[0] - 0.01)
        for i in range(len(thresholds_sort) - 1):
            mid = thresholds_sort[i] + (thresholds_sort[i + 1] - thresholds_sort[i]) / 2
            thresholds.append(mid)
        thresholds.append(thresholds_sort[-1] + 0.01)
        return np.array(thresholds)

    def fit(self, train_features, train_labels, pool):
        self.k_retrain_counter += 1
        exec_loo = True
        # if self.k_retrain_counter >= self.k_retrain_threshold or self.kp is None:
        #     exec_loo = True
        #     self.k_retrain_counter = -1
        # else:
        #     exec_loo = False
        log_s = []
        log_s.append(f"DT-ST-Torch fit round ^^^{self.fit_counter}^^^, current round FPR:{self.target_FPR}")

        self.train_features = train_features
        self.train_labels = np.array(train_labels)

        self.p_index = [i for i in range(len(train_labels)) if train_labels[i] == 1]
        self.n_index = [i for i in range(len(train_labels)) if train_labels[i] == 0]

        if len(self.p_index) <= 7:
            self.pending_kp = np.array([i for i in range(1, len(self.p_index) - 1, 2)])
        else:
            self.pending_kp = np.array([1, 3, 5, 7])
        if len(self.n_index) <= 7:
            self.pending_kn = np.array([i for i in range(1, len(self.n_index) - 1, 2)])
        else:
            self.pending_kn = np.array([1, 3, 5, 7])


        log_s.append(f"\tPending kp: {self.pending_kp}")
        log_s.append(f"\tPending kn: {self.pending_kn}")
        log_s.append(f"\tFit results:")

        p_index = [i for i in range(len(train_labels)) if train_labels[i] == 1]
        n_index = [i for i in range(len(train_labels)) if train_labels[i] == 0]
        # start = time.time()
        neighbor2 = NeighborsFast2(self.train_features, p_index, n_index, self.pending_kp, self.pending_kn)
        # end = time.time()
        # print(f"generate neighbor used {end-start}")
        loo_labels_all = []
        for cur_loo_index in range(len(train_labels)):
            cur_train_labels = [train_labels[i] for i in range(len(train_labels)) if i != cur_loo_index]
            loo_labels_all.append(cur_train_labels)
        loo_labels_all = np.array(loo_labels_all)

        loop_count = 0
        loo_tpr = []
        loo_fpr = []
        loo_f1 = []
        pending_k_pair = []
        results_async = []

        fit_p_threshold = np.zeros((len(self.pending_kp), len(self.pending_kn)))
        fit_n_threshold = np.zeros((len(self.pending_kp), len(self.pending_kn)))

        for i in range(len(self.pending_kp)):
            for j in range(len(self.pending_kn)):
                kp = self.pending_kp[i]
                kn = self.pending_kn[j]
                pending_k_pair.append((kp, kn))

                nn_pair_dist_p, nn_pair_dist_n = neighbor2.get_nn_ave_dist_all(kp, kn)
                p_thresholds = self.gen_thresholds_v2(nn_pair_dist_p)
                n_thresholds = self.gen_thresholds_v2(nn_pair_dist_n)
                device = torch.device(f"cuda:{self.init_gpu_id}")
                cur_p_threshold, cur_n_threshold, cur_tpr_max, cur_fpr_cloest = ATDT.tune_threshold_torch(
                    device, train_labels, p_thresholds, n_thresholds, nn_pair_dist_p, nn_pair_dist_n,
                    self.target_FPR)

                fit_p_threshold[i, j] = cur_p_threshold
                fit_n_threshold[i, j] = cur_n_threshold

                if exec_loo:
                    gpu_id = self.init_gpu_id + (loop_count % self.gpu_count)
                    results_async.append(pool.apply_async(ATDT.loo_mp_helper, args=(
                                        kp, kn, self.target_FPR, neighbor2, loo_labels_all, gpu_id)))

        if exec_loo:
            log_s.append(f"\tRun LOO on: {pending_k_pair}")
            for i in range(len(results_async)):
                y_hat_loo = results_async[i].get()
                s = scores.score_pr(y_hat_loo, train_labels)
                loo_tpr.append(s[4])
                loo_fpr.append(s[6])
                loo_f1.append(s[-1])

            bsf_i = np.argmax(np.array(loo_f1))
            self.kp = pending_k_pair[bsf_i][0]
            self.kn = pending_k_pair[bsf_i][1]

        i = np.where(self.pending_kp == self.kp)[0][0]
        j = np.where(self.pending_kn == self.kn)[0][0]
        self.p_threshold = fit_p_threshold[i][j]
        self.n_threshold = fit_n_threshold[i][j]
        log_s.append(f"\t\tLOO TPR: {loo_tpr}")
        log_s.append(f"\t\tLOO FPR: {loo_fpr}")
        log_s.append(f"\t\tLOO F1: {loo_f1}")
        log_s.append(f"\t current round kp = {self.kp}, kn = {self.kn}")

        loop_count += 1

        self.fit_counter += 1
        log_s.append(f"************************************************************")
        log_s.append("")
        log(log_s, self.log_path, False)

    @staticmethod
    def loo_mp_helper(kp, kn, target_FPR, neighbor2, loo_labels_all, gpu_id):
        exp_dt_loo = ATDTLOO(kp, kn, target_FPR)
        y_hat_loo = exp_dt_loo.fit(neighbor2, loo_labels_all, gpu_id)
        return y_hat_loo

    def predict(self, test_features):
        test_features = np.array(test_features)
        if test_features.shape[1] != self.train_features.shape[1]:
            raise RuntimeError
        results = []
        # neighbors = [Neighbors(i, test_features[i, :], is_include_self=False) for i in range(test_features.shape[0])]
        if test_features.shape[1] != len(self.p_index) + len(self.n_index):
            raise RuntimeError
        neighbors = NeighborsFast(test_features.reshape((-1, test_features.shape[1])), self.p_index, self.n_index, is_include_self=False)
        test_p, test_n = neighbors.get_loo_test_dist(self.kp, self.kn)
        for i in range(len(test_p)):
            if test_p[i] <= self.p_threshold and test_n[i] >= self.n_threshold:
                results.append(1)
            else:
                results.append(0)
        return results

    @staticmethod
    def tune_threshold_torch(device, train_labels, af_thresholds, non_af_thresholds, nn_pair_dist_af, nn_pair_dist_non_af,
                       target_FPR):

        tpr_m, fpr_m = ATDT._gen_matrix_torch(device, train_labels, af_thresholds,
                                                       non_af_thresholds, nn_pair_dist_af,
                                                       nn_pair_dist_non_af, target_FPR)
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
        r_index = bsf_m[0].cpu().detach().numpy()
        c_index = bsf_m[1].cpu().detach().numpy()
        select_af_thresholds = af_thresholds[r_index]
        select_non_af_thresholds = non_af_thresholds[c_index]

        af_threshold = select_af_thresholds.mean()
        non_af_threshold = select_non_af_thresholds.mean()

        # return af_threshold, non_af_threshold, r_index, c_index, tpr_m.cpu().detach().numpy(), fpr_m.cpu().detach().numpy(), tpr_max.cpu().detach().numpy(), fpr_cloest.cpu().detach().numpy()
        return af_threshold, non_af_threshold, tpr_max.cpu().detach().item(), fpr_cloest.cpu().detach().item()

    @staticmethod
    def _gen_matrix_torch(device, train_labels, af_thresholds, non_af_thresholds, nn_pair_dist_af, nn_pair_dist_non_af,
                          target_FPR,
                          delta=0.01):
        n = len(af_thresholds)
        m = len(non_af_thresholds)

        train_labels_sum = torch.tensor(train_labels.sum(), dtype=torch.float32, device=device)
        train_inverse_label_sum = torch.tensor(len(train_labels) - train_labels.sum(), dtype=torch.float32,
                                               device=device)

        train_labels = torch.tensor(train_labels, dtype=torch.float32, device=device).reshape((1, -1))
        train_labels_inverse = 1. - train_labels
        nn_pair_dist_non_af = torch.tensor(nn_pair_dist_non_af, dtype=torch.float32, device=device).reshape((1, -1))
        non_af_thresholds = torch.tensor(non_af_thresholds, dtype=torch.float32, device=device).reshape((-1, 1))
        non_af_result = nn_pair_dist_non_af - non_af_thresholds
        non_af_result = torch.where(non_af_result >= 0, 1., 0.)
        non_af_result = torch.transpose(non_af_result, 0, 1)

        nn_pair_dist_af = torch.tensor(nn_pair_dist_af, dtype=torch.float32, device=device).reshape((1, -1))
        af_thresholds = torch.tensor(af_thresholds, dtype=torch.float32, device=device).reshape((-1, 1))
        af_result = nn_pair_dist_af - af_thresholds
        af_result = torch.where(af_result <= 0, 1., 0.)


        train_labels = torch.mul(train_labels, af_result)
        tpr = torch.mm(train_labels, non_af_result)
        tpr.div_(train_labels_sum)

        train_labels_inverse = torch.mul(train_labels_inverse, af_result)
        fpr = torch.mm(train_labels_inverse, non_af_result)
        fpr.div_(train_inverse_label_sum)

        # return tpr.cpu().detach().numpy(), fpr.cpu().detach().numpy()
        return tpr, fpr
        # return train_labels, train_labels_inverse

    # def _gen_matrix(self, af_thresholds, non_af_thresholds, delta=0.05):
    #     n = len(af_thresholds)
    #     m = len(non_af_thresholds)
    #     tpr_m = np.zeros((n, m))
    #     fpr_m = np.ones((n, m))
    #
    #     for i in range(n):
    #         for j in range(m, -1, -1):
    #             tpr,fpr = self._gen_matrix_single_cell(af_thresholds[i], non_af_thresholds[j])
    #             if fpr > self.target_FPR + delta:
    #                 break
    #             tpr_m[i, j] = tpr
    #             fpr_m[i, j] = fpr
    #     return tpr_m, fpr_m
    #
    # def _gen_matrix_single_cell(self, cur_af_threshold, cur_non_af_threshold):
    #     y_hat = np.where((self.nn_pair_dist_af <= cur_af_threshold) & (self.nn_pair_dist_non_af >= cur_non_af_threshold), 1, 0)
    #     cur_score = scores.score_tpr_fpr2(y_hat, self.train_labels)
    #     return (cur_score[4], cur_score[5])
    #



