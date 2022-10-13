import time

import numpy as np
import torch

from models.model_fewsig.ATDT.ATDT_torch import ATDT
from multiprocessing import Pool

from models.model_fewsig.NCFAE.NCFA_ensemble import NCFAEnsemble
from models.model_fewsig.online_train.assistant_st3 import AssistantST3


class FewSig:
    @staticmethod
    def get_one_vote_y_hat(y_hat, vote_i):
        y_hat_cur_vote = []
        for i in range(len(y_hat)):
            if len(y_hat[i]) == 1:
                y_hat_cur_vote.append(y_hat[i][0])
            else:
                y_hat_cur_vote.append(y_hat[i][vote_i][0])
        y_hat_cur_vote = np.array(y_hat_cur_vote)
        return y_hat_cur_vote

    def __init__(self, DT_target_FPR, LR, gpu_count, init_gpu_id=0):
        self.gpu_count = gpu_count
        self.DT_target_FPR = DT_target_FPR
        self.init_gpu_id = init_gpu_id
        self.nca_init_args = {
            "device": torch.device("cuda:0"),
            # "device": torch.device("cpu"),
            "loss_func_name": "FOCAL",
            "loss_args": {"alpha": None, "gamma": 2, "softmax_neg_dist": True, "softmax_norm": True,
                          "softmax_de_max": False,
                          "dist_func": "L2"},
            "num_iter": 250,
            "verbose": False,
            "optimizer_name": "SGD",
            "optimizer_args": {"lr": LR}
        }

    def evaluate(self, train_set, test_set, loader, log_path=None, record_time=False):
        gpu_count = self.gpu_count
        model = NCFAEnsemble(1, [x for x in range(1, 20)], self.nca_init_args, gpu_count, gpu_init_id=self.init_gpu_id)
        # tmp_log = "/home/zhongs/synology_mnt/tmp_log6.log"
        assist_model = ATDT(self.DT_target_FPR, gpu_count, log_path, self.init_gpu_id)
        nca_ensemble_st = AssistantST3(model, assist_model, train_set, loader)
        # nca_ensemble_st = AssistantSTLoop(model, assist_model, train_3ch, dm)
        pool = Pool(processes=self.gpu_count)
        if record_time:
            init_time_start = time.time()
            model_for_init = NCFAEnsemble(1, [x for x in range(1, 20)], self.nca_init_args, gpu_count)
            assist_model_init = ATDT(self.DT_target_FPR, gpu_count, log_path, self.init_gpu_id)
            model_for_init.fit(train_set.get_features(), train_set.get_labels(), pool)
            assist_model_init.fit(train_set.get_features(), train_set.get_labels(), pool)
            init_time_end = time.time()
            self.init_time = init_time_end - init_time_start

        y_hat, nn_train_index, added_test_ids = nca_ensemble_st.evaluate(test_set, pool, record_time)
        if record_time:
            self.test_time = nca_ensemble_st.test_time
            self.retrain_time = nca_ensemble_st.retrain_time
        pool.close()
        print("waiting joined")
        pool.join()
        print("joined")
        if record_time:
            return None
        added_test_ids = np.array([x[0] for x in added_test_ids])
        added_test_ids_set = set(added_test_ids)
        pseudo_labels = []
        for cur_id in test_set.get_ids():
            if cur_id in added_test_ids_set:
                pseudo_labels.append(1)
            else:
                pseudo_labels.append(0)

        y_hat_split = [FewSig.get_one_vote_y_hat(y_hat, i) for i in range(19)]

        return np.array(y_hat_split), np.array(pseudo_labels)

