import copy

import numpy as np
import torch

from models.model_fewsig.NCFAE.NCFA_KNN import NCFAKNN


class NCFAEnsemble:
    def __init__(self, k, num_vote, nca_init_args, gpu_count, gpu_init_id=0):
        self.models = []
        self.k = k
        'num_vote can be a list'
        self.num_vote = num_vote
        self.nca_init_args = nca_init_args
        self.gpu_count = gpu_count
        self.gpu_init_id = gpu_init_id

    def __str__(self):
        return f"Distance features NCA ensemble: k:{self.k}, nc_list:{self.nc_list}, vote:{self.num_vote}"

    def fit(self, train_features, train_labels, pool):
        # train_features = copy.deepcopy(train_features)
        # train_labels = copy.deepcopy(train_labels)
        feature_len = train_features.shape[1]
        a = int(np.round(feature_len / 20))
        self.nc_list = [nc * a for nc in range(1, 20)]
        self.models = []
        results_async = []
        # if self.pool is None:
        #     # self.models.append(ExpNCATorchEnsembleAuto.fit_mp_helper(cur_nc, self.k, self.nca_init_args,
        #     #                                                          train_features, train_labels))
        #     self.pool = Pool(processes=4)


        for i in range(len(self.nc_list)):
            cur_nc = self.nc_list[i]
            gpu_num = self.gpu_init_id + (i % self.gpu_count)
            nca_init_args = copy.deepcopy(self.nca_init_args)
            nca_init_args["device"] = torch.device(f"cuda:{gpu_num}")
            results_async.append(pool.apply_async(NCFAEnsemble.fit_mp_helper, args=(
                cur_nc, self.k, nca_init_args, train_features, train_labels)))

        for i in range(len(results_async)):
            self.models.append(results_async[i].get())
        #
        # self.pool.close()
        # # print("waiting joined")
        # self.pool.join()
        # # print("joined")
        # self.pool = None

    @staticmethod
    def fit_mp_helper(cur_nc, k, nca_init_args, train_features, train_labels):
        cur_model = NCFAKNN(cur_nc, k, nca_init_args)
        cur_model.fit(train_features, train_labels)
        return cur_model

    def predict(self, test_features):
        results = []
        # nn_index_results = []
        for cur_model in self.models:
            cur_y_hat = cur_model.predict(test_features)
            # cur_y_hat, cur_nn_train_index = cur_model.predict2(test_features)
            results.append(cur_y_hat)
            # nn_index_results.append(cur_nn_train_index)
        results = np.array(results)
        # nn_index_results = np.array(nn_index_results)
        if isinstance(self.num_vote, list):
            all_votes_resutls = []
            for cur_num_vote in self.num_vote:
                y_hat = []
                for i in range(results.shape[1]):
                    cur_y = self.vote(results[:, i], cur_num_vote)
                    y_hat.append(cur_y)
                all_votes_resutls.append(np.array(y_hat))
            return np.array(all_votes_resutls)
        else:
            y_hat = []
            for i in range(results.shape[1]):
                cur_y = self.vote(results[:, i], self.num_vote)
                y_hat.append(cur_y)
            # return np.array(y_hat), nn_index_results
            return np.array(y_hat)

    def vote(self, y_hat_list, num_vote):
        if y_hat_list.sum() >= num_vote:
            return 1
        else:
            return 0