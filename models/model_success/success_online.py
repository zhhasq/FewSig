import time

import numpy as np

from exp_data_loader.train_test_dataset import TestSet
from models.model_success.success_st import SuccessST
from tqdm import tqdm

from models.model_wei.exp_Fdist_knn import ExpFDistKNN


class SuccessOL:

    def evaluate(self, train_set, test_set, loader, log_path=None, record_time=False):
        self.loader = loader
        self.model = ExpFDistKNN(1)
        # self.assist_model = SuccessST()
        init_start_t = time.time()
        self.model.fit(train_set.get_features(), train_set.get_labels())
        init_end_t = time.time()
        if record_time:
            self.init_time = init_end_t - init_start_t
        # # self.assist_model.fit(train_set.get_features(), train_set.get_labels())
        self.train_data = train_set.get_features()
        self.train_label = train_set.get_labels()
        self.train_set = train_set

        DM = self.loader.dm
        new_train_set = self.train_set
        y_hat = []
        # for i in range(len(test_set)):
        test_start_time = time.time()
        for i in tqdm(range(len(test_set))):
            # print(f"{i / len(test_set):.4f}")
            'First do prediction'
            cur_test_id = test_set.get_ids()[i]
            cur_test_index = self.loader.get_index([cur_test_id])[0]
            all_index = self.loader.get_index(new_train_set.get_ids())
            cur_test = DM[cur_test_index, all_index]
            cur_y_hat = self.model.predict([cur_test])[0]
            y_hat.append(cur_y_hat)
            'Then re-train the model with current Future set'
            test_labels = test_set.get_labels()[:i+1]
            test_arids = test_set.get_ids()[:i + 1]
            cur_F = TestSet(None, test_labels, test_arids, test_set.snr, dist_info=None, test_ts_data=None)
            success = SuccessST(self.train_set, cur_F, self.loader)
            new_train_set = success.fit()
            self.model = ExpFDistKNN(1)
            self.model.fit(new_train_set.get_features(), new_train_set.get_labels())
        test_end_time = time.time()
        if record_time:
            self.ave_round_t = (test_end_time-test_start_time)/len(test_set)
        pseudo_labels = []
        new_train_labels = []
        test_labels = test_set.get_labels()
        test_ids = test_set.get_ids()
        cur_F = TestSet(None, test_labels, test_ids, test_set.snr, dist_info=None, test_ts_data=None)
        success = SuccessST(self.train_set, cur_F, self.loader)
        new_train_set = success.fit()

        new_train_id_label_dict = dict()
        new_train_id = new_train_set.get_ids()
        new_train_labels = new_train_set.get_labels()
        for i in range(len(new_train_id)):
            cur_id = new_train_id[i]
            new_train_id_label_dict[cur_id] = new_train_labels[i]
        F_ids = test_set.get_ids()
        for cur_id in F_ids:
            cur_pseudo_label = new_train_id_label_dict.get(cur_id)
            if cur_pseudo_label is None:
                raise  RuntimeError
            pseudo_labels.append(cur_pseudo_label)


        return np.array(y_hat), np.array(pseudo_labels)

