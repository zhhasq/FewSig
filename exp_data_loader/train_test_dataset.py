import numpy as np
from torch.utils.data import Dataset

from utils.time_series_utils import z_norm


class TrainSet(Dataset):

    def __init__(self, train_features, train_labels, train_arid, snr, dist_info=None, train_ts_data=None):
        self.train_features = train_features
        self.train_labels = train_labels
        self.train_arid = np.array(train_arid)
        self.num_cp = self.get_num_cp()
        self.num_cn = self.get_num_cn()
        self.snr = snr
        self.train_ts_data = train_ts_data
        self.dist_info = dist_info
        self.cp_index_set = None
        self.cn_index_set = None


    def select(self, keep_index_set):
        select_train_features = self.train_features[:, keep_index_set]
        select_train_features = select_train_features[keep_index_set, :]
        select_train_labels = self.train_labels[keep_index_set]
        self.train_arid = np.array(self.train_arid)
        select_train_arid = self.train_arid[keep_index_set]
        if self.train_ts_data.ndim == 3:
            select_train_ts_data = self.train_ts_data[:, keep_index_set, :]
        else:
            select_train_ts_data = self.train_ts_data[keep_index_set, :]
        return TrainSet(select_train_features, select_train_labels, select_train_arid, self.snr, self.dist_info, select_train_ts_data)

    def select_ids(self, keep_id_set):
        keep_index_set = []
        for i in range(len(self.train_arid)):
            if self.train_arid[i] in keep_id_set:
                keep_index_set.append(i)
        return self.select(keep_index_set), keep_index_set

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        return self.train_features[idx, :], self.train_labels[idx]

    def get_cp_index_set(self):
        if self.cp_index_set is None:
            self.cp_index_set = set([i for i in range(len(self.train_labels)) if self.train_labels[i] == 1])

        return self.cp_index_set

    def get_cn_index_set(self):
        if self.cn_index_set is None:
            self.cn_index_set = set([i for i in range(len(self.train_labels)) if self.train_labels[i] == 0])
        return self.cn_index_set

    def get_id(self, index):
        return self.train_arid[index]

    def get_ids(self):
        return self.train_arid

    def get_num_cp(self):
        return  self.train_labels.sum()

    def get_num_cn(self):
        return len(self.train_labels) - self.train_labels.sum()

    def get_dist_info(self):
        return self.dist_info

    def __str__(self):
        if self.train_features is None:
            return f"Train dataset, number of samples: {len(self.train_labels)}, number of features: not available" + \
                   f" number of positives: {self.num_cp}, number of negatives: {self.num_cn}"
        else:
            return f"Train dataset, number of samples: {len(self.train_labels)}, number of features: {self.train_features.shape[1]}" + \
                   f" number of positives: {self.num_cp}, number of negatives: {self.num_cn}"

    def get_features(self):
        return self.train_features

    def get_labels(self):
        return self.train_labels
    def get_ts_data(self):
        return self.train_ts_data

    def get_train_cp(self):
        p_index = []
        for i in range(len(self.train_labels)):
            cur_train_label = self.train_labels[i]
            if cur_train_label == 1:
                p_index.append(i)
        p_features = []

        for cur_p_index in p_index:
            p_features.append(self.train_features[cur_p_index, p_index])

        p_features = np.array(p_features)
        p_labels = self.train_labels[p_index]
        return p_features, p_labels

    def norm_features(self):
        self.train_features = z_norm(self.train_features)



class TestSet(Dataset):

    def __init__(self, test_features, test_labels, test_arid, snr, dist_info=None, test_ts_data=None):
        self.test_features = test_features
        self.test_labels = test_labels
        self.test_arid = np.array(test_arid)
        self.num_cp = self.get_num_cp()
        self.num_cn = self.get_num_cn()
        self.snr = snr
        self.test_ts_data = test_ts_data
        self.dist_info = dist_info


    def select(self, keep_index_set):
        select_test_features = self.test_features[:, keep_index_set]
        select_test_labels = self.test_labels
        self.test_arid = np.array(self.test_arid)
        select_test_arid = self.test_arid
        select_test_ts_data = self.test_ts_data

        return TestSet(select_test_features, select_test_labels, select_test_arid, self.snr, self.dist_info, select_test_ts_data)

    def select_ids(self, keep_id_set, keep_train_id_set):
        keep_index_set = []
        for i in range(len(self.test_arid)):
            if self.test_arid[i] in keep_id_set:
                keep_index_set.append(i)
        return self.select2(keep_index_set, keep_train_id_set)

    def select2(self, keep_test_index_set, keep_train_index_set=None):
        if keep_train_index_set is not None:
            select_test_features = self.test_features[:, keep_train_index_set]
        else:
            select_test_features = self.test_features
        select_test_features = select_test_features[keep_test_index_set, :]

        select_test_labels = self.test_labels[keep_test_index_set]
        self.test_arid = np.array(self.test_arid)
        select_test_arid = self.test_arid[keep_test_index_set]
        if self.test_ts_data.ndim == 3:
            select_test_ts_data = self.test_ts_data[:, keep_test_index_set, :]
        else:
            select_test_ts_data = self.test_ts_data[keep_test_index_set, :]


        return TestSet(select_test_features, select_test_labels, select_test_arid, self.snr, self.dist_info, select_test_ts_data)

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, idx):
        return self.test_features[idx, :], self.test_labels[idx]

    def get_ids(self):
        return self.test_arid

    def get_num_cp(self):
        return self.test_labels.sum()

    def get_num_cn(self):
        return len(self.test_labels) - self.test_labels.sum()

    def get_features(self):
        return self.test_features

    def get_labels(self):
        return self.test_labels

    def get_ts_data(self):
        return self.test_ts_data

    def get_dist_info(self):
        return self.dist_info

    def norm_features(self):
        self.test_features = z_norm(self.test_features)

    def __str__(self):
        if self.test_features is None:
            return f"Test dataset, number of samples: {len(self.test_labels)}, number of features: not available" + \
                   f" number of positives: {self.num_cp}, number of negatives: {self.num_cn}"
        else:
            return f"Test dataset, number of samples: {len(self.test_labels)}, number of features: {self.test_features.shape[1]}" + \
                   f" number of positives: {self.num_cp}, number of negatives: {self.num_cn}"
