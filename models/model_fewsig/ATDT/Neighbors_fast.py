import numpy as np

class NeighborsFast:
    def __init__(self, dm, train_p_index, train_n_index, is_include_self=True):
        self.dm = np.copy(dm)
        if is_include_self:
            np.fill_diagonal(self.dm, self.dm.max()+1, wrap=False)
        self.dm_p = self.dm[:, train_p_index]
        self.dm_n = self.dm[:, train_n_index]
        self.train_p_index_set = set(train_p_index)
        self.train_n_index_set = set(train_n_index)
        self.dm_p_sort = np.sort(self.dm_p, axis = 1)
        self.dm_n_sort = np.sort(self.dm_n, axis = 1)
        self.pre_kp = -1
        self.pre_kn = -1

    def get_nn_ave_dist(self, hold_out_index, kp, kn):
        train_index = [i for i in range(self.dm.shape[1]) if i != hold_out_index]
        if kp < len(self.train_p_index_set) and kn < len(self.train_n_index_set):
            if hold_out_index in self.train_p_index_set:
                cur_train_p_index = [i for i in self.train_p_index_set if i != hold_out_index]
                if kn != self.pre_kn:
                    self.pre_kn = kn
                    self.pre_dm_n_kn_ave = self.dm_n_sort[:, :kn].mean(axis=1)
                n_results = self.pre_dm_n_kn_ave[train_index]
                cur_dm_p = np.sort(self.dm[:, cur_train_p_index], axis=1)
                p_mean = cur_dm_p[:, :kp].mean(axis=1)
                p_results = p_mean[train_index]

            elif hold_out_index in self.train_n_index_set:
                cur_train_n_index = [i for i in self.train_n_index_set if i != hold_out_index]
                # cur_train_p_index = list(self.train_p_index_set)
                if kp != self.pre_kp:
                    self.pre_kp = kp
                    self.pre_dm_p_kp_ave = self.dm_p_sort[:, :kp].mean(axis=1)
                p_results = self.pre_dm_p_kp_ave[train_index]

                cur_dm_n = np.sort(self.dm[:, cur_train_n_index], axis=1)
                n_mean = cur_dm_n[:, :kn].mean(axis=1)
                n_results = n_mean[train_index]

            return p_results, n_results

        else:
            'Will include the self'
            raise RuntimeError


    def get_loo_test_dist(self, kp, kn):
        if kp < len(self.train_p_index_set) and kn < len(self.train_n_index_set):
            p_results = self.dm_p_sort[:, :kp].mean(axis=1)
            n_results = self.dm_n_sort[:, :kn].mean(axis=1)
            return p_results, n_results
        else:
            'Will include the self'
            raise RuntimeError

    def get_nn_ave_dist_all(self, kp, kn):
        if kp < len(self.train_p_index_set) and kn < len(self.train_n_index_set):
            n_results = self.dm_n_sort[:, :kn].mean(axis=1)
            p_results = self.dm_p_sort[:, :kp].mean(axis=1)
            return p_results, n_results
        else:
            'Will include the self'
            raise RuntimeError