import numpy as np

class NeighborsFast2:
    def __init__(self, dm, train_p_index, train_n_index, kp_list, kn_list, is_include_self=True):
        self.dm = np.copy(dm)
        if is_include_self:
            np.fill_diagonal(self.dm, self.dm.max()+1, wrap=False)
        self.dm_p = self.dm[:, train_p_index]
        self.dm_n = self.dm[:, train_n_index]
        self.train_p_index_set = set(train_p_index)
        self.train_n_index_set = set(train_n_index)
        self.kp_list = kp_list
        self.kn_list = kn_list
        if self.kp_list[-1] >= len(self.train_p_index_set) or self.kn_list[-1] >= len(self.train_n_index_set):
            'Will include the self'
            raise RuntimeError
        self.dm_p_sort = np.sort(self.dm_p, axis = 1)
        self.dm_n_sort = np.sort(self.dm_n, axis = 1)
        self.loo_results_p = np.zeros((len(self.kp_list), self.dm.shape[0], self.dm.shape[0]-1))
        self.loo_results_n = np.zeros((len(self.kn_list), self.dm.shape[0], self.dm.shape[0]-1))
        self.test_loo_results_p = np.zeros((len(self.kp_list), self.dm.shape[0]))
        self.test_loo_results_n = np.zeros((len(self.kn_list), self.dm.shape[0]))

        for i in range(dm.shape[0]):
            self.gen_nnd_one_hold_out(i)

    def gen_nnd_one_hold_out(self, hold_out_index):
        train_index = [i for i in range(self.dm.shape[1]) if i != hold_out_index]

        if hold_out_index in self.train_p_index_set:
            cur_dm_kn = self.dm_n_sort[train_index, :]
            for j in range(len(self.kn_list)-1, -1, -1):
                cur_kn = self.kn_list[j]
                cur_dm_kn = cur_dm_kn[:, :cur_kn]
                self.loo_results_n[j, hold_out_index, :] = cur_dm_kn.mean(axis=1)
                self.test_loo_results_n[j, hold_out_index] = self.dm_n_sort[hold_out_index, :cur_kn].mean()

            cur_train_p_index = [i for i in self.train_p_index_set if i != hold_out_index]
            tmp = self.dm[:, cur_train_p_index]
            tmp = np.sort(tmp, axis=1)
            cur_dm_kp = tmp[train_index, :]
            for i in range(len(self.kp_list)-1, -1, -1):
                cur_kp = self.kp_list[i]
                cur_dm_kp = cur_dm_kp[:, :cur_kp]
                self.loo_results_p[i, hold_out_index, :] = cur_dm_kp.mean(axis=1)
                self.test_loo_results_p[i, hold_out_index] = tmp[hold_out_index, :cur_kp].mean()

        elif hold_out_index in self.train_n_index_set:
            cur_dm_kp = self.dm_p_sort[train_index, :]
            for i in range(len(self.kp_list) - 1, -1, -1):
                cur_kp = self.kp_list[i]
                cur_dm_kp = cur_dm_kp[:, :cur_kp]
                self.loo_results_p[i, hold_out_index, :] = cur_dm_kp.mean(axis=1)
                t = self.dm_p_sort[hold_out_index, :cur_kp].mean()
                self.test_loo_results_p[i, hold_out_index] = t

            cur_train_n_index = [i for i in self.train_n_index_set if i != hold_out_index]
            tmp = self.dm[:, cur_train_n_index]
            tmp = np.sort(tmp, axis=1)
            cur_dm_kn = tmp[train_index, :]
            for j in range(len(self.kn_list) - 1, -1, -1):
                cur_kn = self.kn_list[j]
                cur_dm_kn = cur_dm_kn[:, :cur_kn]
                self.loo_results_n[j, hold_out_index, :] = cur_dm_kn.mean(axis=1)
                self.test_loo_results_n[j, hold_out_index] = tmp[hold_out_index, :cur_kn].mean()


    def get_nn_ave_dist_all(self, kp, kn):
        if kp < len(self.train_p_index_set) and kn < len(self.train_n_index_set):
            n_results = self.dm_n_sort[:, :kn].mean(axis=1)
            p_results = self.dm_p_sort[:, :kp].mean(axis=1)
            return p_results, n_results
        else:
            'Will include the self'
            raise RuntimeError

    def get_loo_p_nnd(self, kp):
        kp_i = np.where(self.kp_list == kp)[0][0]
        return self.loo_results_p[kp_i], self.test_loo_results_p[kp_i]

    def get_loo_n_nnd(self, kn):
        kn_i = np.where(self.kn_list == kn)[0][0]
        return self.loo_results_n[kn_i], self.test_loo_results_n[kn_i]