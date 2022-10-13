import numpy as np

from exp_data_loader.train_test_dataset import TrainSet

class SuccessBase:
    def __init__(self, E, F, dm):
        self.E = E
        self.F = F
        self.dm = dm
        self.E_index = [x for x in range(len(self.E))]
        self.F_index = [self.E_index[-1] + 1 + x for x in range(len(self.F))]
        self.clusters = [{x} for x in self.E_index]
        self.base_cluster_labels = self.E.get_labels()

    def fit(self):
        max_E_index = self.E_index[-1]
        new_clusters = []
        dist_list = []
        for i in self.F_index:
            for j in range(self.dm.shape[1]):
                dist_list.append((i,j,self.dm[i, j]))
        dist_sorted = sorted(dist_list, key= lambda x:x[2])

        'index, set'
        F_cluster = dict()
        for cur_f_index in self.F_index:
            F_cluster[cur_f_index] = {cur_f_index}

        'index, label'
        complete_F_dict = dict()

        for x in dist_sorted:
            if x[0] == x[1]:
                continue
            if x[1] <= max_E_index:
                'X[1] in E'
                if complete_F_dict.get(x[0]) is None:
                    self.clusters[x[1]].add(x[0])
                    F_cluster[x[0]] = self.clusters[x[1]]
                    complete_F_dict[x[0]] = self.base_cluster_labels[x[1]]
                else:
                    'x[0] have a label, so it is in base cluster, then it can not merge again'
                    pass
            else:
                'x[1] is in F'
                if complete_F_dict.get(x[0]) is not None and complete_F_dict.get(x[1]) is not None:
                    'x[0] and x[1] both have a label, no matter whether they are same, cannot merge'
                    pass
                elif complete_F_dict.get(x[1]) is not None:
                    'x[1] have a label already'
                    x0_cluster = F_cluster.get(x[0])
                    x1_cluster = F_cluster.get(x[1])
                    for tmp in x0_cluster:
                        complete_F_dict[tmp] = complete_F_dict.get(x[1])
                        x1_cluster.add(tmp)
                        F_cluster[tmp] = x1_cluster

                elif complete_F_dict.get(x[0]) is not None:
                    'x[0] have a label already'
                    x0_cluster = F_cluster.get(x[0])
                    x1_cluster = F_cluster.get(x[1])
                    for tmp in x1_cluster:
                        complete_F_dict[tmp] = complete_F_dict.get(x[0])
                        x0_cluster.add(tmp)
                        F_cluster[tmp] = x0_cluster
                else:
                    'neither x[0] and x[1] has a label'
                    x0_cluster = F_cluster.get(x[0])
                    x1_cluster = F_cluster.get(x[1])
                    new_c = set()
                    new_c= new_c.union(x0_cluster, x1_cluster)
                    F_cluster[x[0]] = new_c
                    F_cluster[x[1]] = new_c
        F_labels = []
        for i in self.F_index:
            F_labels.append(complete_F_dict.get(i))
        return F_labels

class SuccessST:
    def __init__(self, E, F, data_loader):
        self.E = E
        self.F = F
        self.E_ids = self.E.get_ids()
        self.F_ids = self.F.get_ids()
        self.E_id_set = set(self.E_ids)
        self.F_id_set = set(self.F_ids)
        self.all_ids = np.concatenate((self.E_ids, self.F_ids))
        self.data_loader = data_loader
        self.dm = self.get_dm()
        self.model = SuccessBase(self.E, self.F, self.dm)

    def get_dm(self):
        all_index = self.data_loader.get_index(self.all_ids)
        cur_dm = np.copy(self.data_loader.dm)
        cur_dm = cur_dm[:, all_index]
        cur_dm = cur_dm[all_index, :]
        return cur_dm

    def fit(self):
        F_labels = self.model.fit()
        E_labels = self.E.get_labels()
        labels = [x for x in E_labels] + [x for x in F_labels]
        return TrainSet(self.dm, np.array(labels), self.all_ids, self.E.snr, self.E.dist_info, None)


    # def label_F(self):
    #     tmp = self.model.fit(self.dm)
    #     tmp
    #     labels_from_cluster = tmp.labels_
    #     cluster_count_dict = dict()
    #     for i in range(len(self.all_ids)):
    #         cur_label = labels_from_cluster[i]
    #         cur_id = self.all_ids[i]
    #         if cluster_count_dict.get(cur_label) is None:
    #             cluster_count_dict[cur_label] = [cur_id]
    #         else:
    #             tmp = cluster_count_dict.get(cur_label)
    #             tmp.append(cur_id)
    #
    #     're-assign the labels'
    #     E_labels = self.E.get_labels()
    #     E_id_label_dict = dict()
    #     for i in range(len(E_labels)):
    #         E_id_label_dict[self.E_ids[i]] = E_labels[i]
    #
    #     cluster_new_labels = dict()
    #     for k, v in cluster_count_dict.items():
    #         v_true_labels = []
    #         for x in v:
    #             if x in self.E_id_set:
    #                 v_true_labels.append(E_id_label_dict.get(x))
    #         v_true_labels = np.array(v_true_labels)
    #         if v_true_labels.sum() > (1-v_true_labels).sum():
    #             cluster_new_labels[k] = 1
    #         else:
    #             cluster_new_labels[k] = 0
    #
    #     'final labels for F'
    #     F_labels = []
    #     for i in range(len(self.all_ids)):
    #         cur_label = labels_from_cluster[i]
    #         cur_id = self.all_ids[i]
    #         if cur_id in self.F_id_set:
    #             F_labels.append(cluster_new_labels.get(cur_label))
    #     return F_labels
