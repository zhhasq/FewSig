import numpy as np

from models.model_fewsig.neighbors import Neighbors


class ExpDistSelect:
    @staticmethod
    def select_best_dist_code_UEA(data_loader, train_id, test_id, dist_code_list):
        losses = []

        for cur_dist_code in dist_code_list:
            train, test = data_loader.get_train_test_from_id(cur_dist_code, train_id, test_id)
            losses.append(ExpDistSelect.loss1(train))
        losses = np.array(losses)
        return losses, losses.min(), dist_code_list[np.argmin(losses)]

    @staticmethod
    def select_best_dist_code(data_loader, data_code, dist_code_list):
        losses = []
        for cur_dist_code in dist_code_list:
            train_3ch, test_3ch = data_loader.get_train_test(data_code, cur_dist_code, 20)
            losses.append(ExpDistSelect.loss1(train_3ch))
        losses = np.array(losses)
        return losses, dist_code_list[np.argmin(losses)]


    @staticmethod
    def loss1(train_set):
        ts_len = train_set.train_ts_data.shape[1]
        train_dse = train_set.get_features()
        train_labels = train_set.get_labels()
        neighbors = [Neighbors(i, train_dse[i, :]) for i in range(len(train_set))]

        ps_index_set = set([i for i in range(len(train_set)) if train_labels[i] == 1])
        ns_index_set = set([i for i in range(len(train_set)) if train_labels[i] == 0])

        loss = 0
        for i in range(len(neighbors)):
            cur_nei = neighbors[i]
            cur_label = train_labels[i]
            cur_loss = 0
            if cur_label == 1:
                cur_loss += cur_nei.get_NN(1, ps_index_set)[0].dist
                cur_loss -= cur_nei.get_NN(1, ns_index_set)[0].dist
            elif cur_label == 0:
                cur_loss += cur_nei.get_NN(1, ns_index_set)[0].dist
                cur_loss -= cur_nei.get_NN(1, ps_index_set)[0].dist
            if cur_loss > 0:
                loss += cur_loss / ts_len
        return loss