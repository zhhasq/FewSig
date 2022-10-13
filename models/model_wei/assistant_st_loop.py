import time

import numpy as np
from tqdm import tqdm

class AssistantSTLoop:
    def __init__(self, model, assist_model, train_set, loader):
        self.model = model
        self.assist_model = assist_model

        self.train_data = train_set.get_features()
        self.train_label = train_set.get_labels()
        self.train_set = train_set
        self.loader = loader

    def evaluate(self, test_set, pool=None, log_path=None, record_time=False):
        init_time_start = time.time()
        if pool is not None:
            self.model.fit(self.train_data, self.train_label, pool)
            self.assist_model.fit(self.train_data, self.train_label, pool)
        else:
            self.model.fit(self.train_data, self.train_label)
            self.assist_model.fit(self.train_data, self.train_label)
        init_time_end = time.time()
        if record_time:
            self.init_time = init_time_end - init_time_start

        test_data = test_set.get_features()
        test_labels = test_set.get_labels()

        update_train_features = self.train_data.copy()
        update_train_labels = self.train_label.copy()
        update_test_features = test_data.copy()
        test_arid = test_set.get_ids()
        test_index = self.loader.get_index(test_arid)
        update_train_index = self.loader.get_index(self.train_set.get_ids())

        y_hat = []
        nn_train_index = []
        added_test_arid = []
        added_index_set = set()
        not_added_index_set = set()
        # for i in range(len(test_labels)):
        #     print(f"{i/len(test_labels):.4f}")

        round_start_time = time.time()
        for i in tqdm(range(len(test_labels))):
            cur_arid = test_arid[i]
            cur_index = self.loader.get_index([cur_arid])[0]
            cur_test_features = np.array([update_test_features[i, :]])

            assist_y_hat = self.assist_model.predict(cur_test_features)[0]
            if assist_y_hat == 1:
                added_test_arid.append((cur_arid, len(update_train_labels)))
                update_train_features, update_test_features, update_train_labels, update_train_index = self.update(
                    update_train_features, update_test_features, update_train_labels, cur_index, update_train_index,
                    test_index)
                if pool is not None:
                    self.assist_model.fit(update_train_features, update_train_labels, pool)
                else:
                    self.assist_model.fit(update_train_features, update_train_labels)
                y_hat.append([1])
                added_index_set.add(i)
                stop = False
                while not stop:
                    stop = True
                    if len(not_added_index_set) > 0:
                        not_added_index_list = [x for x in not_added_index_set]

                        for j in not_added_index_list:
                            r = self.assist_model.predict([update_test_features[j, :]])[0]
                            if r == 1:
                                stop = False
                                added_index_set.add(j)
                                added_test_arid.append((test_arid[j], len(update_train_labels)))
                                not_added_index_set.remove(j)
                                cur_test_arid = test_arid[j]
                                cur_index = self.loader.get_index([cur_test_arid])[0]
                                update_train_features, update_test_features, update_train_labels, update_train_index = self.update(
                                    update_train_features, update_test_features, update_train_labels, cur_index,
                                    update_train_index,
                                    test_index)
                                if pool is not None:
                                    self.assist_model.fit(update_train_features, update_train_labels, pool)
                                else:
                                    self.assist_model.fit(update_train_features, update_train_labels)
                                break
                if pool is not None:
                    self.model.fit(update_train_features, update_train_labels, pool)
                else:
                    self.model.fit(update_train_features, update_train_labels)
            else:
                not_added_index_set.add(i)
                cur_y_hat = self.model.predict(cur_test_features)
                # if len(cur_y_hat) == 1:
                #     y_hat.append(cur_y_hat[0])
                # cur_y_hat, cur_nn_list = self.model.predict(cur_test_features)
                # else:
                y_hat.append(cur_y_hat)
        round_end_time = time.time()

        if record_time:
            self.ave_round_t = (round_end_time-round_start_time)/len(test_labels)
            return None, None, None

        return y_hat, nn_train_index, added_test_arid


    def update(self, train_features, test_features, train_labels, true_pending_test_index, train_index,
                     test_index):
        dm = self.loader.dm
        train_to_test_dist_arr = []
        for i in train_index:
            train_to_test_dist_arr.append(dm[i, true_pending_test_index])
        update_train_index = train_index.copy()
        update_train_index.append(true_pending_test_index)
        #update_train_labels = train_labels.copy()
        update_train_labels = np.append(train_labels, 1)

        update_train_features = np.hstack(
            (train_features, np.array(train_to_test_dist_arr).reshape((train_features.shape[0], 1))))
        pending_to_train_dist_arr = []
        for i in update_train_index:
            pending_to_train_dist_arr.append(dm[true_pending_test_index, i])
        update_train_features = np.vstack((update_train_features, np.array(pending_to_train_dist_arr)))

        test_to_train_dist_arr = []
        for i in test_index:
            test_to_train_dist_arr.append(dm[i, true_pending_test_index])
        update_test_features = np.hstack(
            (test_features, np.array(test_to_train_dist_arr).reshape((test_features.shape[0], 1))))

        return update_train_features, update_test_features, update_train_labels, update_train_index