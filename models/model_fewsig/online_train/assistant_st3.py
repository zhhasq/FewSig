import time

import numpy as np
from tqdm import tqdm, trange


class AssistantST3:
    def __init__(self, model, assist_model, train_set, loader):
        self.model = model
        self.assist_model = assist_model


        self.train_data = train_set.get_features()
        self.train_label = train_set.get_labels()
        self.train_set = train_set
        self.loader = loader

    def evaluate(self, test_set, pool, record_time=False):
        self.model.fit(self.train_data, self.train_label, pool)
        self.assist_model.fit(self.train_data, self.train_label, pool)

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

        if record_time:
            test_time = []
            re_train_time = []
        # for i in range(len(test_labels)):
        for i in trange(len(test_labels), desc='ST loop'):
            if record_time and len(test_time) >= 5:
                self.test_time = np.array(test_time).mean()
                self.retrain_time = np.array(re_train_time).mean()
                return None, None, None
            # print(f"{i/len(test_labels):.4f}")
            cur_arid = test_arid[i]
            cur_index = self.loader.get_index([cur_arid])[0]
            test_start_time = time.time()
            cur_test_features = np.array([update_test_features[i, :]])
            cur_y_hat = self.model.predict(cur_test_features)
            # if len(cur_y_hat) == 1:
            #     y_hat.append(cur_y_hat[0])
            # # cur_y_hat, cur_nn_list = self.model.predict(cur_test_features)
            # else:
            #     y_hat.append(cur_y_hat)
            # nn_train_index.append(cur_nn_list)
            assist_y_hat = self.assist_model.predict(cur_test_features)[0]
            test_end_time = time.time()
            if assist_y_hat == 1:
                retrain_start = time.time()
                added_test_arid.append((cur_arid, len(update_train_labels)))
                update_train_features, update_test_features, update_train_labels, update_train_index = self.update(
                    update_train_features, update_test_features, update_train_labels, cur_index, update_train_index,
                    test_index)

                self.model.fit(update_train_features, update_train_labels, pool)
                self.assist_model.fit(update_train_features, update_train_labels, pool)
                retrain_end = time.time()
                if record_time:
                    test_time.append(test_end_time-test_start_time)
                    re_train_time.append(retrain_end-retrain_start)
                y_hat.append([1])
            else:
                if len(cur_y_hat) == 1:
                    y_hat.append(cur_y_hat[0])
                # cur_y_hat, cur_nn_list = self.model.predict(cur_test_features)
                else:
                    y_hat.append(cur_y_hat)

        if record_time:
            'in case there are less than 5 samples added'
            self.test_time = np.array(test_time).mean()
            self.retrain_time = np.array(re_train_time).mean()
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