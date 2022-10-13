import os

import numpy as np
from sktime.datasets import load_UCR_UEA_dataset, load_from_tsfile

from exp_data_loader.metric import CorrSearchInfo, EuSearchInfo, DTWSearchInfo
from exp_data_loader.train_test_dataset import TrainSet, TestSet

class UAESelect:
    @staticmethod
    def get_select1(dist_code=None):
        select_set = []
        select_warp = []
        with open(os.path.join(os.getcwd(), "exp_data", "UAE_select1.txt"), 'r') as f:
            lines = f.readlines()
        for cur_line in lines:
            if "%" in cur_line:
                continue
            tmp = cur_line.split(',')
            select_set.append(tmp[0])
            select_warp.append(int(tmp[1]))

        select_data = []
        for name in select_set:
            select_data.append(UAEUtils.get_UAE_data(name))

        # dist_code = 'D-Q100-S100-W100'
        if dist_code is not None:
            for cur_data in select_data:
                cur_data.get_dm(dist_code)
        return select_data

class UAELoader:
    def __init__(self, name, path, ts_data, labels, id, ori_labels=None, num_process=20, batch=500):
        self.name = name
        self.data_dir = path
        self.ts_data = ts_data
        self.labels = labels
        self.ori_labels = ori_labels
        self.id = id
        self.num_process = num_process
        self.batch_size = batch
        self.dm = None
        self.id_index_dict = self.id_to_index_dict()
        self.ps_id = np.array([self.id[i] for i in range(len(self.id))  if self.labels[i] == 1])
        self.ns_id = np.array([self.id[i] for i in range(len(self.id)) if self.labels[i] == 0])

    def id_to_index_dict(self):
        results = dict()
        for i in range(len(self.id)):
            results[self.id[i]] = i
        return results

    def get_index(self, arid_list):
        results = []
        for cur_arid in arid_list:
            results.append(self.id_index_dict.get(cur_arid))
        return results

    def __str__(self):
        return f"Name:{self.name}\n\tpositive samples:{len(self.ps_id)}\n\tnegative samples{len(self.ns_id)}\n\ttime series length:{self.ts_data.shape[1]}"

    def get_dm(self, dist_code, down_sample=None, re_compute=False):
        self.dist_code = dist_code
        if re_compute or not os.path.exists(os.path.join(self.data_dir, f"{dist_code}-dm.npy")):
            'C-Q80-S100'
            tmp = dist_code.split('-')
            ts_len = self.ts_data.shape[1]
            'arrival index is middle point'
            arrival_index = int(0 + (ts_len - 1 - 0) / 2)
            query_percent = int(tmp[1][1:])
            query_len = int(ts_len * query_percent/100)
            query_start = int(query_len / 2)
            query_end = query_start
            search_percent = int(tmp[2][1:])
            search_len = int(ts_len * search_percent/100)
            search_start = int(search_len / 2)
            search_end = search_start

            if tmp[0] == "C":
                metric = CorrSearchInfo(False, 1, arrival_index, ts_len,
                                        query_start, query_end, search_start, search_end, self.num_process, self.batch_size)
            elif tmp[0] == "E":
                metric = EuSearchInfo(False, 1, arrival_index, ts_len,
                                      query_start, query_end, search_start, search_end, self.num_process, self.batch_size)
            elif tmp[0] == "D":
                'C-Q80-S100-W100'
                warp = float(tmp[3][1:])
                if warp < 1 and warp > 0:
                    warp = int(warp * query_len)
                elif warp >= 1:
                    warp = int(warp)
                    if warp >= int(query_len/2):
                        warp = int(query_len/2)
                elif warp == 0:
                    warp = int(query_len/2)

                metric = DTWSearchInfo(warp, False, 1, arrival_index, ts_len,
                                       query_start, query_end, search_start, search_end, self.num_process, self.batch_size)
            print(metric)
            if down_sample is not None:
                dm = metric.execute(self.ts_data[:, ::down_sample])
                np.save(os.path.join(self.data_dir, f"{dist_code}-dm-down_{down_sample}.npy"), dm)
            else:
                dm = metric.execute(self.ts_data)
                if not re_compute:
                    np.save(os.path.join(self.data_dir, f"{dist_code}-dm.npy"), dm)

        else:
            if down_sample is not None:
                dm = np.load(os.path.join(self.data_dir, f"{dist_code}-dm-down_{down_sample}.npy"))
            else:
                dm = np.load(os.path.join(self.data_dir, f"{dist_code}-dm.npy"))
        if np.isnan(dm).any():
            # dm = np.nan_to_num(dm, nan=dm.max())
            raise RuntimeError
        self.dm = dm
        return dm

    def get_train_test_from_id(self, dist_code, train_id, test_id):
        if dist_code is not None:
            if self.dm is None:
                self.get_dm(dist_code)
            if self.dist_code != dist_code:
                self.dist_code = dist_code
                self.get_dm(dist_code)
            train_features = self.dm[:, train_id]
            train_features = train_features[train_id, :]
            train_ts_data = self.ts_data[train_id, :]
            train_labels = self.labels[train_id]
            train_set = TrainSet(train_features, train_labels, train_id, -1, dist_info=None,
                                 train_ts_data=train_ts_data)

            test_features = self.dm[:, train_id]
            test_features = test_features[test_id, :]
            test_labels = self.labels[test_id]
            test_ts_data = self.ts_data[test_id, :]

            if self.ori_labels is not None:
                train_ori_labels = self.ori_labels[train_id]
                train_ori_label_count = dict()
                for x in train_ori_labels:
                    if train_ori_label_count.get(x) is None:
                        train_ori_label_count[x] = 1
                    else:
                        train_ori_label_count[x] = train_ori_label_count.get(x) + 1
                l_train = [(class_name, count) for class_name, count in train_ori_label_count.items()]
                l_train = sorted(l_train, key=lambda x: x[1])

                test_ori_labels = self.ori_labels[test_id]
                test_ori_label_count = dict()
                for x in test_ori_labels:
                    if test_ori_label_count.get(x) is None:
                        test_ori_label_count[x] = 1
                    else:
                        test_ori_label_count[x] = test_ori_label_count.get(x) + 1
                l_test = [(class_name, count) for class_name, count in test_ori_label_count.items()]
                l_test = sorted(l_test, key=lambda x: x[1])
                if len(l_train) < len(l_test):
                    print()

            test_set = TestSet(test_features, test_labels, test_id, -1, dist_info=None, test_ts_data=test_ts_data)
            return train_set, test_set
        else:
            train_ts_data = self.ts_data[train_id, :]
            train_labels = self.labels[train_id]
            train_set = TrainSet(None, train_labels, train_id, -1, dist_info=None,
                                 train_ts_data=train_ts_data)


            test_labels = self.labels[test_id]
            test_ts_data = self.ts_data[test_id, :]

            test_set = TestSet(None, test_labels, test_id, -1, dist_info=None, test_ts_data=test_ts_data)
            return train_set, test_set


    def get_train_test(self, dist_code, num_p):
        if self.dm is None:
            self.get_dm(dist_code)
        if self.dist_code != dist_code:
            self.dist_code = dist_code
            self.get_dm(dist_code)
        ps_id = np.array([self.id[i] for i in range(len(self.labels)) if self.labels[i] == 1])
        ns_id = np.array([self.id[i] for i in range(len(self.labels)) if self.labels[i] == 0])
        np.random.shuffle(ps_id)
        np.random.shuffle(ns_id)
        num_n = int(len(ns_id)/2)
        train_id = np.concatenate((ps_id[:num_p], ns_id[:num_n]))
        train_features = self.dm[:, train_id]
        train_features = train_features[train_id, :]
        train_ts_data = self.ts_data[train_id, :]
        train_labels = self.labels[train_id]
        train_set = TrainSet(train_features, train_labels, train_id, -1, dist_info=None, train_ts_data=train_ts_data)

        test_id = np.concatenate((ps_id[num_p:], ns_id[num_n:]))
        test_features = self.dm[:, train_id]
        test_features = test_features[test_id, :]
        test_labels = self.labels[test_id]
        test_ts_data = self.ts_data[test_id, :]

        test_set = TestSet(test_features, test_labels, test_id, -1, dist_info=None, test_ts_data=test_ts_data)
        return train_set, test_set

class UAEUtils:
    @staticmethod
    def load_UAE_from_disk(name, uts_source=None, verbose=True):
        if uts_source is None:
            data_path = os.path.join("/home/zhongs/btc_mount/Univariate_ts", name)
        else:
            data_path = os.path.join(uts_source, name)
        if not os.path.exists(data_path):
            print(f"\t{name} not exists")
            return None
        else:
            if os.path.exists(os.path.join(data_path, f"{name}_TRAIN.ts")):
                try:
                    train_x, train_y = load_from_tsfile(
                        os.path.join(data_path, f"{name}_TRAIN.ts"), return_data_type="numpy2d"
                    )
                    test_x, test_y = load_from_tsfile(
                        os.path.join(data_path, f"{name}_TEST.ts"), return_data_type="numpy2d"
                    )
                    ts_data = np.vstack((train_x, test_x))
                    labels = np.concatenate((train_y, test_y))
                    return UAEUtils.check_criteria2(name, ts_data, labels, verbose)
                except ValueError:
                    print(f"\tnot all series were of equal length")
                    return None
            # elif os.path.exists(os.path.join(data_path, f"{name}_TRAIN.arff")):
            #     try:
            #         train_x, train_y = load_from_arff_to_dataframe(
            #             os.path.join(data_path, f"{name}_TRAIN.arff")
            #         )
            #         train_y = np.array(train_y)
            #         train_ts_data_tmp = train_x.to_numpy()
            #         train_ts_data = []
            #         for cur_row in train_ts_data_tmp:
            #             one_row_values = []
            #             for v in cur_row[0]:
            #                 one_row_values.append(v)
            #             train_ts_data.append(one_row_values)
            #         train_ts_data = np.array(train_ts_data)
            #
            #         test_x, test_y = load_from_arff_to_dataframe(
            #             os.path.join(data_path, f"{name}_TEST.arff")
            #         )
            #         test_y = np.array(test_y)
            #         test_ts_data_tmp = test_x.to_numpy()
            #         test_ts_data = []
            #         for cur_row in test_ts_data_tmp:
            #             one_row_values = []
            #             for v in cur_row[0]:
            #                 one_row_values.append(v)
            #             test_ts_data.append(one_row_values)
            #         test_ts_data = np.array(test_ts_data)
            #
            #         ts_data = np.vstack((train_ts_data, test_ts_data))
            #         labels = np.concatenate((train_y, test_y))
            #         return UAEUtils.check_criteria2(name, ts_data, labels)
            #
            #     except ValueError:
            #         print(f"\tnot all series were of equal length")
            #         return None
            else:
                print(f"\t{name} not exists")
                return None

    @staticmethod
    def load_UAE(name):
        try:
            X, y = load_UCR_UEA_dataset(name=name)
        except FileNotFoundError:
            print(f"\t{name} not exists")
            return None
        # r = UAEUtils.check_criteria1(X, y)
        r = UAEUtils.check_criteria2(X, y)
        return r

    @staticmethod
    def check_criteria1(X, y):
        print(f"\tdim:{X.shape[1]}, number of samples:{X.shape[0]}")
        if X.shape[1] > 1:
            print("\tDrop reason: multivariate")
            return None
        label_count = dict()
        for x in y:
            if label_count.get(x) is None:
                label_count[x] = 1
            else:
                label_count[x] = label_count.get(x) + 1
        print(f"\t{label_count}")
        if len(label_count) > 2:
            print("\t Drop reason: class number > 2")
            return None
        l = [(class_name, count) for class_name, count in label_count.items()]
        l = sorted(l, key=lambda x: x[1])
        p = l[0][0]
        n = l[1][0]
        print(f"\tPositive label:{l[0][0]} count: {l[0][1]}")
        print(f"\tNegative label:{l[1][0]} count: {l[1][1]}")
        if l[0][1] > (1 / 3) * (l[0][1] + l[1][1]):
            print("\tDrop reason: Too large the positive set")
            return None
        labels = []
        for v in y:
            if v == p:
                labels.append(1)
            elif v == n:
                labels.append(0)
        labels = np.array(labels)
        ts_data_tmp = X.to_numpy()
        ts_data = []
        for cur_row in ts_data_tmp:
            one_row_values = []
            for v in cur_row[0]:
                one_row_values.append(v)
            ts_data.append(one_row_values)
        ts_data = np.array(ts_data)
        print(f"\ttime series length: {ts_data.shape[1]}")
        return [ts_data, labels]

    @staticmethod
    def check_criteria2(name, ts_data, labels, verbose=True):
        if verbose:
            print(f"\tlength:{ts_data.shape[1]}, number of samples:{ts_data.shape[0]}")
        if name != "InsectSound" and ts_data.shape[0] > 800:
            print("\tToo large dataset > 800")
            return None
        label_count = dict()
        for x in labels:
            if label_count.get(x) is None:
                label_count[x] = 1
            else:
                label_count[x] = label_count.get(x) + 1
        if verbose:
            print(f"\t{label_count}")
        # if len(label_count) > 2:
        # print("\t Drop reason: class number > 2")
        # return None
        l = [(class_name, count) for class_name, count in label_count.items()]
        l = sorted(l, key=lambda x: x[1])
        if name == "OliveOil":
            p = set('4')
        elif name == "PigAirwayPressure":
            p = set(['25', '27', '35'])
        elif name == "PigArtPressure":
            p = set(['10', '22', '39'])
        elif name == "PigCVP":
            p = set(['38', '44', '46'])
        elif name == "Fungi":
            p = set(['4', '5'])


        else:
            if l[0][1] < l[1][1]:
                p = l[0][0]
            else:
                'random choose one label as positive'
                pending_p = np.array([x for x in label_count.keys()])
                np.random.shuffle(pending_p)
                p = pending_p[0]
                if verbose:
                    print(f"\tPositive label:{p} count: {label_count.get(p)}")
        # else:
        #     l = [(class_name, count) for class_name, count in label_count.items()]
        #     l = sorted(l, key=lambda x: x[1])
        #     p = l[0][0]
        #     n = l[1][0]
        #     print(f"\tPositive label:{l[0][0]} count: {l[0][1]}")
        #     print(f"\tNegative label:{l[1][0]} count: {l[1][1]}")
        # if l[0][1] > (1 / 3) * (l[0][1] + l[1][1]):
        #     print("\tDrop reason: Too large the positive set")
        #     return None
        final_labels = []
        for v in labels:
            if v == p or v in p:
                final_labels.append(1)
            else:
                final_labels.append(0)
        final_labels = np.array(final_labels)
        return [ts_data, final_labels, labels]

    @staticmethod
    def get_UAE_data(name, root, uts_source, num_process, batch):
        data_dir = os.path.join(root, name)

        'If data set not on the disk, use load_UCR_UEA_dataset to load the data'
        'else: load from disk'

        if not os.path.exists(os.path.join(data_dir,  "ts_data.npy")):
            # r = UAEUtils.load_UAE(name)
            r = UAEUtils.load_UAE_from_disk(name, uts_source)
            if r is not None:
                ts_data = r[0]
                labels = r[1]
                ori_labels = r[2]
                id = np.array([x for x in range(len(labels))])

                if np.isnan(ts_data).any():
                    print(f"{name} has None")
                    new_ts_data = []
                    new_label = []
                    new_ori_labels = []
                    for i in range(ts_data.shape[0]):
                        if np.isnan(ts_data[i, :]).any():
                            continue
                        else:
                            new_ts_data.append(ts_data[i, :])
                            new_label.append(labels[i])
                            new_ori_labels.append(ori_labels[i])
                    ts_data = np.array(new_ts_data)
                    labels = np.array(new_label)
                    ori_labels = np.array(new_ori_labels)

                ts_data = np.array(ts_data)
                labels = np.array(labels)
                ori_labels = np.array(ori_labels)
                id = np.array([x for x in range(len(labels))])

                print(f"\tSaving to {data_dir}")
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                np.save(os.path.join(data_dir, "id"), id)
                np.save(os.path.join(data_dir, "ts_data"), ts_data)
                np.save(os.path.join(data_dir, "labels"), labels)
            else:
                return None
        else:
            r = UAEUtils.load_UAE_from_disk(name, uts_source, verbose=False)
            ori_labels = r[2]
            # ori_labels = None
            tmp_data = r[0]
            if np.isnan(tmp_data).any():
                non_nan_ori_labels = []
                print(f"{name} has None")
                for i in range(tmp_data.shape[0]):
                    if np.isnan(tmp_data[i, :]).any():
                        continue
                    else:
                        non_nan_ori_labels.append(ori_labels[i])
                non_nan_ori_labels = np.array(non_nan_ori_labels)
            else:
                non_nan_ori_labels = ori_labels

            ts_data = np.load(os.path.join(data_dir, "ts_data.npy"))
            print(f"\tlength:{ts_data.shape[1]}, number of samples:{ts_data.shape[0]}")
            labels = np.load(os.path.join(data_dir, "labels.npy"))
            label_count = dict()
            for x in labels:
                if label_count.get(x) is None:
                    label_count[x] = 1
                else:
                    label_count[x] = label_count.get(x) + 1

            print(f"new labels: \t{label_count}")
            label_count = dict()
            for x in non_nan_ori_labels:
                if label_count.get(x) is None:
                    label_count[x] = 1
                else:
                    label_count[x] = label_count.get(x) + 1

            print(f"original labels: \t{label_count}")
            id = np.load(os.path.join(data_dir, "id.npy"))
            # if np.isnan(labels).any():
            #     # raise RuntimeError
            #     return None
        return UAELoader(name, data_dir, ts_data, labels, id, non_nan_ori_labels, num_process, batch)


    @staticmethod
    def load_arff(file_path, name):
        print(f"Checking {name} in {file_path}")
        ts_data = []
        labels = []
        labels_count = dict()
        with open(file_path, 'r') as f:
            lines = f.readlines()
        data_line = False
        for i in range(len(lines)):
            cur_line = lines[i]
            cur_line = cur_line.replace('\n', '')
            if "@" in cur_line and ("target" in cur_line or "attribute class" in cur_line or "ATTRIBUTE class" in cur_line):
                tmp = cur_line.split("{")
                tmp2= tmp[1].split(",")
                c2 = tmp2[1].replace('}', '')
                c2 = c2.replace(' ', '')
                c2 = c2.replace('\n', '')
                labels_count[tmp2[0]] = 0
                labels_count[c2] = 0
                print(f"\t{labels_count}")
                continue


            if cur_line == '@data' or cur_line == '@DATA':
                data_line = True
                continue
            if data_line:
                points = cur_line.split(',')
                ts_data.append([np.float(x) for x in points[:-1]])
                cur_label = points[-1]
                cur_label = cur_label.replace(' ', '')
                cur_label = cur_label.replace('\n', '')
                labels.append(cur_label)
                labels_count[cur_label] = labels_count.get(cur_label) + 1
                # print()

        l = [(class_name, count) for class_name, count in labels_count.items()]
        l = sorted(l, key=lambda x: x[1])
        p = l[0][0]
        n = l[1][0]
        print(f"\tPositive label:{l[0][0]} count: {l[0][1]}")
        print(f"\tNegative label:{l[1][0]} count: {l[1][1]}")
        if l[0][1] > (1 / 3) * (l[0][1] + l[1][1]):
            print("\tDrop reason: Too large the positive set")
            return None
        labels_convert = []
        for v in labels:
            if v == p:
                labels_convert.append(1)
            elif v == n:
                labels_convert.append(0)
        labels_convert = np.array(labels_convert)
        ts_data = np.array(ts_data)
        print(f"\ttime series length: {ts_data.shape[1]}")
        id = np.array([x for x in range(len(labels))])
        data_dir = os.path.join(os.getcwd(), "exp_data", "data_source", name)
        print(f"\tSaving to {data_dir}")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        np.save(os.path.join(data_dir, "id"), id)
        np.save(os.path.join(data_dir, "ts_data"), ts_data)
        np.save(os.path.join(data_dir, "labels"), labels_convert)