import argparse
import copy
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from exp_data_loader.train_test_dataset import TrainSet, TestSet
from exp_data_loader.UEA_loader import UEAUtils

from models.model_DTWD.dtwd_online import DTWDOL
from models.model_fewsig.dist_select import ExpDistSelect
from models.model_fewsig.fewsig import FewSig
from models.model_sstsc.train_ssl_ol_UEA import SSTSCUEAOL
from models.model_success.success_online import SuccessOL
from models.model_wei.wei_online import WeiOL
from math import comb
from tqdm import tqdm, trange

from utils import scores
from utils.tools import log

class NotEnoughPS(Exception):
    pass

class OnlineExp:
    def split_helper(self, loop_max, num_init_ps,  ps_ids, ps_ori_class, ns_ori_class, max_same):
        train = []
        test = []
        count = 0
        tried_count = 0
        all_selection_p = []
        while count < loop_max:
            if tried_count > 300000:
                print(f"Tried 300000 times, stop, number of possible split {len(all_selection_p)}")
                break
            tried_count += 1
            if len(ps_ori_class) == 1:
                num_init_ps_each_ori_class = num_init_ps
            else:
                num_init_ps_each_ori_class = int(num_init_ps / len(ps_ori_class)) + 1
            # cur_select = np.random.choice(ps_ids_shuffle, num_init_ps, replace=False)
            cur_select = []
            for key, v in ps_ori_class.items():
                np.random.shuffle(v)
                cur_select.extend(v[:num_init_ps_each_ori_class])
            cur_select_set = set(cur_select)

            next = False
            for pre_sele in all_selection_p:
                if len(pre_sele.intersection(cur_select_set)) >= max_same:
                    next = True
                    break
            if not next:
                all_selection_p.append(cur_select_set)
                cur_train_ns = []
                cur_test_ns = []
                for k, v in ns_ori_class.items():
                    l = len(v)
                    hl = int(l/2)
                    np.random.shuffle(v)
                    cur_train_ns.extend(v[:hl])
                    cur_test_ns.extend(v[hl:])

                cur_select_train_id = np.concatenate((cur_select, cur_train_ns))
                cur_test_ps = np.array([x for x in ps_ids if x not in cur_select_set])
                cur_select_test_id = np.concatenate((cur_test_ps, cur_test_ns))

                np.random.shuffle(cur_select_train_id)
                train.append(cur_select_train_id)
                np.random.shuffle(cur_select_test_id)
                test.append(cur_select_test_id)
                count += 1
        return np.array(train), np.array(test)

    def _gen_split2(self, ps_ids, ns_ids, num_init_ps, split_max, max_same, ori_labels):
        'Each class split balance'
        ps_ori_labels = ori_labels[ps_ids]
        ps_ori_labels_set = set(ps_ori_labels)
        if len(ps_ori_labels_set) > 1:
            if self.data_name == "Fungi":
                if len(ps_ori_labels_set) != 2:
                    raise RuntimeError
            elif self.data_name == "PigAirwayPressure":
                if len(ps_ori_labels_set) != 3:
                    raise RuntimeError
            elif self.data_name == "PigArtPressure":
                if len(ps_ori_labels_set) != 3:
                    raise RuntimeError
            elif self.data_name == "PigCVP":
                if len(ps_ori_labels_set) != 3:
                    raise RuntimeError
            else:
                raise RuntimeError
        num_train_ns = int(len(ns_ids) / 2)
        ns_ids_copy = np.copy(ns_ids)

        ps_ids_shuffle = np.copy(ps_ids)
        ps_ori_class = dict()
        for p_id in ps_ids_shuffle:
            ori_label = ori_labels[p_id]
            if ps_ori_class.get(ori_label) is None:
                ps_ori_class[ori_label] = [p_id]
            else:
                tmp = ps_ori_class.get(ori_label)
                tmp.append(p_id)

        for k in ps_ori_labels_set:
            print(f"Select true label: {k}, num of samples: {len(ps_ori_class.get(k))}")

        ns_ori_class = dict()
        for n_id in ns_ids_copy:
            ori_label = ori_labels[n_id]
            if ns_ori_class.get(ori_label) is None:
                ns_ori_class[ori_label] = [n_id]
            else:
                tmp = ns_ori_class.get(ori_label)
                tmp.append(n_id)
        # np.random.shuffle(ps_ids_shuffle)
        count = 0
        if len(ps_ids_shuffle) <= num_init_ps:
            print(f"Total number positive samples {len(ps_ids_shuffle)} less than initial ps size")
            raise NotEnoughPS
        num_comb = comb(len(ps_ids_shuffle), num_init_ps)
        if num_comb < self.batch:
            print(f"Total number combination {num_comb} less than batch size {self.batch}")
            max_same = num_init_ps
            loop_max = num_comb
            train, test = self.split_helper(loop_max, num_init_ps, ps_ids, ps_ori_class, ns_ori_class, max_same)
        else:
            loop_max = np.min((num_comb, split_max))
            train, test = self.split_helper(loop_max, num_init_ps,  ps_ids, ps_ori_class, ns_ori_class, max_same)
            while train.shape[0] < self.batch:
                max_same = max_same + 1
                if max_same == num_init_ps:
                    break
                print(f"max same increase to {max_same}")
                train, test = self.split_helper(loop_max, num_init_ps, ps_ids, ps_ori_class, ns_ori_class, max_same)
        return train, test

    def _split_check(self, ps_ids, ns_ids, num_init_ps, split_max, max_same, ori_labels):
        'Each class split balance'
        ps_ori_labels = ori_labels[ps_ids]
        ps_ori_labels_set = set(ps_ori_labels)
        if len(ps_ori_labels_set) > 1:
            if self.data_name == "Fungi":
                if len(ps_ori_labels_set) != 2:
                    raise RuntimeError
            elif self.data_name == "PigAirwayPressure":
                if len(ps_ori_labels_set) != 3:
                    raise RuntimeError
            elif self.data_name == "PigArtPressure":
                if len(ps_ori_labels_set) != 3:
                    raise RuntimeError
            elif self.data_name == "PigCVP":
                if len(ps_ori_labels_set) != 3:
                    raise RuntimeError
            else:
                raise RuntimeError
        num_train_ns = int(len(ns_ids) / 2)
        ns_ids_copy = np.copy(ns_ids)

        ps_ids_shuffle = np.copy(ps_ids)
        ps_ori_class = dict()
        for p_id in ps_ids_shuffle:
            ori_label = ori_labels[p_id]
            if ps_ori_class.get(ori_label) is None:
                ps_ori_class[ori_label] = [p_id]
            else:
                tmp = ps_ori_class.get(ori_label)
                tmp.append(p_id)

        for k in ps_ori_labels_set:
            print(f"Select true label: {k}, num of samples: {len(ps_ori_class.get(k))}")

        ns_ori_class = dict()
        for n_id in ns_ids_copy:
            ori_label = ori_labels[n_id]
            if ns_ori_class.get(ori_label) is None:
                ns_ori_class[ori_label] = [n_id]
            else:
                tmp = ns_ori_class.get(ori_label)
                tmp.append(n_id)
        # np.random.shuffle(ps_ids_shuffle)
        count = 0
        if len(ps_ids_shuffle) <= num_init_ps:
            print(f"Total number positive samples {len(ps_ids_shuffle)} less than initial ps size")
            raise NotEnoughPS

        return


    def __init__(self, data_name, data_path_args, algo_name, num_init_ps, batch, max_same=None,
                 FewSig_args=None, sstsc_args=None, gpu_num=1, loader=None, record_time=False, select_check=False):
        print()
        print(f"******************** {data_name} ***************************")
        self.sstsc_args = sstsc_args
        self.split_max = 3000
        self.gpu_count = gpu_num
        self.data_name = data_name
        self.UEA_root_path = data_path_args.get("UEA_data")
        self.input_loader = loader
        self.record_time = record_time
        self.batch = batch
        self.algo_name = algo_name
        self.uts_source = data_path_args.get("source")
        self.dist_core = data_path_args.get("dist_core")
        self.dist_batch_size = data_path_args.get("dist_batch_size")
        if loader is None:
            self.data_load = UEAUtils.get_UEA_data(self.data_name, self.UEA_root_path, self.uts_source, self.dist_core, self.dist_batch_size)
            self.split = f"split_p{num_init_ps}"
            self.split_path = os.path.join(self.data_load.data_dir, self.split)
        else:
            self.data_load = loader
            self.split_path = loader.get_split_path(num_init_ps)
        self.num_init_ps = num_init_ps

        if max_same is None:
            max_same = self.num_init_ps
        if not os.path.exists(self.split_path):
            ps_ids = self.data_load.ps_id
            ns_ids = self.data_load.ns_id
            train, test = self._gen_split2(ps_ids, ns_ids, self.num_init_ps, self.split_max, max_same,
                                          self.data_load.ori_labels)
            if train is None:
                raise NotEnoughPS
            print(f"There are total {train.shape[0]} different splits of train/test")


            os.mkdir(self.split_path)
            np.save(os.path.join(self.split_path, "train_id_all.npy"), train)
            np.save(os.path.join(self.split_path, "test_id_all.npy"), test)
        else:
            if select_check:
                'This is just for checking the selection'
                ps_ids = self.data_load.ps_id
                ns_ids = self.data_load.ns_id
                if self.data_load.ori_labels is not None:
                    self._split_check(ps_ids, ns_ids, self.num_init_ps, self.split_max, max_same,
                                                  self.data_load.ori_labels)
            tmp = np.load(os.path.join(self.split_path, "train_id_all.npy"))
            print(f"There are total {tmp.shape[0]} different splits of train/test")

        if record_time:
            if self.algo_name == "SSTSC":
                self.time_log_path = os.path.join(self.split_path, f"batch_{batch}_timelog.log")
            else:
                self.time_log_path = os.path.join(self.split_path, f"{self.algo_name}_batch_{batch}_timelog.log")
        else:
            self.time_log_path = None

        self.train_ids_all = np.load(os.path.join(self.split_path, "train_id_all.npy"))
        self.test_ids_all = np.load(os.path.join(self.split_path, "test_id_all.npy"))

        result_name = self.algo_name
        if self.algo_name == 'Wei':
            self.algo = WeiOL()
            self.dist_code = 'E-Q100-S100'
        elif self.algo_name == "DTWD":
            self.algo = DTWDOL()
            self.dist_code_dtw = 'D-Q100-S100-W0'
            self.dist_code_eu = 'E-Q100-S100'
        elif self.algo_name == "SUCCESS":
            self.algo = SuccessOL()
            self.dist_code = 'D-Q100-S100-W0'
        elif self.algo_name == "SSTSC":
            self.algo = SSTSCUEAOL()
            self.dist_code = None
        elif self.algo_name == "FewSig":
            if FewSig_args is None:
                raise RuntimeError
            target_FPR = FewSig_args.get("target_FPR")
            LR = FewSig_args.get("LR")
            init_gpu_id = FewSig_args.get("init_gpu_count")
            if init_gpu_id is None:
                init_gpu_id = 0
            self.algo = FewSig(target_FPR, LR, gpu_num, init_gpu_id)
            result_name = f"{self.algo_name}_{target_FPR}_{LR}"


        self.batch_path = os.path.join(self.split_path, f"batch_{self.batch}.npy")
        if not os.path.exists(self.batch_path):
            tmp = [x for x in range(self.train_ids_all.shape[0])]
            if len(tmp) < self.batch:
                self.select_split_id = tmp
            else:
                np.random.shuffle(tmp)
                self.select_split_id = tmp[:self.batch]
            np.save(self.batch_path, self.select_split_id)
        else:
            self.select_split_id = np.load(self.batch_path)

        self.resutls_path = os.path.join(self.split_path, result_name)
        if not os.path.exists(self.resutls_path):
            os.mkdir(self.resutls_path)

    def start(self):
        if self.algo_name == "SSTSC":
            return self.start_sstsc()
        results = []
        for i in trange(len(self.select_split_id), desc=f'Testing {self.algo_name}@{self.data_name}'):
            cur_exp_id = self.select_split_id[i]
            cur_results_path = os.path.join(self.resutls_path, f"{cur_exp_id}.npy")
            if not os.path.exists(cur_results_path) or self.record_time:
                cur_train_id = self.train_ids_all[cur_exp_id, :]
                cur_test_id = self.test_ids_all[cur_exp_id, :]
                # cur_test_id = np.array(cur_test_id, dtype=int)
                if self.algo_name == "DTWD":
                    if self.input_loader is not None:
                        data_loader_dtw = copy.deepcopy(self.input_loader)
                        data_loader_eu = copy.deepcopy(self.input_loader)
                    else:
                        data_loader_dtw = UEAUtils.get_UEA_data(self.data_name, self.UEA_root_path, self.uts_source, self.dist_core, self.dist_batch_size)

                        data_loader_eu = UEAUtils.get_UEA_data(self.data_name, self.UEA_root_path, self.uts_source, self.dist_core, self.dist_batch_size)

                    cur_train_dtw, cur_test_dtw = data_loader_dtw.get_train_test_from_id(self.dist_code_dtw,
                                                                                         cur_train_id,
                                                                                         cur_test_id)
                    cur_train_eu, cur_test_eu = data_loader_eu.get_train_test_from_id(self.dist_code_eu,
                                                                                      cur_train_id,
                                                                                      cur_test_id)
                    y_hat, pseudo_label = self.algo.evaluate(cur_train_dtw, cur_test_dtw, cur_train_eu, cur_test_eu,
                                                             data_loader_dtw, data_loader_eu, self.record_time)
                    test_labels = cur_test_dtw.get_labels()
                    if self.record_time:
                        log_time = [
                            f"{cur_exp_id},{self.algo.init_time},{self.algo.ave_round_t}"]
                        log(log_time, self.time_log_path, print_console=False, clear=False)
                        continue

                elif self.algo_name == "FewSig":
                    dist_code_list = []
                    query = [98, 96, 94, 92, 90, 88, 86, 84, 82, 80]
                    warp = [1, 3, 5, 7, 9, 10, 12, 14, 16, 18, 20, 25, 30]
                    dist_code_list = []
                    for q in query:
                        for w in warp:
                            dist_code_list.append(f"D-Q{q}-S100-W{w}")
                    for w in warp:
                        dist_code_list.append(f"D-Q100-S100-W{w}")
                    dist_code_list.append(f"E-Q100-S100")

                    losses, min_loss, best_dist_code = ExpDistSelect.select_best_dist_code_UEA(self.data_load, cur_train_id, cur_test_id,
                                                                                               dist_code_list)
                    # print(f"Best dist code: {best_dist_code}")
                    cur_train, cur_test = self.data_load.get_train_test_from_id(best_dist_code, cur_train_id,
                                                                                cur_test_id)
                    if self.record_time:
                        cur_results = None
                        self.algo.evaluate(cur_train, cur_test, self.data_load, record_time=self.record_time)
                        log_time = [
                            f"{cur_exp_id},{self.algo.init_time},{self.algo.test_time},{self.algo.retrain_time}"]
                        log(log_time, self.time_log_path, print_console=False, clear=False)
                        continue
                    else:
                        y_hat, pseudo_label = self.algo.evaluate(cur_train, cur_test, self.data_load)
                        test_labels = cur_test.get_labels()
                        r = [scores.score_pr(x, test_labels) for x in y_hat]
                        for i in range(len(r)):
                            print(r[i])
                else:
                    cur_train, cur_test = self.data_load.get_train_test_from_id(self.dist_code, cur_train_id,
                                                                                cur_test_id)

                    y_hat, pseudo_label = self.algo.evaluate(cur_train, cur_test, self.data_load,
                                                             log_path=self.time_log_path, record_time=self.record_time)
                    test_labels = cur_test.get_labels()
                    if self.algo_name == "SUCCESS" and self.record_time:
                        log_time = [
                            f"{cur_exp_id},{self.algo.init_time},{self.algo.ave_round_t}"]
                        log(log_time, self.time_log_path, print_console=False, clear=False)
                        continue
                    if self.algo_name == "Wei" and self.record_time:
                        log_time = [
                            f"{cur_exp_id},{self.algo.init_time},{self.algo.ave_round_t}"]
                        log(log_time, self.time_log_path, print_console=False, clear=False)
                        continue

                if self.algo_name == "FewSig":
                    cur_results = []
                    cur_results.append(test_labels)
                    for x in y_hat:
                        cur_results.append(x)
                    cur_results.append(pseudo_label)
                else:
                    cur_results = [test_labels, y_hat, pseudo_label]

                cur_results = np.array(cur_results)
                np.save(cur_results_path, cur_results)


            else:
                cur_results = np.load(cur_results_path)


            results.append(cur_results)
        self.results = results
        return results

    def load_results_part(self):
        results = []
        for i in trange(len(self.select_split_id), desc=f'Testing {self.algo_name}@{self.data_name}'):
            cur_exp_id = self.select_split_id[i]
            cur_results_path = os.path.join(self.resutls_path, f"{cur_exp_id}.npy")
            if not os.path.exists(cur_results_path):
                continue
            cur_results = np.load(cur_results_path)
            results.append(cur_results)
        print(f"Partial load, total round {len(results)}")
        self.results = results
        return results

    def start_sstsc(self):
        process = self.sstsc_args.get("process_num")
        batch_size = self.sstsc_args.get("batch_size")

        results = []
        val_scores = []
        if self.record_time and os.path.exists(self.time_log_path):
            return
        pool = Pool(processes=process)
        for i in trange(len(self.select_split_id), desc=f'Testing {self.algo_name}@{self.data_name}'):
            cur_exp_id = self.select_split_id[i]
            cur_results_path = os.path.join(self.resutls_path, f"{cur_exp_id}.npy")
            result_async = []
            # if self.record_time and (not os.path.exists(self.time_log_path))
            if self.record_time:
                cur_train_id = self.train_ids_all[cur_exp_id, :]
                cur_test_id = self.test_ids_all[cur_exp_id, :]
                cur_train, cur_test = self.data_load.get_train_test_from_id(self.dist_code, cur_train_id,
                                                                            cur_test_id)
                test_labels = cur_test.get_labels()
                val_index = []
                p_val = 0
                n_val = 0
                i = 0

                while p_val < self.num_init_ps or n_val < int(len(cur_test) / 4):
                    if test_labels[i] == 1 and p_val < self.num_init_ps:
                        val_index.append(i)
                        p_val += 1
                    elif test_labels[i] == 0 and n_val < int(len(cur_test) / 4):
                        val_index.append(i)
                        n_val += 1
                    i += 1
                    if i >= len(test_labels):
                        print(f"{self.algo_name} doesn't have enough p samples for validation")
                        return None

                new_test_index = [x for x in range(len(cur_test)) if x not in set(val_index)]
                val_set = TrainSet(None, test_labels[val_index],
                                   cur_test.get_ids()[val_index],
                                   cur_test.snr, dist_info=None, train_ts_data=cur_test.test_ts_data[val_index, :])
                new_test_set = TestSet(None, test_labels[new_test_index],
                                       cur_test.get_ids()[new_test_index],
                                       cur_test.snr, dist_info=None,
                                       test_ts_data=cur_test.test_ts_data[new_test_index, :])

                new_test_labels = new_test_set.get_labels()
                new_test_ts = new_test_set.test_ts_data
                start_time = time.time()
                round_count = 0
                i=1

                cur_test_set = TestSet(None, new_test_labels[:i + 1], None, -1, dist_info=None,
                                       test_ts_data=new_test_ts[:i + 1, :])
                init_round_start_time = time.time()
                OnlineExp.sstsc_multi_gpu_helper(0, cur_train, cur_test_set, val_set, batch_size, self.sstsc_args)
                init_round_end_time = time.time()

                mid_iter_start_time = time.time()
                i = int(len(new_test_labels)/2)
                cur_test_set = TestSet(None, new_test_labels[:i + 1], None, -1, dist_info=None,
                                       test_ts_data=new_test_ts[:i + 1, :])
                OnlineExp.sstsc_multi_gpu_helper(0, cur_train, cur_test_set, val_set, batch_size, self.sstsc_args)
                mid_iter_end_time = time.time()
                log_time = [f"{cur_exp_id},{init_round_end_time-init_round_start_time},{mid_iter_end_time-mid_iter_start_time},{len(test_labels)}"]
                log(log_time, self.time_log_path, print_console=False, clear=False)
                continue

            if not os.path.exists(cur_results_path):
                cur_train_id = self.train_ids_all[cur_exp_id, :]
                cur_test_id = self.test_ids_all[cur_exp_id, :]
                cur_train, cur_test = self.data_load.get_train_test_from_id(self.dist_code, cur_train_id,
                                                                                cur_test_id)
                test_labels = cur_test.get_labels()
                val_index = []
                p_val = 0
                n_val = 0
                i = 0

                while p_val < self.num_init_ps or n_val < int(len(cur_test)/4):
                    if test_labels[i] == 1 and p_val < self.num_init_ps:
                        val_index.append(i)
                        p_val += 1
                    elif test_labels[i] == 0 and n_val < int(len(cur_test)/4):
                        val_index.append(i)
                        n_val += 1
                    i += 1
                    if i >= len(test_labels):
                        print(f"{self.algo_name} doesn't have enough p samples for validation")
                        return None

                new_test_index = [x for x in range(len(cur_test)) if x not in set(val_index)]
                # val_set = TrainSet(cur_test.get_features()[val_index,:], test_labels[val_index], cur_test.get_ids()[val_index],
                #                    cur_test.snr, dist_info=None, train_ts_data=cur_test.test_ts_data[val_index, :])
                # new_test_set = TestSet(cur_test.get_features()[new_test_index,:], test_labels[new_test_index], cur_test.get_ids()[new_test_index],
                #                    cur_test.snr, dist_info=None, test_ts_data=cur_test.test_ts_data[new_test_index, :])
                val_set = TrainSet(None, test_labels[val_index],
                                   cur_test.get_ids()[val_index],
                                   cur_test.snr, dist_info=None, train_ts_data=cur_test.test_ts_data[val_index, :])
                new_test_set = TestSet(None, test_labels[new_test_index],
                                       cur_test.get_ids()[new_test_index],
                                       cur_test.snr, dist_info=None,
                                       test_ts_data=cur_test.test_ts_data[new_test_index, :])

                new_test_labels = new_test_set.get_labels()
                new_test_ts = new_test_set.test_ts_data

                for i in range(1, len(new_test_set)):
                    gpu_num = i % self.gpu_count
                    cur_test_set = TestSet(None, new_test_labels[:i+1], None, -1, dist_info=None, test_ts_data=new_test_ts[:i+1, :])
                    result_async.append(pool.apply_async(OnlineExp.sstsc_multi_gpu_helper, args=(
                        gpu_num, cur_train, cur_test_set, val_set, batch_size, self.sstsc_args)))

                y_hat = []
                true_labels = []
                # val_scores = []
                cur_val_scores = []
                for j in tqdm(range(len(result_async))):
                    r = result_async[j].get()
                    # print(j, r, test_labels[1 + j])
                    y_hat.append(r[0])
                    true_labels.append(r[1])
                    cur_val_score = r[2]
                    cur_val_scores.append(cur_val_score)
                val_scores.append(np.array(cur_val_scores))
                # cur_results = [new_test_labels[1:], y_hat, None]
                for i in range(len(true_labels)):
                    if true_labels[i] != new_test_labels[i+1]:
                        raise RuntimeError
                cur_results = [true_labels, y_hat]
                cur_results = np.array(cur_results)
                np.save(cur_results_path, cur_results)
                cur_val_results_path = os.path.join(self.resutls_path, f"{cur_exp_id}_val.npy")
                np.save(cur_val_results_path, np.array(cur_val_score))
            else:
                cur_results = np.load(cur_results_path)
                cur_val_scores = np.load(os.path.join(self.resutls_path, f"{cur_exp_id}_val.npy"))
                val_scores.append(cur_val_scores)

            results.append(cur_results)

        pool.close()
        print("waiting joined")
        pool.join()
        print("joined")

        self.results = results
        self.sstsc_val_score = val_scores
        return results, val_scores

    @staticmethod
    def sstsc_multi_gpu_helper(gpu_num, train_set, test_set, val_set, batch=256, addi_args=None):
        opt = OnlineExp.parse_option()
        setattr(opt, 'batch_size', batch)
        setattr(opt, 'model_name', "SemiInterOL")
        setattr(opt, 'label_ratio', -1)
        sstsc_root = os.path.join(os.getcwd(), "SSTSC_dir")
        if not os.path.exists(sstsc_root):
            os.makedirs(sstsc_root)

        if addi_args.get("config_path") is None:
            setattr(opt, 'config_dir', os.path.join(sstsc_root, "config"))
        else:
            setattr(opt, 'config_dir', addi_args.get("config_path"))
        if addi_args.get("ucr_path") is None:
            setattr(opt, 'ucr_path', os.path.join(sstsc_root, "datasets"))
        else:
            setattr(opt, 'ucr_path', addi_args.get("ucr_path"))

        if addi_args.get("ckpt_dir") is None:
            setattr(opt, 'ckpt_dir', os.path.join(sstsc_root, "ckpt"))
        else:
            setattr(opt, 'ckpt_dir', addi_args.get("ckpt_dir"))

        if addi_args.get("log_path") is None:
            log_path = os.path.join(sstsc_root, "log")
        else:
            log_path = addi_args.get("log_path")

        setattr(opt, 'gpu', f"{gpu_num}")
        cuda_d = torch.device(f'cuda:{gpu_num}')
        algo = SSTSCUEAOL()
        cur_y_hat, val_score = algo.evaluate(opt, log_path, train_set, test_set, val_set)
        return cur_y_hat, test_set.get_labels()[-1], val_score

    def get_ave_final_F1(self, vote=None):
        f1_all = []
        for i in range(len(self.results)):
            cur_r = self.results[i]
            # if self.algo_name == "SSTSC":
            #     s = scores.score_pr(np.array(cur_r[0]), np.array(cur_r[1]))
            # else:
            if self.algo_name == "FewSig":
                print(i)
                if vote is None:
                    tmp = []
                    for i in range(1,5,1):
                        s = scores.score_pr(cur_r[i, :], cur_r[0, :])
                        print(s)
                        tmp.append(s[-1])
                    f1_all.append(np.array(tmp).mean())
                else:
                    tmp = []
                    s = scores.score_pr(cur_r[vote, :], cur_r[0, :])

                    tmp.append(s[-1])
                    f1_all.append(np.array(tmp).mean())

            elif self.algo_name == "SSTSC":
                include_val = self.sstsc_args.get("include_val_score")
                if include_val:
                    cur_val = self.sstsc_val_score[i][-1, :]
                    s = scores.score_pr(cur_r[1, :], cur_r[0, :])
                    tp = s[0] + cur_val[0]
                    tn = s[1] + cur_val[1]
                    fp = s[2] + cur_val[2]
                    fn = s[3] + cur_val[3]
                    recall = tp / (tp + fn)
                    if tp == 0:
                        precision = 0
                    else:
                        precision = tp / (tp + fp)
                    if fp == 0:
                        fpr = 0
                    else:
                        fpr = fp / (fp + tn)
                    if precision + recall == 0:
                        f1 = 0
                    else:
                        f1 = (2 * precision * recall) / (precision + recall)
                    f1_all.append(f1)
                else:
                    s = scores.score_pr(cur_r[1, :], cur_r[0, :])
                    # tp, tn, fp, fn, recall, precision, fpr, f1
                    f1_all.append(s[-1])
            else:
                s = scores.score_pr(cur_r[1, :], cur_r[0, :])
                    # tp, tn, fp, fn, recall, precision, fpr, f1
                f1_all.append(s[-1])
            m = np.array(f1_all).mean()
            if math.isnan(m):
                print()
        return f1_all, m



    def load_results(self):
        pass

    @staticmethod
    def parse_option():
        parser = argparse.ArgumentParser('argument for training')  # 创建对象
        parser.add_argument('--save_freq', type=int, default=200,
                            help='save frequency')  # 添加参数  保存频率
        parser.add_argument('--batch_size', type=int, default=256,
                            help='batch_size')
        parser.add_argument('--K', type=int, default=4, help='Number of augmentation for each sample')  # 每个样本的扩增数量
        parser.add_argument('--alpha', type=float, default=0.3, help='Past future split point')  # 过去的未来分割点

        parser.add_argument('--feature_size', type=int, default=64,
                            help='feature_size')
        parser.add_argument('--num_workers', type=int, default=16,
                            help='num of workers to use')
        parser.add_argument('--epochs', type=int, default=1000,
                            help='number of training epochs')
        parser.add_argument('--patience', type=int, default=200,
                            help='training patience')
        parser.add_argument('--aug_type', type=str, default='none', help='Augmentation type')  # 扩充类型
        parser.add_argument('--piece_size', type=float, default=0.2,
                            help='piece size for time series piece sampling')  # 时间序列的样片大小
        parser.add_argument('--stride', type=float, default=0.2,
                            help='stride for forecast model')
        parser.add_argument('--horizon', type=float, default=0.1,
                            help='horizon for forecast model')

        parser.add_argument('--class_type', type=str, default='3C', help='Classification type')  # 分类类型
        parser.add_argument('--gpu', type=str, default='0', help='gpu id')

        # optimization   优化
        parser.add_argument('--learning_rate', type=float, default=0.01,
                            help='learning rate')
        # model dataset
        parser.add_argument('--dataset_name', type=str, default='ECG200',
                            choices=['CricketX',
                                     'CricketY',
                                     'CricketZ'
                                     'ECG200',
                                     'ECG5000',
                                     'CBF',
                                     'UWaveGestureLibraryAll',
                                     'InsectWingbeatSound',
                                     'MFPT', 'XJTU',
                                     'EpilepticSeizure',
                                     'SwedishLeaf',
                                     'WordSynonyms',
                                     'ACSF1'
                                     ],
                            help='dataset')
        parser.add_argument('--nb_class', type=int, default=3,
                            help='class number')
        parser.add_argument('--n_class', type=int, default=2)
        # Hyper-parameters for vat model
        parser.add_argument('--n_power', type=int, default=4, metavar='N',
                            help='the iteration number of power iteration method in VAT')
        parser.add_argument('--xi', type=float, default=3, metavar='W', help='xi for VAT')
        parser.add_argument('--eps', type=float, default=1.0, metavar='W', help='epsilon for VAT')

        # ucr_path = '../datasets/UCRArchive_2018'
        parser.add_argument('--ucr_path', type=str, default='./datasets',
                            help='Data root for dataset.')  # 数据集的数据根
        parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                            help='Data path for checkpoint.')  # 检查点的数据路径

        parser.add_argument('--weight_rampup', default=30, type=int, metavar='EPOCHS',
                            help='the length of rampup weight (default: 30)')
        parser.add_argument('--usp_weight', default=1.0, type=float, metavar='W',
                            help='the upper of unsuperivsed weight (default: 1.0)')
        # method
        parser.add_argument('--backbone', type=str, default='SimConv4')
        parser.add_argument('--model_name', type=str, default='Pi',
                            choices=['SupCE', 'SemiIntra', 'SemiInter', 'Forecast', 'SemiPF', 'Pi'],
                            help='choose method')
        # 选择模型
        parser.add_argument('--config_dir', type=str, default='./config', help='The Configuration Dir')  # 配置目录
        parser.add_argument('--label_ratio', type=float, default=0.4,
                            help='label ratio')

        parser.add_argument('--aftershock', type=bool, default=True, help='Is testing data aftershock')
        parser.add_argument('--ts_dim', type=int, default=1, help='Time series dimension')

        opt = parser.parse_args()
        return opt