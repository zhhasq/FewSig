# -*- coding: utf-8 -*-

import os

import numpy as np

import torch
from models.model_sstsc.utils.utils import get_config_from_json
from models.model_sstsc.optim.train import supervised_train
from itertools import combinations

'''
'MiddlePhalanxOutlineAgeGroup', 
'ProximalPhalanxOutlineAgeGroup', 
'SwedishLeaf', 
'MixedShapesRegularTrain', 
'Crop'
'''


class SSTSCOL:
    @staticmethod
    def predict(opt, log_path,  train_set, test_set):
        # x_train, y_train, x_val, y_val, x_test, y_test, opt.nb_class, _ \
        #     = load_ucr2018(opt.ucr_path, opt.dataset_name)

        # opt = parse_option()
        os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu

        exp = 'exp-cls'

        # Seeds = [0,1,2,3,4]
        Seeds = [42]
        # Runs = range(0, 1, 1)

        aug1 = ['magnitude_warp']
        aug2 = ['time_warp']

        config_dict = get_config_from_json('{}/{}_config.json'.format(
            opt.config_dir, opt.dataset_name))       #从json获取配置

        opt.class_type = config_dict['class_type']
        opt.piece_size = config_dict['piece_size']
        #
        #opt.pretrained_epoch = config_dict['pretrained_epoch']
        #opt.epochs = config_dict['epochs']
        #opt.alpha = config_dict['alpha']

        if opt.model_name == 'SemiPF':
            model_paras='label{}_{}'.format(opt.label_ratio, opt.alpha)
        else:
            model_paras='label{}'.format(opt.label_ratio)

        if aug1 == aug2:
            opt.aug_type = [aug1]   #扩充类型
        elif type(aug1) is list:
            opt.aug_type = aug1 + aug2
        else:
            opt.aug_type = [aug1, aug2]

        log_dir = '{}/results/{}/{}/{}/{}'.format(log_path,
            exp, opt.dataset_name, opt.model_name, model_paras)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # file2print_detail_train = open("{}/train_detail.log".format(log_dir), 'a+')
        # print(datetime.datetime.now(), file=file2print_detail_train)
        # print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_label\tAcc_unlabel\tEpoch_max", file=file2print_detail_train)
        # file2print_detail_train.flush()
        #
        # file2print = open("{}/test.log".format(log_dir), 'a+')
        # print(datetime.datetime.now(), file=file2print)
        # print("Dataset\tAcc_mean\tAcc_std\tEpoch_max",
        #       file=file2print)
        # file2print.flush()
        #
        # file2print_detail = open("{}/test_detail.log".format(log_dir), 'a+')
        # print(datetime.datetime.now(), file=file2print_detail)
        # print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_max\tEpoch_max",
        #       file=file2print_detail)
        # file2print_detail.flush()

        ACCs = {}

        MAX_EPOCHs_seed = {}
        ACCs_seed = {}
        all_predictions = []
        for seed in Seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            opt.ckpt_dir = '{}/ckpt/{}/{}/{}/{}/{}/{}'.format(log_dir,
                exp, opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),
                model_paras, str(seed))

            if not os.path.exists(opt.ckpt_dir):
                os.makedirs(opt.ckpt_dir)

            # print('[INFO] Running at:', opt.dataset_name)

            # x_train, y_train, x_val, y_val, x_test, y_test, opt.nb_class, _ \
            #     = load_ucr2018(opt.ucr_path, opt.dataset_name)


            ACCs_run={}
            MAX_EPOCHs_run = {}

            'Sheng: split on positive samples'
            train_ts_data = train_set.train_ts_data
            test_ts_data = test_set.test_ts_data
            train_labels = train_set.get_labels()
            test_labels = test_set.get_labels()

            train_ps_index = [i for i in range(len(train_set)) if train_set.get_labels()[i] == 1]
            train_ns_index = [i for i in range(len(train_set)) if train_set.get_labels()[i] == 0]
            test_index = [i for i in range(len(test_set))]
            'split input trainset into labeled and validation'
            tmp = int(len(train_ps_index)/2)
            num_lp_train = np.max((tmp, len(train_ps_index)-tmp))
            num_lp_val = np.min((tmp, len(train_ps_index)-tmp))

            all_possible_p_val = []
            train_ps_index_set = set()
            for c in combinations(train_ps_index, num_lp_val):
                if c[0] not in train_ps_index_set and c[1] not in train_ps_index_set:
                    all_possible_p_val.append(c)
                    train_ps_index_set.add(c[0])
                    train_ps_index_set.add(c[1])

            nb_dims = opt.ts_dim
            nb_timesteps = int(train_ts_data.shape[1] / nb_dims)
            input_shape = (nb_timesteps, nb_dims)

            for run in range(len(all_possible_p_val)):
                'First construct validation set'
                cur_val_p = list(all_possible_p_val[run])
                val_p_set = set(cur_val_p)
                np.random.shuffle(train_ns_index)
                num_val_n = int(len(train_ns_index) / 4)
                val_index = cur_val_p + train_ns_index[:num_val_n]

                train_l_n_index = train_ns_index[num_val_n:]
                train_l_p_index = [x for x in train_ps_index if x not in val_p_set]
                train_l_index = train_l_n_index + train_l_p_index
                train_u_index = test_index[:len(test_set)-1]

                test_index = [test_index[-1]]

                x_val = train_ts_data[val_index, :]
                y_val = train_labels[val_index]
                x_train = np.vstack((train_ts_data[train_l_index, :], test_ts_data[train_u_index,:]))
                y_train = np.concatenate((train_labels[train_l_index], test_labels[train_u_index]))
                x_test = test_ts_data[test_index, :]
                y_test = test_labels[test_index]

                x_val = x_val.reshape((-1, input_shape[0], input_shape[1]))
                x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))
                x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
                labeled_index_in_Train = [i for i in range(len(train_l_index))]

                x_train_max = np.max(x_train)
                x_train_min = np.min(x_train)
                x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
                # Test is secret
                x_val = 2. * (x_val - x_train_min) / (x_train_max - x_train_min) - 1.
                x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.
                ################
                ## Train #######
                ################
                if opt.model_name == 'SupCE':
                    acc_test, epoch_max = supervised_train(
                        x_train, y_train, x_val, y_val, x_test, y_test, opt)
                    acc_unlabel=0
                elif 'SemiIntra' in opt.model_name:
                    acc_test, acc_unlabel, epoch_max = train_SemiIntra(
                        x_train, y_train, x_val, y_val, x_test, y_test,opt)
                # elif 'SemiInter' in opt.model_name:
                #     acc_test, acc_unlabel, epoch_max = train_SemiInter(
                #         x_train, y_train, x_val, y_val, x_test, y_test,opt)
                elif 'SemiInterOL' in opt.model_name:
                    acc_test, acc_unlabel, epoch_max, best_predictions = train_SemiInter_OL(
                        x_train, y_train, x_val, y_val, x_test, y_test, opt, labeled_index_in_Train)
                elif 'Forecast' in opt.model_name:
                    acc_test, acc_unlabel, epoch_max = train_Forecasting(
                        x_train, y_train, x_val, y_val, x_test, y_test,opt)
                elif 'SemiPF' in opt.model_name:
                    acc_test, acc_unlabel, epoch_max = train_SemiInterPF(
                        x_train, y_train, x_val, y_val, x_test, y_test,opt)
                elif 'Pseudo' in opt.model_name:
                    acc_test, acc_unlabel, acc_ws, epoch_max = train_pseudo(
                        x_train, y_train, x_val, y_val, x_test, y_test, opt)
                elif 'Vat' in opt.model_name:
                    acc_test, acc_unlabel, acc_ws, epoch_max = train_vat(
                        x_train, y_train, x_val, y_val, x_test, y_test, opt)
                elif 'Pi' in opt.model_name:
                    acc_test, acc_unlabel, acc_ws, epoch_max = train_pi(
                        x_train, y_train, x_val, y_val, x_test, y_test, opt)

                # print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                #     opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
                #     seed, round(acc_test, 2), round(acc_unlabel, 2), epoch_max),
                #     file=file2print_detail_train)
                # file2print_detail_train.flush()


                ACCs_run[run] = acc_test
                MAX_EPOCHs_run[run] = epoch_max
                all_predictions.append(best_predictions)
            ACCs_seed[seed] = round(np.mean(list(ACCs_run.values())), 2)
            MAX_EPOCHs_seed[seed] = np.max(list(MAX_EPOCHs_run.values()))

            # print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            #     opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
            #     seed, ACCs_seed[seed], MAX_EPOCHs_seed[seed]),
            #     file=file2print_detail)
            # file2print_detail.flush()

        ACCs_seed_mean = round(np.mean(list(ACCs_seed.values())), 2)
        ACCs_seed_std = round(np.std(list(ACCs_seed.values())), 2)
        MAX_EPOCHs_seed_max = np.max(list(MAX_EPOCHs_seed.values()))

        # print("{}\t{}\t{}\t{}".format(
        #     opt.dataset_name, ACCs_seed_mean, ACCs_seed_std, MAX_EPOCHs_seed_max),
        #     file=file2print)
        # file2print.flush()
        all_predictions = np.array(all_predictions)
        if all_predictions.sum() >= int(len(all_predictions) / 2):
            return 1
        else:
            return 0





