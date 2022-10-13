# -*- coding: utf-8 -*-

import datetime
import os

import torch
from models.model_sstsc.utils.utils import get_config_from_json
from models.model_sstsc.optim.train import supervised_train


'''
'MiddlePhalanxOutlineAgeGroup', 
'ProximalPhalanxOutlineAgeGroup', 
'SwedishLeaf', 
'MixedShapesRegularTrain', 
'Crop'
'''


class SSTSC:
    @staticmethod
    def execute(opt, log_path, ):
        # opt = parse_option()
        os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu

        exp = 'exp-cls'

        Seeds = [0,1,2,3,4]
        Runs = range(0, 10, 1)

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

        file2print_detail_train = open("{}/train_detail.log".format(log_dir), 'a+')
        print(datetime.datetime.now(), file=file2print_detail_train)
        print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_label\tAcc_unlabel\tEpoch_max", file=file2print_detail_train)
        file2print_detail_train.flush()

        file2print = open("{}/test.log".format(log_dir), 'a+')
        print(datetime.datetime.now(), file=file2print)
        print("Dataset\tAcc_mean\tAcc_std\tEpoch_max",
              file=file2print)
        file2print.flush()

        file2print_detail = open("{}/test_detail.log".format(log_dir), 'a+')
        print(datetime.datetime.now(), file=file2print_detail)
        print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_max\tEpoch_max",
              file=file2print_detail)
        file2print_detail.flush()

        ACCs = {}

        MAX_EPOCHs_seed = {}
        ACCs_seed = {}
        for seed in Seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            opt.ckpt_dir = '{}/ckpt/{}/{}/{}/{}/{}/{}'.format(log_dir,
                exp, opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),
                model_paras, str(seed))

            if not os.path.exists(opt.ckpt_dir):
                os.makedirs(opt.ckpt_dir)

            print('[INFO] Running at:', opt.dataset_name)

            x_train, y_train, x_val, y_val, x_test, y_test, opt.nb_class, _ \
                = load_ucr2018(opt.ucr_path, opt.dataset_name)


            ACCs_run={}
            MAX_EPOCHs_run = {}
            for run in Runs:

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
                elif 'SemiInter' in opt.model_name:
                    acc_test, acc_unlabel, epoch_max = train_SemiInter(
                        x_train, y_train, x_val, y_val, x_test, y_test,opt)
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

                print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                    opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
                    seed, round(acc_test, 2), round(acc_unlabel, 2), epoch_max),
                    file=file2print_detail_train)
                file2print_detail_train.flush()


                ACCs_run[run] = acc_test
                MAX_EPOCHs_run[run] = epoch_max

            ACCs_seed[seed] = round(np.mean(list(ACCs_run.values())), 2)
            MAX_EPOCHs_seed[seed] = np.max(list(MAX_EPOCHs_run.values()))

            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
                seed, ACCs_seed[seed], MAX_EPOCHs_seed[seed]),
                file=file2print_detail)
            file2print_detail.flush()

        ACCs_seed_mean = round(np.mean(list(ACCs_seed.values())), 2)
        ACCs_seed_std = round(np.std(list(ACCs_seed.values())), 2)
        MAX_EPOCHs_seed_max = np.max(list(MAX_EPOCHs_seed.values()))

        print("{}\t{}\t{}\t{}".format(
            opt.dataset_name, ACCs_seed_mean, ACCs_seed_std, MAX_EPOCHs_seed_max),
            file=file2print)
        file2print.flush()






