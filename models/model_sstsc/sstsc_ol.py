import argparse

import torch

from exp_data_loader.train_test_dataset import TestSet
from models.model_sstsc.train_ssl_ol_AS import SSTSCOL
from tqdm import tqdm
from multiprocessing import Pool


class EXPSSTC:
    def __init__(self, train_set, test_set):
        self.opt = EXPSSTC.parse_option()
        setattr(self.opt, 'dataset_name', "Nepal_MKAR")
        setattr(self.opt, 'model_name', "SemiInterOL")
        setattr(self.opt, 'label_ratio', -1)
        setattr(self.opt, 'config_dir',
                "/home/zhongs/corsair_mp600_2/project-repo/aftershock2/exp_model_comparsion/sstsc/config")
        setattr(self.opt, 'ucr_path',
                "/home/zhongs/corsair_mp600_2/project-repo/aftershock2/exp_model_comparsion/sstsc/datasets")
        setattr(self.opt, 'ckpt_dir',
                "/home/zhongs/corsair_mp600_2/project-repo/aftershock2/exp_model_comparsion/sstsc/ckpt")

        self.train_set = train_set
        self.test_set = test_set

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

    def evaluate_mp(self, num_process=4):
        y_hat = [1]
        pool = Pool(processes=num_process)
        result_async = []
        test_ts_data = self.test_set.test_ts_data
        test_labels = self.test_set.get_labels()

        for i in range(2, len(self.test_set)):
            gpu_num = (i-2) % 4
            cur_test_set = TestSet(None, test_labels[:i], None, -1, dist_info=None, test_ts_data=test_ts_data[:i, :])
            result_async.append(pool.apply_async(EXPSSTC.predict_single_mp_helper, args=(
                self.opt, gpu_num, self.train_set, cur_test_set)))

        for j in tqdm(range(len(result_async))):
            r = result_async[j].get()
            print(j, r, test_labels[1+j])
            y_hat.append(r)
        pool.close()
        print("waiting joined")
        pool.join()
        print("joined")
        return y_hat

    @staticmethod
    def predict_single_mp_helper(opt, gpu_num, train_set, test_set):
        log_path = "/home/zhongs/corsair_mp600_2/project-repo/aftershock2/exp_model_comparsion/sstsc/log"
        setattr(opt, 'dataset_name', "Nepal_MKAR")
        setattr(opt, 'gpu', f"{gpu_num}")
        cuda_d = torch.device(f'cuda:{gpu_num}')

        cur_y_hat = SSTSCOL.predict(opt, log_path, train_set, test_set)
        return cur_y_hat
