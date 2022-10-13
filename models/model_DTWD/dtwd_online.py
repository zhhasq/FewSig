import numpy as np

from exp_data_loader.train_test_dataset import TrainSet, TestSet
from models.model_wei.wei_online import WeiOL


def transform(dtw_set, eu_set, is_train):
    eu_dm = eu_set.get_features()
    dtw_dm = dtw_set.get_features()
    dtwd_dm = np.zeros(shape=dtw_dm.shape)
    for i in range(dtw_dm.shape[0]):
        for j in range(dtwd_dm.shape[1]):
            if eu_dm[i, j] == 0:
                dtwd_dm[i, j] = 0
            else:
                dtwd_dm[i, j] = dtw_dm[i, j] / eu_dm[i, j]

    if is_train:
        new_set = TrainSet(dtwd_dm, dtw_set.get_labels(), dtw_set.get_ids(),
                           dtw_set.snr, dist_info=dtw_set.dist_info,
                           train_ts_data=dtw_set.get_ts_data())
    else:
        new_set = TestSet(dtwd_dm, dtw_set.get_labels(), dtw_set.get_ids(),
                          dtw_set.snr, dist_info=dtw_set.dist_info,
                          test_ts_data=dtw_set.get_ts_data())
    return new_set


class DTWDOL:

    def evaluate(self, train_dtw, test_dtw, train_eu, test_eu, loader_dtw, loader_eu, record_time=False):

        train = transform(train_dtw, train_eu, True)
        test = transform(test_dtw, test_eu, False)

        dtwd_dm = np.zeros_like(loader_dtw.dm)
        if loader_dtw.dist_code.split('-')[0] != 'D' or loader_eu.dist_code.split('-')[0] != 'E':
            raise  RuntimeError

        dtw_dm = loader_dtw.dm
        eu_dm = loader_eu.dm
        for i in range(dtwd_dm.shape[0]):
            for j in range(dtwd_dm.shape[1]):
                if eu_dm[i, j] == 0:
                    dtwd_dm[i, j] = 0
                else:
                    dtwd_dm[i, j] = dtw_dm[i, j] / eu_dm[i, j]
        loader_dtw.dm = dtwd_dm
        loader = loader_dtw

        wei = WeiOL()

        if record_time:
            wei.evaluate(train, test, loader, record_time=record_time)
            self.init_time = wei.init_time
            self.ave_round_t = wei.ave_round_t
            return None, None
        else:
            return wei.evaluate(train, test, loader)
