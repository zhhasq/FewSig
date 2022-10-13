import numpy as np

from models.model_wei.exp_Fdist_knn import ExpFDistKNN
from models.model_wei.assistant_st_loop import AssistantSTLoop

class WeiOL:

    def evaluate(self, train, test, loader, log_path=None, record_time=False):

        model = ExpFDistKNN(1)
        assist_model = ExpFDistKNN(1)
        online_exp = AssistantSTLoop(model, assist_model, train, loader)
        y_hat, nn_train_index, added_test_arids = online_exp.evaluate(test, record_time=record_time)
        if record_time:
            self.init_time = online_exp.init_time
            self.ave_round_t = online_exp.ave_round_t
            return None, None

        y_hat = np.array([x[0] for x in y_hat])


        added_test_ids = np.array([x[0] for x in added_test_arids])
        added_test_ids_set = set(added_test_ids)
        pseudo_labels = []
        for cur_id in test.get_ids():
            if cur_id in added_test_ids_set:
                pseudo_labels.append(1)
            else:
                pseudo_labels.append(0)


        # return y_hat, added_test_ids
        return y_hat, np.array(pseudo_labels)
