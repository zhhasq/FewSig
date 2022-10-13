
# dist_code = 'D-Q100-S100-W100'
from models.model_success.success_st import SuccessST
from models.model_fewsig import ExpDataNepalMKAR
from utils import scores

exp_data = ExpDataNepalMKAR()
data_code = 'U-AS5-T-SNR2'
dist_code = 'BEST1NN'
train_3ch, test_3ch = exp_data.get_train_test(data_code, dist_code, 20)
loader = exp_data.get_dm(data_code, dist_code, 20)
success = SuccessST(train_3ch, test_3ch, loader)
y_hat = success.fit()
r = scores.score_pr(y_hat, test_3ch.get_labels())

E_index = success.model.E_index
E_index_set = set(E_index)
for x in success.model.clusters:
    count = 0
    for y in x:
        if y in E_index_set:
            count += 1
    if count != 1:
        print("Error")