import os

import numpy as np

from exp_model_comparsion.online_exp import OnlineExp

select_name = []
# select_warp = []
with open(os.path.join(os.getcwd(), "exp_data", "../resources/UEA_select.txt"), 'r') as f:
    lines = f.readlines()
for cur_line in lines:
    if "%" in cur_line:
        continue
    # tmp = cur_line.split(',')
    # select_name.append(tmp[0])
    # select_warp.append(int(tmp[1]))
    cur_name = cur_line.replace("\n", "")
    select_name.append(cur_name)

trail = 30
all_score_results = []
all_vote_results = []
for name in select_name:
    cur_score_result = []
    cur_vote_results = []
    # cur_score_result.append(name)
    FewSig_args = {
        "target_FPR": 0.01,
        "LR": 0.02,
        "init_gpu_count": 0
    }
    data_path_args = {
        "source": "/home/user/FewSig/Univariate_ts",
        "UEA_data": "/home/user/FewSig/UEA_select",
        "dist_core": 20,
        "dist_batch_size": 2000
    }
    SSTSC_args = {
        "process_num": 1,
        "batch_size": 256,
        "include_val_score": True

    }
    wei = OnlineExp(name, data_path_args, "Wei", 5, trail, 3)
    results = wei.start()
    wei_f1, wei_ave_f1 = wei.get_ave_final_F1()

    fewsig = OnlineExp(name, data_path_args, "FewSig", 5, trail, 3, FewSig_args=FewSig_args, gpu_num=4, select_check=True)
    results = fewsig.start()
    fewsig_f1, fewsig_ave_f1 = fewsig.get_ave_final_F1()

    fewsig_f1_v1, fewsig_ave_f1_v1 = fewsig.get_ave_final_F1(1)
    fewsig_f1_v2, fewsig_ave_f1_v2 = fewsig.get_ave_final_F1(2)
    fewsig_f1_v3, fewsig_ave_f1_v3 = fewsig.get_ave_final_F1(3)
    fewsig_f1_v4, fewsig_ave_f1_v4 = fewsig.get_ave_final_F1(4)
    fewsig_f1_v5, fewsig_ave_f1_v5 = fewsig.get_ave_final_F1(5)

    dtwd = OnlineExp(name, data_path_args, "DTWD", 5, trail, 3)
    results = dtwd.start()
    dtwd_f1, dtwd_ave_f1 = dtwd.get_ave_final_F1()

    success = OnlineExp(name, data_path_args, "SUCCESS", 5, trail, 3)
    results = success.start()
    success_f1, success_ave_f1 = success.get_ave_final_F1()

    sstsc = OnlineExp(name, data_path_args, "SSTSC", 5, trail, 3, sstsc_args=SSTSC_args, gpu_num=1, select_check=False)
    if name == "SmoothSubspace":
        sstsc_f1 = 0
        sstsc_ave_f1 = 0
    else:
        results = sstsc.start()
        sstsc_f1, sstsc_ave_f1 = sstsc.get_ave_final_F1()
    print(f"{name}, FewSig: {fewsig_ave_f1} wei: {wei_ave_f1} DTWD: {dtwd_ave_f1} SUCCESS: {success_ave_f1}, SSTSC: {sstsc_ave_f1}")
    cur_score_result.append(wei_ave_f1)
    cur_score_result.append(dtwd_ave_f1)
    cur_score_result.append(success_ave_f1)
    cur_score_result.append(sstsc_ave_f1)
    cur_score_result.append(fewsig_ave_f1)
    cur_score_result.append(fewsig_ave_f1_v1)
    cur_score_result.append(fewsig_ave_f1_v2)
    cur_score_result.append(fewsig_ave_f1_v3)
    cur_score_result.append(fewsig_ave_f1_v4)
    cur_score_result.append(fewsig_ave_f1_v5)

    all_score_results.append(cur_score_result)


for i in range(len(select_name)):
    name = select_name[i]
    cur_score = all_score_results[i]
    print(
        f"{name}, fewsig: {cur_score[0]} wei: {cur_score[1]} DTWD: {cur_score[2]} SUCCESS: {cur_score[3]}, SSTSC: {cur_score[4]}" + os.linesep)
all_score_results = np.array(all_score_results)
labels = ["FewSig", "Wei's_OL", "DTWD_OL", "SUCCESS_OL", "SSTSC_OL"]
# print(f"Ave. {all_score_results[:,0].mean()},{all_score_results[:,1].mean()},{all_score_results[:,2].mean()},"
#       f"{all_score_results[:,3].mean()},{all_score_results[:,4].mean()} ")
# labels = ["fewsig", "Wei", "DTWD"]
# fit = plt.figure()
# plot_critical_difference(all_score_results, labels, cliques=None, is_errors=False, alpha=0.05, width=10, textspace=2.5, reverse=True,)


for i in range(len(select_name)):
    name = select_name[i]
    cur_score = all_score_results[i]
    print(f"{name} & {cur_score[0]:.4f} & {cur_score[1]:.4f} &{cur_score[2]:.4f} &{cur_score[3]:.4f} "
          f"&{cur_score[4]:.4f} &{cur_score[5]:.4f} &{cur_score[6]:.4f} &{cur_score[7]:.4f} &{cur_score[8]:.4f} "
          f"& {cur_score[9]:.4f} \\\\")
    print("\\hline")