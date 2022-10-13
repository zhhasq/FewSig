import os

import numpy as np

from exp_model_comparsion.online_exp import OnlineExp

select_name = []
# select_warp = []
with open(os.path.join(os.getcwd(), "exp_data", "../resources/paper_all.txt"), 'r') as f:
    lines = f.readlines()
for cur_line in lines:
    if "%" in cur_line:
        continue
    # tmp = cur_line.split(',')
    # select_name.append(tmp[0])
    # select_warp.append(int(tmp[1]))
    cur_name = cur_line.replace("\n", "")
    select_name.append(cur_name)

batch = 30
all_score_results = []
all_vote_results = []
for name in select_name:
    cur_score_result = []
    cur_vote_results = []
    # cur_score_result.append(name)
    FNCAE_args = {
        "target_FPR": 0.01,
        "LR": 0.02
    }
    data_path_args = {
        "source": "/home/zhongs/btc_mount/Univariate_ts",
        # "UAE_data": "/home/zhongs/btc_mount/UAE_TS/data_source"
        "UAE_data": "/home/zhongs/sdm_results_backup/data_source"
    }
    SSTSC_args = {
        "process_num": 1,
        "batch_size": 256,
        "include_val_score": True

    }
    wei = OnlineExp(name, data_path_args, "Wei", 5, batch, 3, FNCAE_args, select_check=False)
    results = wei.start()
    wei_f1, wei_ave_f1 = wei.get_ave_final_F1()

    fncae = OnlineExp(name, data_path_args, "FNCAE", 5, batch, 3, FNCAE_args, select_check=False)
    results = fncae.start()
    fncae_f1, fncae_ave_f1 = fncae.get_ave_final_F1()

    fncae_f1_v1, fncae_ave_f1_v1 = fncae.get_ave_final_F1(1)
    fncae_f1_v2, fncae_ave_f1_v2 = fncae.get_ave_final_F1(2)
    fncae_f1_v3, fncae_ave_f1_v3 = fncae.get_ave_final_F1(3)
    fncae_f1_v4, fncae_ave_f1_v4 = fncae.get_ave_final_F1(4)
    fncae_f1_v5, fncae_ave_f1_v5 = fncae.get_ave_final_F1(5)

    dtwd = OnlineExp(name, data_path_args, "DTWD", 5, batch, 3, FNCAE_args, select_check=False)
    results = dtwd.start()
    dtwd_f1, dtwd_ave_f1 = dtwd.get_ave_final_F1()

    success = OnlineExp(name, data_path_args, "SUCCESS", 5, batch, 3, select_check=False)
    results = success.start()
    success_f1, success_ave_f1 = success.get_ave_final_F1()

    sstsc = OnlineExp(name, data_path_args, "SSTSC", 5, batch, 3, sstsc_args=SSTSC_args, gpu_num=1, select_check=False)
    if name == "SmoothSubspace":
        sstsc_f1 = 0
        sstsc_ave_f1 = 0
    else:
        results = sstsc.start()
        sstsc_f1, sstsc_ave_f1 = sstsc.get_ave_final_F1()
    print(f"{name}, FNCAE: {fncae_ave_f1} wei: {wei_ave_f1} DTWD: {dtwd_ave_f1} SUCCESS: {success_ave_f1}, SSTSC: {sstsc_ave_f1}")
    cur_score_result.append(wei_ave_f1)
    cur_score_result.append(dtwd_ave_f1)
    cur_score_result.append(success_ave_f1)
    cur_score_result.append(sstsc_ave_f1)
    cur_score_result.append(fncae_ave_f1)
    cur_score_result.append(fncae_ave_f1_v1)
    cur_score_result.append(fncae_ave_f1_v2)
    cur_score_result.append(fncae_ave_f1_v3)
    cur_score_result.append(fncae_ave_f1_v4)
    cur_score_result.append(fncae_ave_f1_v5)

    all_score_results.append(cur_score_result)


for i in range(len(select_name)):
    name = select_name[i]
    cur_score = all_score_results[i]
    print(
        f"{name}, FNCAE: {cur_score[0]} wei: {cur_score[1]} DTWD: {cur_score[2]} SUCCESS: {cur_score[3]}, SSTSC: {cur_score[4]}" + os.linesep)
all_score_results = np.array(all_score_results)
labels = ["FewSig", "Wei's_OL", "DTWD_OL", "SUCCESS_OL", "SSTSC_OL"]
# print(f"Ave. {all_score_results[:,0].mean()},{all_score_results[:,1].mean()},{all_score_results[:,2].mean()},"
#       f"{all_score_results[:,3].mean()},{all_score_results[:,4].mean()} ")
# labels = ["FNCAE", "Wei", "DTWD"]
# fit = plt.figure()
# plot_critical_difference(all_score_results, labels, cliques=None, is_errors=False, alpha=0.05, width=10, textspace=2.5, reverse=True,)


for i in range(len(select_name)):
    name = select_name[i]
    cur_score = all_score_results[i]
    print(f"{name} & {cur_score[0]:.4f} & {cur_score[1]:.4f} &{cur_score[2]:.4f} &{cur_score[3]:.4f} "
          f"&{cur_score[4]:.4f} &{cur_score[5]:.4f} &{cur_score[6]:.4f} &{cur_score[7]:.4f} &{cur_score[8]:.4f} "
          f"& {cur_score[9]:.4f} \\\\")
    print("\\hline")