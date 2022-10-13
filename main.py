import torch

from exp_model_comparsion.online_exp import OnlineExp, NotEnoughPS

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    data_path_args = {
        "source": "/home/zhongs/btc_mount/Univariate_ts",
        "UAE_data": "/home/zhongs/btc_mount/UAE_TS/data_source",
        "dist_core": 20,
        "dist_batch_size": 2000
    }

    for name in ["ACSF1"]:

        FNCAE_args = {
            "target_FPR": 0.02,
            "LR": 0.02,
            "init_gpu_count": 0
        }

        p_init = 5
        max_same = 3
        batch = 30
        try:
            fncae = OnlineExp(name, data_path_args, "FewSig", p_init, batch, max_same, FNCA_arg=FNCAE_args, gpu_num=4,
                              select_check=True)
            if fncae is not None:
                results = fncae.start()
                f1, ave_f1 = fncae.get_ave_final_F1()
        except NotEnoughPS:
            print("Not enough positive samples")

