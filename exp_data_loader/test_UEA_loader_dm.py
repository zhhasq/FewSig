import os

from exp_data_loader.UEA_loader import UEAUtils
from tqdm import tqdm

if __name__ == "__main__":
    select_name = []
    # select_warp = []
    with open(os.path.join(os.getcwd(), "exp_data", "UEA_select2.txt"), 'r') as f:
        lines = f.readlines()
    for cur_line in lines:
        if "%" in cur_line:
            continue
        # tmp = cur_line.split(',')
        # select_name.append(tmp[0])
        # select_warp.append(int(tmp[1]))
        cur_name = cur_line.replace("\n", "")
        select_name.append(cur_name)

    select_data = []
    query = [98, 96, 94, 92, 90, 88, 86, 84, 82, 80]
    warp = [1, 3, 5, 7, 9, 10, 12, 14,16, 18,20,25,30]
    dist_code_list = []
    for q in query:
        for w in warp:
            dist_code_list.append(f"D-Q{q}-S100-W{w}")
    for w in warp:
        dist_code_list.append(f"D-Q100-S100-W{w}")
    dist_code_list.append(f"E-Q100-S100")
    for name in tqdm(select_name):
        print(name)
        cur_data = UEAUtils.get_UEA_data(name)
        if cur_data is None:
            continue
        select_data.append(cur_data)
        for cur_code in dist_code_list:
            cur_data.get_dm(cur_code)
    # for x in select_data:
    #     print(x)

    # dist_code = 'E-Q100-S100'
    # for cur_data in select_data:
    #     cur_data.get_dm(dist_code)