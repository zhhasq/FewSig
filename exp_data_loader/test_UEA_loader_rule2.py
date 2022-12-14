import os

from exp_data_loader.UEA_loader import UEAUtils

print(os.getcwd())
with open(os.path.join(os.getcwd(), "exp_data", "UEA_REF.csv")) as f:
    lines = f.readlines()

save_set = []
for line in lines:
    print()
    print(f"Checking: {line}")
    tmp = line.split(',')
    if (int(tmp[1]) + int(tmp[2]) > 800):
        print("\t Drop reason: Too big dataset")
        continue
    name = tmp[0].replace("\ufeff", '')
    r = UEAUtils.get_UEA_data(name)
    if r is not None:
        save_set.append(name)
# #
with open(os.path.join(os.getcwd(), "exp_data", "UEA_select2.txt"), 'w') as writer:
    for x in save_set:
        writer.write(x + os.linesep)