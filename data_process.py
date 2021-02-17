import random
import numpy as np
import pandas as pd
file_path = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/paired/train_paired_full.tsv'

data_num = 20
new_path = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/paired/train_paired_num_{}.tsv'.format(data_num)
with open(file_path, 'r') as f:
    lines = f.readlines()

orig_num = int((len(lines) - 1) / 2)

# for i in range(1,orig_num):
#     paired = (lines[2*i-1], lines[2*i])
#     # print (paired[0][:7])
#     # print (paired[1][:7])
#     # if paired[0][:7]==paired[1][:7]:
#         # breakpoint()

# breakpoint
subset_idx = np.random.choice(orig_num, data_num, replace=False)
with open(new_path, 'w') as f:
    f.write(lines[0])
    for idx in subset_idx:
        if idx > 0:
            f.write(lines[idx * 2 -1 ])
            f.write(lines[idx * 2])

