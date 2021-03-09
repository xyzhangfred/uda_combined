import random
import numpy as np
import pandas as pd
import os 

ORIG_PATH = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/paired/train_paired.tsv'
NEW_DIR = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/cf_train_{}/'
def make_subset(file_path, new_dir,data_num= 100):

    out_dir = new_dir.format(data_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    new_path = os.path.join(out_dir, 'train_paired.tsv')
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    orig_num = int((len(lines) - 1) / 2)
    subset_idx = np.random.choice(orig_num, data_num, replace=False)
    with open(new_path, 'w') as f:
        f.write(lines[0])
        for idx in subset_idx:
            if idx > 0:
                f.write(lines[idx * 2 -1 ])
                f.write(lines[idx * 2])

for num in [2,5,10,50]:
    make_subset(data_num=num, file_path = ORIG_PATH, new_dir = NEW_DIR)