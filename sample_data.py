import random
import numpy as np
import pandas as pd
import os 

def make_subset(source_path, new_dir,data_num= 100):

    out_dir = new_dir.format(data_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    old_path = os.path.join(source_path, 'train_paired.tsv')
    
    new_path = os.path.join(out_dir, 'train_paired.tsv')
    
    with open(old_path, 'r') as f:
        lines = f.readlines()

    orig_num = int((len(lines) - 1) / 2)
    subset_idx = np.random.choice(orig_num, data_num, replace=False)
    with open(new_path, 'w') as f:
        f.write(lines[0])
        for idx in subset_idx:
            if idx > 0:
                f.write(lines[idx * 2 -1 ])
                f.write(lines[idx * 2])



def main(cfg, model_cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, help='type of data')
    parser.add_argument('--target_path', type=str, help='type of data')
    args = parser.parse_args()

    new_dir = os.path.join(args.target_path, 'cf_train_{}')
    prev_num = None
    for num in [200,100,50,20,10,5,2]:
        if prev_num is None:
            make_subset(data_num=num, source_path = args.source_path, new_dir = new_dir)
        else:
            make_subset(data_num=num, source_path = args.target_path.format(prev_num), new_dir = new_dir)
        prev_num = num


    