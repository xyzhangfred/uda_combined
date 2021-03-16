import random
import numpy as np
import pandas as pd
import os 
import argparse

def make_subset_cf(source_path, new_dir,data_num= 100):
    out_dir = new_dir.format(data_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    old_path = os.path.join(source_path, 'train_paired.tsv')
    new_path = os.path.join(out_dir, 'train_paired.tsv')
    
    with open(old_path, 'r') as f:
        lines = f.readlines()

    orig_num = int((len(lines) - 1) / 2)
    if data_num > orig_num:
        print ('data num is {} bigger than orig num {}'.format(data_num,orig_num))
        data_num = orig_num
    subset_idx = np.random.choice(orig_num, data_num, replace=False)
    with open(new_path, 'w') as f:
        f.write(lines[0])
        for idx in subset_idx:
            f.write(lines[idx * 2 + 1 ])
            f.write(lines[idx * 2 + 2])

def make_subset_matres(source_path, new_dir,data_num= 100):
    out_dir = new_dir.format(data_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    old_path = os.path.join(source_path, 'matres_train.csv')
    new_path = os.path.join(out_dir, 'matres_train.csv')
    with open(old_path, 'r') as f:
        lines = f.readlines()
    orig_num = len(lines) - 1
    if data_num > orig_num:
        print ('data num is {} bigger than orig num {}'.format(data_num,orig_num))
        data_num = orig_num
  
    subset_idx = np.random.choice(orig_num, data_num, replace=False)

    with open(new_path, 'w') as f:
        f.write(lines[0])
        for idx in subset_idx:
            f.write(lines[idx + 1])

def make_subset_mctaco(source_path, new_dir,data_num= 100):
    out_dir = new_dir.format(data_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    old_path = os.path.join(source_path, 'paired_mctaco_train.txt')
    new_path = os.path.join(out_dir, 'paired_mctaco_train.txt')
    with open(old_path, 'r') as f:
        lines = f.readlines()
    orig_num = len(lines)
    if data_num > orig_num:
        print ('data num is {} bigger than orig num {}'.format(data_num,orig_num))
        data_num = orig_num
  
    subset_idx = np.random.choice(orig_num, data_num, replace=False)

    with open(new_path, 'w') as f:
        for idx in subset_idx:
            f.write(lines[idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, help='source_path')
    parser.add_argument('--target_path', type=str, help='target_path')
    parser.add_argument('--data_type', type=str, default = 'cf', help='data type')
    parser.add_argument('--baseline',action = 'store_true', help='sample for baseline or not, if yes, sample twice as much')
    args = parser.parse_args()
    if args.data_type == 'cf':
        new_dir = os.path.join(args.target_path, 'cf_train_{}')
    elif args.data_type == 'matres':
        new_dir = os.path.join(args.target_path, 'num_{}')
    elif args.data_type == 'mctaco':
        new_dir = os.path.join(args.target_path, 'num_{}')
  
    if args.baseline:
        new_dir = os.path.join(new_dir,'baseline')

    prev_num = None
    for num in [200,100,50,20,10,5,2]:
        if args.baseline:
            num = 2*num
        if prev_num is None:
            if args.data_type == 'cf':
                make_subset_cf(data_num=num, source_path = args.source_path, new_dir = new_dir)
            elif args.data_type == 'matres':
                make_subset_matres(data_num=num, source_path = args.source_path, new_dir = new_dir)
            elif args.data_type == 'mctaco':
                make_subset_mctaco(data_num=num, source_path = args.source_path, new_dir = new_dir)
    
        else:
            if args.data_type == 'cf':
                make_subset_cf(data_num=num, source_path = new_dir.format(prev_num), new_dir = new_dir)
            elif args.data_type == 'matres':
                make_subset_matres(data_num=num, source_path = new_dir.format(prev_num), new_dir = new_dir)
            elif args.data_type == 'mctaco':
                make_subset_mctaco(data_num=num, source_path = new_dir.format(prev_num), new_dir = new_dir)

        prev_num = num

if __name__ == '__main__':
    main()


    