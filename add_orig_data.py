import random
import numpy as np
import pandas as pd
import os 
import argparse

SOURCE='data/cf/cf_mar23/rep_0/cf_train_200'
TARGET='data/cf/cf_mar23/rep_0/orig_train_{}'

orig_train_path = '/home/xiongyi/dataxyz/data/imdb/all_reviews_train'

def add_orig_cf(source_path,orig_train_path, target_dir,total_num= 1000):
    label_2_id={'Negative':0,'Positive':1}
    out_dir = target_dir.format(total_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #obtain the orig from source
    with open(os.path.join(source_path, 'train_paired.tsv'),'r') as f:
        lines = f.readlines()
    
    orig_lines = []
    for idx, line in enumerate(lines):
        if idx % 2 == 1:
            line = line.strip().split('\t')
            label = label_2_id[line[0]]
            sent = line[1]
            new_line=sent+'\t'+str(label)+'\n'
            orig_lines.append(new_line)
    orig_num = len(orig_lines)
    #add to total num
    with open(orig_train_path,'r') as f:
        total_orig_lines = f.readlines()
    
    added_lines = np.random.choice(total_orig_lines,total_num-orig_num,replace=False)
    added_lines=[l.split('\t')[0] +'\t' + l.split('\t')[2] for l in added_lines]
    total_lines = orig_lines+list(added_lines)
    with open(os.path.join(out_dir, 'orig_train.txt'),'w') as f:
        f.writelines(total_lines)



def add_orig_matres(source_path, new_dir,total_num= 100):
    out_dir = new_dir.format(total_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    old_path = os.path.join(source_path, 'matres_train.csv')
    new_path = os.path.join(out_dir, 'matres_train.csv')
    with open(old_path, 'r') as f:
        lines = f.readlines()
    orig_num = len(lines) - 1
    if total_num > orig_num:
        print ('data num is {} bigger than orig num {}'.format(total_num,orig_num))
        total_num = orig_num
  
    subset_idx = np.random.choice(orig_num, total_num, replace=False)

    with open(new_path, 'w') as f:
        f.write(lines[0])
        for idx in subset_idx:
            f.write(lines[idx + 1])

def add_orig_mctaco(source_path, new_dir,total_num= 100):
    out_dir = new_dir.format(total_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    old_path = os.path.join(source_path, 'paired_mctaco_train.txt')
    new_path = os.path.join(out_dir, 'paired_mctaco_train.txt')
    with open(old_path, 'r') as f:
        lines = f.readlines()
    orig_num = len(lines)
    if total_num > orig_num:
        print ('data num is {} bigger than orig num {}'.format(total_num,orig_num))
        total_num = orig_num
  
    subset_idx = np.random.choice(orig_num, total_num, replace=False)

    with open(new_path, 'w') as f:
        for idx in subset_idx:
            f.write(lines[idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, help='source_path')
    parser.add_argument('--orig_train_path', type=str, help='orig_train_path')
    parser.add_argument('--target_path', type=str, help='target_path')
    parser.add_argument('--total_num', type=int, help='total num')
    parser.add_argument('--data_type', type=str, default = 'cf', help='data type')
    args = parser.parse_args()
    if args.data_type == 'cf':
        new_dir = os.path.join(args.target_path,'orig_train_{}')
    elif args.data_type == 'matres':
        new_dir = os.path.join(args.target_path, 'num_{}')
    elif args.data_type == 'mctaco':
        new_dir = os.path.join(args.target_path, 'num_{}')

    # main()
    add_orig_cf(args.source_path,args.orig_train_path, new_dir, args.total_num)
if __name__ == '__main__':
    main()


    