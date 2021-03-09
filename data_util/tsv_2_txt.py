orig_file = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/paired/train_paired.tsv'
new_file = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/paired/train_paired.txt'

with open(orig_file, 'r') as f:
    lines = f.readlines()

with open(new_file,'w') as f:
    for l in lines:
        f.write(l.split('\t')[1] +'\n')