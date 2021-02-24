import random

with open('orig_train.txt') as f:
    orig_train = f.readlines()

with open('cf_train.txt') as f:
    cf_train = f.readlines()

orig_cf_train = orig_train[1:] + cf_train[1:]
random.shuffle(orig_cf_train)
orig_cf_train.insert(0,orig_train[0])

with open('orig_cf_train.txt', 'w') as f:
    f.writelines(orig_cf_train)


with open('orig_dev.txt') as f:
    orig_dev = f.readlines()

with open('cf_dev.txt') as f:
    cf_dev = f.readlines()

orig_cf_dev = orig_dev[1:] + cf_dev[1:]
random.shuffle(orig_cf_dev)
orig_cf_dev.insert(0,orig_dev[0])

with open('orig_cf_dev.txt', 'w') as f:
    f.writelines(orig_cf_dev)