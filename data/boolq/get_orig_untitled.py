import json

with open('/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/boolq/train.jsonl', 'r') as json_file:
    json_list = list(json_file)

unlabeled = []
for json_str in json_list:
    dat = json.loads(json_str)
    p=dat['passage']
    q = dat['question']
    unlabeled.append(p+'\t' + q + '\n')

with open('/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/boolq/unlabeled_orig.tsv', 'w') as f:
    f.writelines(unlabeled)