import numpy as np
path = '/home/xiongyi/dataxyz/data/imdb/all_reviews_train'
new_path = 'data/cf_orig/num_{}'
with open(path,'r') as f:
    lines = f.readlines()
orig_num = 100
total_nums = [1000,500,200]
curr_lines = lines
all_sample_lines={}
for n in total_nums:
    num = n - orig_num
    sample_lines = np.random.choice(curr_lines, num, replace=False)
    curr_lines = sample_lines
    all_sample_lines[num]= sample_lines
    
    with open(new_path.format(num),'w') as f:
        f.writelines(sample_lines)
