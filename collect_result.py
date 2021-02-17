import pandas as pd

res_dir = '/home/zhang.xio/xyz/repos/uda_combined/results_feb14/cfnum_{}_p_{}_r_{}_rep_{}/acc.txt'

res_df = pd.DataFrame(columns = ['train_num','p','r','rep','acc_orig','acc_cf'])

idx = 0
for train_num in [200, 500]:
    for p in [0,1,2,5]:
        for r in [0,1,2,5]:
            for rep in [0,1,2,5]:
                res_file = res_dir.format(train_num,p,r,rep)
                with open(res_file, 'r') as f:
                    lines = f.readlines()
                acc_orig = lines[-1].split(',')[1]
                acc_cf = lines[-1].split(',')[2]
                res_df.loc[idx] = [train_num,p,r,rep,acc_orig,acc_cf] 
                idx += 1
res_df.to_csv('feb_14.csv')
breakpoint()
