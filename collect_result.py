import pandas as pd

res_dir = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/results_feb28/matres_p_{}_r_{}_u_{}_rep_{}/acc.txt'

res_df = pd.DataFrame(columns = ['p','r','u','rep','acc_orig','acc_cf'])

idx = 0
for p in [0,1,2,5]:
    for r in [0,1,2,5]:
        for u in [0,1]:
            for rep in range(6):
                res_file = res_dir.format(p,r,u,rep)
                try :
                    with open(res_file, 'r') as f:
                        lines = f.readlines()
                except:
                    continue
                acc_orig = lines[-1].split(',')[1].strip()
                acc_cf = lines[-1].split(',')[2].strip()
                res_df.loc[idx] = [p,r,u,rep,acc_orig,acc_cf] 
                idx += 1
res_df.to_csv('mar_1_matres_0.csv')
breakpoint()
