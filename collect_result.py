import pandas as pd

res_dir = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/results_mar9/mctaco_{}_{}_u_{}_ln_{}_rep_{}/acc.txt'
res_df = pd.DataFrame(columns = ['p','r','u','rep','acc_orig','acc_cf'])

idx = 0
for rep in range(6):
    for p in [0,2,5]:
        for r in [0,0.1,0.2,0.5]:
            for u in [0,1]:
                for layer in [1]:
                    res_non = res_dir.format(p, r,u,layer, rep)
                    try :
                        with open(res_non, 'r') as f:
                            lines = f.readlines()
                    except:
                        continue
                    acc_orig = lines[-1].split(',')[1].strip()
                    acc_cf = lines[-1].split(',')[2].strip()
                    res_df.loc[idx] = [p,r,1,rep,acc_orig,acc_cf] 
                    idx += 1

res_bl = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/results_mar9/mctaco_p_0_r_0_u_{}_rep_{}/acc.txt'          
for rep in range(6):
    for u in [0,1]:
        res = res_bl.format(u,rep)
        try :
            with open(res, 'r') as f:
                lines = f.readlines()
        except:
            continue
        acc_orig = lines[-1].split(',')[1].strip()
        acc_cf = lines[-1].split(',')[2].strip()
        res_df.loc[idx] = [0,0,u,rep,acc_orig,acc_cf] 
        idx += 1

res_df.to_csv('mar_9_mctaco.csv')
res_df = res_df.astype({'acc_orig': 'float','acc_cf': 'float'})
mean = res_df.groupby(['p','r','u']).mean()
std = res_df.groupby(['p','r','u']).std()
breakpoint()
