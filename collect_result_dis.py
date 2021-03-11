import pandas as pd

res_dir = '/home/zhang.xio/xyz/repos/uda_combined/results_mar9_2/cf/cfnum_{}_p_{}_r_{}_u_{}_hid_{}_layer_{}_rep_{}/acc.txt'

res_df = pd.DataFrame(columns = ['train_num','p','r','u','rep','hid','layer','acc_orig','acc_cf'])

idx = 0
for p in [0]:
    for r in [0]:
        for u in [0,1]:
            for rep in range(10):
                for num in [2,5,10,20,50,100,]:
                    for hid in [300]:
                        for layer in [1]:
                            res_file = res_dir.format(num,p,r,u,hid,layer,rep)
                            try :
                                with open(res_file, 'r') as f:
                                    lines = f.readlines()
                                acc_orig = lines[-1].strip().split(',')[1]
                                acc_cf = lines[-1].strip().split(',')[2]
                                res_df.loc[idx] = [num,p,r,u,rep,hid,layer, float(acc_orig),float(acc_cf)] 
                                idx += 1
                            except:
                                continue
                            
res_df.to_csv('mar_9_base_cf.csv')
    