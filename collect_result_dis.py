import pandas as pd

res_dir = '/home/zhang.xio/xyz/repos/uda_combined/results_mar8/cf/cfnum_{}_p_{}_r_{}_u_1_hid_{}_layer_{}_rep_{}/acc.txt'

res_df = pd.DataFrame(columns = ['train_num','p','r','rep','hid','layer','acc_orig','acc_cf'])

idx = 0
for p in [1]:
    for r in [1]:
        for rep in range(10):
            for num in [20,100,200,]:
                for hid in [100,300]:
                    for layer in [1,3,5]:
                        res_file = res_dir.format(num,p,r,hid,layer,rep)
                        try :
                            with open(res_file, 'r') as f:
                                lines = f.readlines()
                        except:
                            continue
                        acc_orig = lines[-1].strip().split(',')[1]
                        acc_cf = lines[-1].strip().split(',')[2]
                        res_df.loc[idx] = [num,p,r,rep,hid,layer, float(acc_orig),float(acc_cf)] 
                        idx += 1
res_df.to_csv('mar_8_cf.csv')
    