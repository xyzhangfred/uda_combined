export CUDA_VISIBLE_DEVICES=0
SUP_DIR=data/cf_train_100
eval_data_dir=data/paired
DATA_TYPE=imdb_cf
# for rep in "0" "1" 
# do
#     for u in "0" 
#     do
#         p=0
#         r=0
#         python main.py --p $p --r $r --u $u --results_dir ./results_feb24/cf_200_${p}_${r}_u_${u}rep_${rep} --sup_data_dir $SUP_DIR --eval_data_dir $eval_data_dir --data_type $DATA_TYPE
#     done
# done
for rep in "0" "1" 
do
for p in "1" "5"
	do
        for r in "1"  "5"
        do
            for theta in "-1" "2"
            do
                python main.py --p $p --r $r --u 1 --theta $theta --results_dir debug --sup_data_dir $SUP_DIR --eval_data_dir $eval_data_dir --data_type $DATA_TYPE
            done
        done
    done
done
