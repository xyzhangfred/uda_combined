export CUDA_VISIBLE_DEVICES=0
eval_data_dir=data/paired
DATA_TYPE=imdb_cf
NUM=2
ORIG_NUM=100
for rep in "0" "1" 
do
    SUP_DIR=data/cf_train_${NUM}
    ORIG_SUP_DIR=data/cf_orig/num_${ORIG_NUM}
    for u in "0" 
    do
        p=0
        r=0
        python main.py --p $p --r $r --u $u --results_dir ./debug/cf_num_${NUM} --sup_data_dir $SUP_DIR \
         --eval_data_dir $eval_data_dir --data_type $DATA_TYPE --orig_sup_data_dir $ORIG_SUP_DIR --unbalanced
    done
done
# for rep in "0" "1" 
# do
# for p in "1" "5"
# 	do
#         for r in "1"  "5"
#         do
#             for theta in "-1" "2"
#             do
#                 python main.py --p $p --r $r --u 1 --theta $theta --results_dir debug --sup_data_dir $SUP_DIR --eval_data_dir $eval_data_dir --data_type $DATA_TYPE
#             done
#         done
#     done
# done
