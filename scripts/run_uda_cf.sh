P=1
R=1
SUP_DIR=data/cf_train_200
eval_data_dir=data/paired
DATA_TYPE=imdb_cf

python main.py --p $P --r $R --results_dir ./results/debug_$P_$R --sup_data_dir $SUP_DIR --eval_data_dir $eval_data_dir --data_type $DATA_TYPE
