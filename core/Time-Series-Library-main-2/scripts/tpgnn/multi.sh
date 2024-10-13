#!/usr/bin/env bash
data_path='datasets/PeMS/V_228.csv' #path to the MTS data
cycle=$((12*24)) #12 samples an hour, 24 hour a day
data_root='datasets/PeMS' #Directory to the MTS data
#preparing dataset stamp
python3 ./data_provider/data_process.py gen_stamp --data_path=$data_path --cycle=$cycle --data_root=$data_root

data_path='datasets/PeMS/V_228.csv' #path to the MTS data
adj_path='datasets/PeMS/W_228.csv'  #path to the adjacency matrix, None if not exists
data_root='datasets/PeMS' #Directory to the MTS data

stamp_path="${data_root}/time_stamp.npy"
#training model
python main_stamp.py train --device=3 --n_route=228 --n_his=12 --n_pred=12 --n_train=34 --n_val=5 --n_test=5 --mode=1 --name='PeMS'\
    --data_path="datasets/PeMS/V_228.csv" --adj_matrix_path="datasets/PeMS/W_228.csv" --stamp_path=$stamp_path
    