data_path='datasets/PeMS/V_228.csv' #path to the MTS data
cycle=$((12*24)) #12 samples an hour, 24 hour a day
data_root='datasets/PeMS' #Directory to the MTS data
#preparing dataset stamp
python3 ./data_provider/data_process.py gen_stamp --data_path=$data_path --cycle=$cycle --data_root=$data_root