from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
    
# from data_arrti import Dataset_Opennem
from data_provider.data_Opennem import Dataset_Opennem #
# from data_provider.data_Opennem_ETTm1 import Dataset_Opennem_ETTm1 #
import os
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from data_provider.forecast_dataloader import *


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'Opennem': Dataset_Opennem,
    #'Opennem_ETTm1': Dataset_Opennem_ETTm1
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    print(f"Data provider called with flag={flag}, data={args.data}")

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name in ['anomaly_detection', 'classification']:
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        print(f"Classification: flag={flag}, len(data_set)={len(data_set)}")
    elif args.task_name == 'forecasting':
        df = args.data_update
        
        data_set = ForecastDataset(
            df,
            window_size=args.window_size,
            horizon=args.horizon,
            normalize_method=args.norm_method,
            norm_statistic=args.norm_statistic,
        )
        
        print(flag, len(data_set))
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        print(f"Other tasks: flag={flag}, data_set length before DataLoader: {len(data_set)}")

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    print(f"DataLoader created: batch_size={batch_size}, shuffle_flag={shuffle_flag}, drop_last={drop_last}")
    return data_set, data_loader