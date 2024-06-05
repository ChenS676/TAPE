# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import *
import torch
import logging
import os.path as osp 
import numpy as np
import itertools
from tqdm import tqdm
import time
import wandb
import pandas as pd

from torch_geometric import seed_everything
from torch_geometric.data.makedirs import makedirs
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import dump_cfg, makedirs_rm_exist#, dump_run_cfg
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from distutils.util import strtobool
import argparse

from graphgps.train.ns_train import Trainer_NS
# from torch_geometric.loader import NeighborLoader
from graphgps.network.custom_gnn import create_model
from graphgps.network.gsaint import GraphSAINTRandomWalkSampler
from data_utils.load import load_data_lp
from graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger, get_logger
import pprint

FILE_PATH = f'{get_git_repo_root_path()}/'

def get_loader_RW(data, batch_size, walk_length, num_steps, sample_coverage):
    return GraphSAINTRandomWalkSampler(data, batch_size=batch_size, 
                                       walk_length=walk_length, 
                                       num_steps=num_steps, sample_coverage=sample_coverage)

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/vgae.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        default='pubmed',
                        help='data name')
    parser.add_argument('--device', dest='device', required=True, 
                        help='device id')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    # parser.add_argument('--wandb', dest='wandb', required=True, action='store_true',
    #                     help='data name')

    return parser.parse_args()

def save_results_to_file(result_dict, cfg, output_dir):
    """
    Saves the results and the configuration to a CSV file.
    """
    # Create a DataFrame from the result dictionary
    result_df = pd.DataFrame([result_dict])
    
    # Add configuration details as columns
    print(cfg)
    result_df['ModelType'] = cfg.type
    result_df['BatchSize'] = cfg.batch_size
    result_df['LearningRate'] = cfg.lr
    result_df['HiddenChannels'] = cfg.hidden_channels
    result_df['OutChannels'] = cfg.out_channels
    result_df['BatchSizeSampler'] = cfg.batch_size_sampler
    result_df['NumNeighbors'] = cfg.num_neighbors
    result_df['NumHops'] = cfg.num_hops
    
    # Specify the output file path
    output_file = os.path.join(output_dir, 'results_summary.csv')
    
    # Check if file exists to append or write header
    if os.path.exists(output_file):
        result_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(output_file, mode='w', header=True, index=False)
    
    print(f"Results saved to {output_file}")

hyperparameter_space = {
    'GAT': {'out_channels': [2**7, 2**8], 'hidden_channels':  [2**8],
                                'heads': [2**2, 2], 'negative_slope': [0.1], 'dropout': [0], 
                                'num_layers': [5, 6, 7], 'base_lr': [0.015]},
    'GAE': {'out_channels': [32], 'hidden_channels': [32]},
    'VGAE': {'out_channels': [32], 'hidden_channels': [32]},
    'GraphSage': {'out_channels': [2**8, 2**9], 'hidden_channels': [2**8, 2**9]}, 'base_lr': [0.015, 0.1, 0.01]
}

hyperparameter_gsaint = {
        'batch_size': [32, 64, 128],
        'lr': [0.1, 0.01],
        'batch_size_sampler': [32, 64, 128],
        'num_neighbors': [5, 10, 20, 30, 40], #50, 100 
        'num_hops': [6, 7, 8]# 1, 2 is very small number of hops, 9 and 10 take a lot of time and give bad results
}

yaml_file = {   
             'GAT': 'core/yamls/cora/gcns/gat.yaml',
             'GAE': 'core/yamls/arxiv_2023/gcns/gae.yaml',
             'VGAE': 'core/yamls/arxiv_2023/gcns/vgae.yaml',
             'GraphSage': 'core/yamls/cora/gcns/graphsage.yaml'
            }

def project_main():
    
    args = parse_args()

    # args.cfg_file = yaml_file[args.model]
    
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)
    
    cfg.data.name = args.data
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    print('DDEVICE: ', cfg.device, args.device)
    print("cuda" if torch.cuda.is_available() else "cpu")
    # cfg.train.epochs = args.epoch
    
    
    # Set Pytorch environment
    # torch.set_num_threads(20) #!!!!Computations become slower

    loggers = create_logger(args.repeat)

    for model_type in ['GAE']:#, 'GAE', 'VGAE', 'GAT', 'GraphSage']:
        cfg.model.type = model_type
        args.cfg_file = yaml_file[model_type]
        
        # save params
        custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)

        output_dir = os.path.join(FILE_PATH, f"results_{model_type}")
        os.makedirs(output_dir, exist_ok=True)
        print(cfg.data.split_index)
        # for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        # Set configurations for each run TODO clean code here 
        id = wandb.util.generate_id()
        cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}' 
        custom_set_run_dir(cfg, cfg.wandb.name_tag)

        cfg.seed = 0#seed
        cfg.run_id = 0#run_id
        seed_everything(cfg.seed)
        
        cfg = config_device(cfg)
        cfg.data.name = args.data

        splits, _, data = load_data_lp[cfg.data.name](cfg.data)
        cfg.model.in_channels = splits['train'].x.shape[1]

        print_logger = set_printing(cfg)
        print_logger.info(f"The {cfg['data']['name']} graph {splits['train']['x'].shape} is loaded on {splits['train']['x'].device}, \n Train: {2*splits['train']['pos_edge_label'].shape[0]} samples,\n Valid: {2*splits['train']['pos_edge_label'].shape[0]} samples,\n Test: {2*splits['test']['pos_edge_label'].shape[0]} samples")
        dump_cfg(cfg)    

        hyperparameter_search = hyperparameter_space[cfg.model.type]
        combined_hyperparameters = {**hyperparameter_search, **hyperparameter_gsaint}
        
        print_logger.info(f"hypersearch space: {combined_hyperparameters}")
        
        keys = combined_hyperparameters.keys()
        values = combined_hyperparameters.values()
        combinations = itertools.product(*values)
        
        for combination in combinations:

            param_dict = dict(zip(keys, combination))
    
            for key, value in param_dict.items():
                setattr(cfg.model, key, value)
            
            cfg.train.batch_size = cfg.model.batch_size
            cfg.train.lr = cfg.model.lr
            print_logger.info(f"out : {cfg.model.out_channels}, hidden: {cfg.model.hidden_channels}")
            print_logger.info(f"bs : {cfg.train.batch_size}, lr: {cfg.model.lr}")
                        
            start_time = time.time()
                
            model = create_model(cfg)
            
            logging.info(f"{model} on {next(model.parameters()).device}" )
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info(f'Num parameters: {cfg.params}')
            
            optimizer = create_optimizer(model, cfg)
            
            # if model_type == 'GAT' and ((cfg.train.batch_size == 32 and cfg.train.lr == 0.1) or (cfg.train.lr == 0.01 and cfg.model.batch_size_sampler == 32) or (cfg.train.lr == 0.01 and cfg.model.batch_size_sampler == 64 and cfg.model.num_neighbors in [5, 10])):
            #     continue

            # if model_type == 'VGAE' and cfg.model.out_channels == 32 and cfg.model.hidden_channels == 32 and cfg.train.batch_size == 32 and cfg.train.lr == 0.1 and (cfg.model.batch_size_sampler == 32 or (cfg.model.batch_size_sampler == 64 and cfg.model.num_neighbors == 5 and cfg.model.num_hops == 6)):
            #         continue
            # if model_type == 'GAE' and cfg.model.out_channels == 32 and cfg.model.hidden_channels == 32 and cfg.train.batch_size == 32 and cfg.train.lr == 0.1 and cfg.model.batch_size_sampler == 32:
            #     if (cfg.model.num_neighbors in [5, 10] and cfg.model.num_hops in [6, 7, 8]) or (cfg.model.num_neighbors == 20 and cfg.model.num_hops in [6]):
            #         continue
            # LLM: finetuning
            if cfg.train.finetune: 
                model = init_model_from_pretrained(model, cfg.train.finetune,
                                                cfg.train.freeze_pretrained)
                
            hyper_id = wandb.util.generate_id()
            cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}'
            # wandb.init(id=id, config=cfg, settings=wandb.Settings(_service_wait=300), save_code=True)
            # wandb.watch(model, log="all",log_freq=10)
            custom_set_run_dir(cfg, cfg.wandb.name_tag)
        
            # dump_run_cfg(cfg)
            print_logger.info(f"config saved into {cfg.run_dir}")
            print_logger.info(f'Run {0} with seed {0} and split {0} on device {cfg.device}')
            #print_logger.info(f'Run {run_id} with seed {seed} and split {split_index} on device {cfg.device}')
            
            cfg.model.params = params_count(model)
            print_logger.info(f'Num parameters: {cfg.model.params}')
            # cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
            print('DDEVICE: ', cfg.device)
            trainer = Trainer_NS(FILE_PATH,
                        cfg,
                        model, 
                        None, 
                        data,
                        optimizer,
                        splits,
                        #run_id, 
                        0,
                        args.repeat,
                        loggers, 
                        print_logger,
                        cfg.device,
                        cfg.model.batch_size_sampler,
                        cfg.model.num_neighbors,
                        cfg.model.num_hops)

            trainer.train()

            run_result = {}
            for key in trainer.loggers.keys():
                # refer to calc_run_stats in Logger class
                _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(0)#run_id)
                run_result.update({key: test_bvalid})
            for key in combined_hyperparameters.keys():
                run_result.update({key: getattr(cfg.model, key)})
            run_result.update({'epochs': cfg.train.epochs})
            
            print_logger.info(run_result)
            
            to_file = f'{cfg.data.name}_{cfg.model.type}_tune_result.csv'
            trainer.save_tune(run_result, to_file)
            save_results_to_file(run_result, cfg.model, output_dir)
            print_logger.info(f"runing time {time.time() - start_time}")
        
    # statistic for all runs


if __name__ == "__main__":
    project_main()
