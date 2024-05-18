# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import logging
import itertools
from tqdm import tqdm
import time
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.cmd_args import parse_args
import argparse
import wandb
from graphgps.train.opt_train import Trainer
from graphgps.network.custom_gnn import create_model
from graphgps.config import (dump_cfg, dump_run_cfg)

from data_utils.load import load_data_lp
from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger
import pprint

FILE_PATH = f'{get_git_repo_root_path()}/'

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gat.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='cora',
                        help='data name')
    parser.add_argument('--batch_size', dest='bs', type=int, required=False,
                        default=2**15,
                        help='data name')
    parser.add_argument('--device', dest='device', required=True, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=True,
                        default=300,
                        help='data name')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()


def project_main():
    
    # process params
    args = parse_args()

    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)
    
    cfg.data.name = args.data
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    
    # save params
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    
    # Set Pytorch environment
    torch.set_num_threads(20)

    loggers = create_logger(args.repeat)

    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        # Set configurations for each run TODO clean code here 
        id = wandb.util.generate_id()
        cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}' 
        custom_set_run_dir(cfg, cfg.wandb.name_tag)

        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        
        cfg = config_device(cfg)
        cfg.data.name = args.data

        splits, _, data = load_data_lp[cfg.data.name](cfg.data)
        cfg.model.in_channels = splits['train'].x.shape[1]

        print_logger = set_printing(cfg)
        print_logger.info(f"The {cfg['data']['name']} graph {splits['train']['x'].shape} is loaded on {splits['train']['x'].device}, \n Train: {2*splits['train']['pos_edge_label'].shape[0]} samples,\n Valid: {2*splits['train']['pos_edge_label'].shape[0]} samples,\n Test: {2*splits['test']['pos_edge_label'].shape[0]} samples")
        dump_cfg(cfg)    

        # hyperparameter_search = {'out_channels': [160, 176], 'hidden_channels': [160, 176]} GCN, GAE, VGAE
        hyperparameter_search = {'out_channels': [160, 176], 'hidden_channels': [160, 176],
                              "batch_size": [2**13, 2**14], "lr": [0.01, 0.008, 0.007, 0.015]}
        
        print_logger.info(f"hypersearch space: {hyperparameter_search}")
        for out,  hidden, bs, lr in tqdm(itertools.product(*hyperparameter_search.values())):
            cfg.model.out_channels = out
            cfg.model.hidden_channels = hidden
            cfg.train.batch_size =  bs
            cfg.optimizer.base_lr = lr
            print_logger.info(f"out : {out}, hidden: {hidden}")
            print_logger.info(f"bs : {cfg.train.batch_size}, lr: {cfg.optimizer.base_lr}")
                        
            start_time = time.time()
                
            model = create_model(cfg)
            
            logging.info(f"{model} on {next(model.parameters()).device}" )
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info(f'Num parameters: {cfg.params}')

            optimizer = create_optimizer(model, cfg)

            # LLM: finetuning
            if cfg.train.finetune: 
                model = init_model_from_pretrained(model, cfg.train.finetune,
                                                cfg.train.freeze_pretrained)
            hyper_id = wandb.util.generate_id()
            cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}' 
            custom_set_run_dir(cfg, cfg.wandb.name_tag)
        
            dump_run_cfg(cfg)
            print_logger.info(f"config saved into {cfg.run_dir}")
            print_logger.info(f'Run {run_id} with seed {seed} and split {split_index} on device {cfg.device}')
            
            
            trainer = Trainer(FILE_PATH,
                        cfg,
                        model, 
                        None, 
                        data,
                        optimizer,
                        splits,
                        run_id, 
                        args.repeat,
                        loggers, 
                        print_logger,
                        cfg.device)

            trainer.train()

            run_result = {}
            for key in trainer.loggers.keys():
                # refer to calc_run_stats in Logger class
                _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
                run_result.update({key: test_bvalid})
            
            cfg.model.out_channels = out
            cfg.model.hidden_channels = hidden
            cfg.train.batch_size =  bs
            cfg.optimizer.base_lr = lr
            
            run_result.update({"out_channels": out, "hidden_channels": hidden, "batch_size": bs, "lr": lr})
            print_logger.info(run_result)
            
            to_file = f'{cfg.data.name}_{cfg.model.type}_tune_result.csv'
            trainer.save_tune(run_result, to_file)
            
            print_logger.info(f"runing time {time.time() - start_time}")
        
    # statistic for all runs


if __name__ == "__main__":
    project_main()
