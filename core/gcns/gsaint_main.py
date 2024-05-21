import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from itertools import product
from graphgps.network.gsaint import GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler
from graphgps.train.opt_train import Trainer, Trainer_Saint
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric import seed_everything
from torch_geometric.data.makedirs import makedirs
from torch_geometric.graphgym.utils.device import auto_select_device
from custom_main import run_loop_settings, custom_set_run_dir, set_printing
from data_utils.load import load_data_nc, load_data_lp
from graphgps.network.custom_gnn import create_model
from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger

def get_loader_RW(data, batch_size, walk_length, num_steps, sample_coverage):
    return GraphSAINTRandomWalkSampler(data, batch_size=batch_size, 
                                       walk_length=walk_length, 
                                       num_steps=num_steps, sample_coverage=sample_coverage)

if __name__ == "__main__":
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    torch.set_num_threads(cfg.num_threads)
    # Best params: {'batch_size': 64, 'walk_length': 10, 'num_steps': 30, 'sample_coverage': 100, 'accuracy': 0.82129}
    batch_sizes = [64]#[8, 16, 32, 64]
    walk_lengths = [10]#[10, 15, 20]
    num_steps = [30]#[10, 20, 30]
    sample_coverages = [100]#[50, 100, 150]

    best_acc = 0
    best_params = {}

    loggers = create_logger(args.repeat)
    
    for batch_size, walk_length, num_steps, sample_coverage in product(batch_sizes, walk_lengths, num_steps, sample_coverages):
        for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)): # In run_loop_settings we should send 2 parameeters

            # Set configurations for each run
            custom_set_run_dir(cfg, run_id)
            set_printing(cfg) # We should send cfg else we get error Attribute error: run_dir
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            cfg = config_device(cfg)
            splits, _, data = load_data_lp[cfg.data.name](cfg.data)
            cfg.model.in_channels = data.x.size(1)

            print_logger = set_printing(cfg)
            print_logger.info(f"The {cfg['data']['name']} graph {splits['train']['x'].shape} is loaded on {splits['train']['x'].device}, \n Train: {2*splits['train']['pos_edge_label'].shape[0]} samples,\n Valid: {2*splits['train']['pos_edge_label'].shape[0]} samples,\n Test: {2*splits['test']['pos_edge_label'].shape[0]} samples") 
        
            lst_args = cfg.model.type.split('_')
            if lst_args[0] == 'gsaint':
                sampler = get_loader
                cfg.model.type = lst_args[1]
            else:
                sampler = None 

            model = create_model(cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.base_lr)

            # Execute experiment
            trainer = Trainer_Saint(FILE_PATH,
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
                        cfg.device,
                        sampler, 
                        batch_size, 
                        walk_length, 
                        num_steps, 
                        sample_coverage)

            start = time.time()
            trainer.train()
            end = time.time()
            
            print('Training time: ', end - start)
                
        best_auc, best_hits = trainer.train()
        
        for key in loggers:
            print(key)
            trainer.loggers[key].print_statistics(run_id)
            
        print(trainer.results_rank)

    best_auc_metric, result_all_run = trainer.result_statistic()
    
    print(f"best_auc_metric: {best_auc_metric}, result_all_run: {result_all_run}")