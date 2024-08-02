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

from graphgps.train.gsaint_train import  Trainer_Saint

from graphgps.network.gsaint import GraphSAINTRandomWalkSampler
from data_utils.load import load_data_lp
from graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger, get_logger, LinearDecayLR
import pprint
from final_gnn_tune import create_GAE_model

FILE_PATH = f'{get_git_repo_root_path()}/'

def get_loader_RW(data, batch_size, walk_length, num_steps, sample_coverage):
    return GraphSAINTRandomWalkSampler(data, batch_size=batch_size, 
                                       walk_length=walk_length, 
                                       num_steps=num_steps, sample_coverage=sample_coverage)
def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/heart_gnn_models.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='pubmed',
                        help='data name')
    parser.add_argument('--batch_size', dest='bs', type=int, required=False,
                        default=2**15,
                        help='data name')
    parser.add_argument('--sampler', dest='sampler', type=str, required=False,
                        default='gsaint',
                        help='data name')
    parser.add_argument('--device', dest='device', required=True, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=True,
                        default=400,
                        help='data name')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        default='GCN_Variant',
                        help='model name')
    parser.add_argument('--score', dest='score', type=str, required=False, default='mlp_score',
                        help='decoder name')
    parser.add_argument('--wandb', dest='wandb', required=False, 
                        help='data name')
    parser.add_argument('--repeat', type=int, default=3,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()

product_space = {
    '0': 'inner', 
    '1': 'dot'    
}


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
    result_df['BatchSizeSampler'] = cfg.batch_size_sampler
    result_df['HiddenChannels'] = cfg.hidden_channels
    result_df['OutChannels'] = cfg.out_channels
    result_df['NumSteps'] = cfg.num_steps
    result_df['SampleCoverage'] = cfg.sample_coverage
    result_df['WalkLength'] = cfg.walk_length
    
    # Specify the output file path
    output_file = os.path.join(output_dir, 'results_summary.csv')
    
    # Check if file exists to append or write header
    if os.path.exists(output_file):
        result_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(output_file, mode='w', header=True, index=False)
    
    print(f"Results saved to {output_file}")

hyperparameter_space = {
    'GAT_Variant': {'out_channels': [2**4, 2**5, 2**6], 'hidden_channels':  [2**5, 2*4],
                                'heads': [2**2, 2, 2**3], 'negative_slope': [0.1], 'dropout': [0, 0.1], 
                                'num_layers': [1, 2, 3], 
                                'base_lr': [0.015],
                                'score_num_layers_predictor': [1, 2, 3],
                                'score_gin_mlp_layer': [2],
                                'score_hidden_channels': [2**6, 2**5, 2**4], 
                                'score_out_channels': [1], 
                                'score_num_layers': [1, 2, 3], 
                                'score_dropout': [0.1], 
                                'product': [0, 1]},
    
    'GCN_Variant': {'out_channels': [2**4, 2**5, 2**6], 
                    'hidden_channels': [2**4, 2**5, 2**6], 
                    'batch_size': [2**10],
                    'dropout': [0.1, 0],
                    'num_layers': [1, 2, 3],
                    'negative_slope': [0.1],
                    'base_lr': [0.015, 0.001],
                    'score_num_layers_predictor': [1, 2, 3],
                    'score_gin_mlp_layer': [2],
                    'score_hidden_channels': [2**6, 2**5, 2**4], 
                    'score_out_channels': [1], 
                    'score_num_layers': [1, 2, 3], 
                    'score_dropout': [0.1], 
                    'product': [0, 1]},
                    

    'SAGE_Variant': {'out_channels': [2**4, 2**5, 2**6, 2**7, 2**8, 2**9], 
                     'hidden_channels': [2**4, 2**5, 2**6, 2**7, 2**8, 2**9], 
                     'base_lr': [0.015, 0.1, 0.01],
                    'score_num_layers_predictor': [1, 2, 3],
                    'score_gin_mlp_layer': [2],
                    'score_hidden_channels': [2**6, 2**5, 2**4], 
                    'score_out_channels': [1], 
                    'score_num_layers': [1, 2, 3], 
                    'score_dropout': [0.1], 
                    'product': [0, 1],
                     },
    
    'GIN_Variant': {'out_channels': [2**4, 2**5, 2**6, 2**7, 2**8, 2**9], 
                    'hidden_channels': [2**4, 2**5, 2**6, 2**7, 2**8, 2**9],
                    'num_layers': [1, 2, 3, 4], 
                    'base_lr': [0.015, 0.1, 0.01],
                    'mlp_layer': [1, 2, 3],
                    'score_num_layers_predictor': [1, 2, 3],
                    'score_gin_mlp_layer': [2],
                    'score_hidden_channels': [2**6, 2**5, 2**4], 
                    'score_out_channels': [1], 
                    'score_num_layers': [1, 2, 3], 
                    'score_dropout': [0.1], 
                    'product': [0, 1],
                    'batch_size': [2**14],
                    'lr': [0.01],
                    'batch_size_sampler': [2**10], # 32, 64 very bad we get very sparse graphs for Cora
                                                # 32, 64, 128, 256 very bad we get very sparse graphs for Arxiv_2023
                    'walk_length'       : [10],
                    'num_steps'         : [10],
                    'sample_coverage'   : [100]
                    },
    
    'VGAE_Variant': {'out_channels': [32], 'hidden_channels': [32], 'batch_size': [2**10]},
}

hyperparameter_gsaint = {
        'sampler_batch_size': [32],
        'lr': [0.01],
        'batch_size_sampler': [1024], # 32, 64 very bad we get very sparse graphs for Cora
                                     # 32, 64, 128, 256 very bad we get very sparse graphs for Arxiv_2023
        'walk_length'       : [10],
        'num_steps'         : [10],
        'sample_coverage'   : [100]
}

yaml_file = {   
             'GAT_Variant': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
             'GAE_Variant': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
             'GIN_Variant': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
             'GCN_Variant': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
             'SAGE_Variant': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
             'DGCNN': 'core/yamls/cora/gcns/heart_gnn_models.yaml'
            }

def project_main():
    
    args = parse_args()

    args.cfg_file = yaml_file[args.model]

    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.name = args.data
    
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    cfg.sampler.type = args.sampler
    
    cfg_model = eval(f'cfg.model.{args.model}')
    cfg_score = eval(f'cfg.score.{args.score}')
    cfg_sampler = eval(f'cfg.sampler.{args.sampler}')
    cfg.model.type = args.model
    # save params
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)

    # Set Pytorch environment
    torch.set_num_threads(20)

    loggers = create_logger(args.repeat)
    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        # Set configurations for each run TODO clean code here 
        id = wandb.util.generate_id()
        if args.wandb:
            cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{args.model}'

        custom_set_run_dir(cfg, cfg.wandb.name_tag)

        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)

        cfg = config_device(cfg)

        splits, _, data = load_data_lp[cfg.data.name](cfg.data)
        cfg_model.in_channels = splits['train'].x.shape[1]

        print_logger = set_printing(cfg)
        print_logger.info(
            f"The {cfg['data']['name']} graph with shape {splits['train']['x'].shape} is loaded on {splits['train']['x'].device},\n"
            f"Split index: {cfg['data']['split_index']} based on {data.edge_index.size(1)} samples.\n"
            f"Train: {cfg['data']['split_index'][0]}% ({2 * splits['train']['pos_edge_label'].shape[0]} samples),\n"
            f"Valid: {cfg['data']['split_index'][1]}% ({2 * splits['valid']['pos_edge_label'].shape[0]} samples),\n"
            f"Test:  {cfg['data']['split_index'][2]}% ({2 * splits['test']['pos_edge_label'].shape[0]} samples)"
        )

        dump_cfg(cfg)
        hyperparameter_gnn = hyperparameter_space[args.model]
        print_logger.info(f"hypersearch space: {hyperparameter_gnn}")

        keys = hyperparameter_gnn.keys()
        # Generate Cartesian product of the hyperparameter values
        product = itertools.product(*hyperparameter_gnn.values())

        for combination in product:
            for key, value in zip(keys, combination):
                if key == 'product':
                    value = product_space[str(value)]
                if hasattr(cfg_model, key):
                    print_logger.info(f"Object cfg_model has attribute '{key}' with value: {getattr(cfg_model, key)}")
                    setattr(cfg_model, key, value)
                    print_logger.info(f"Object cfg_model.{key} updated to {getattr(cfg_model, key)}")
                elif hasattr(cfg_score, key):
                    print_logger.info(f"Object cfg_score has attribute '{key}' with value: {getattr(cfg_score, key)}")
                    setattr(cfg_score, key, value)
                    print_logger.info(f"Object cfg_score.{key} updated to {getattr(cfg_score, key)}")
                elif hasattr(cfg_sampler, key):
                    print_logger.info(f"Object cfg_score has attribute '{key}' with value: {getattr(cfg_sampler, key)}")
                    setattr(cfg_sampler, key, value)
                    print_logger.info(f"Object cfg_score.{key} updated to {getattr(cfg_sampler, key)}")
                                        
                elif hasattr(cfg.train, key):
                    print_logger.info(f"Object cfg.train has attribute '{key}' with values {getattr(cfg.train, key)}")
                    setattr(cfg.train, key, value)
                    print_logger.info(f"Object cfg.train.{key} updated to {getattr(cfg.train, key)}")
                elif hasattr(cfg.optimizer, key):    
                    print_logger.info(f"Object cfg.train has attribute '{key}' with values {getattr(cfg.optimizer, key)}")
                    setattr(cfg.train, key, value)
                    print_logger.info(f"Object cfg.train.{key} updated to {getattr(cfg.optimizer, key)}")
                
            cfg.sampler.gsaint = cfg_sampler
            cfg.train.lr = cfg.optimizer.base_lr
            print_logger.info(f"out : {cfg_model.out_channels}, hidden: {cfg_model.hidden_channels}")
            print_logger.info(f"bs : {cfg.train.batch_size}, lr: { cfg.optimizer.base_lr}")
                        
            start_time = time.time()
                
            model = create_GAE_model(cfg_model, cfg_score, args.model)
            
            logging.info(f"{model} on {next(model.parameters()).device}" )
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info(f'Num parameters: {cfg.params}')

            optimizer = create_optimizer(model, cfg)
            scheduler = LinearDecayLR(optimizer, start_lr=0.01, end_lr=0.001, num_epochs=cfg.train.epochs)
            
            # LLM: finetuning
            if cfg.train.finetune: 
                model = init_model_from_pretrained(model, cfg.train.finetune,
                                                cfg.train.freeze_pretrained)
                
            hyper_id = wandb.util.generate_id()
            cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}' 
            custom_set_run_dir(cfg, cfg.wandb.name_tag)
        
            # dump_run_cfg(cfg)
            print_logger.info(f"config saved into {cfg.run_dir}")
            print_logger.info(f'Run {run_id} with seed {seed} and split {split_index} on device {cfg.device}')
            
            cfg.model.params = params_count(model)
            print_logger.info(f'Num parameters: {cfg.model.params}')

            if cfg.model.sampler == 'gsaint':
                sampler = get_loader_RW

                trainer = Trainer_Saint(
                    FILE_PATH=FILE_PATH,
                    cfg=cfg, 
                    model=model,
                    emb=None,
                    data=data,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    splits=splits, 
                    run=run_id, 
                    repeat=args.repeat,
                    loggers=loggers,
                    print_logger=print_logger,
                    device=cfg.device,
                    gsaint=sampler, 
                    )

            trainer.train()

            run_result = {}
            for key in trainer.loggers.keys():
                # refer to calc_run_stats in Logger class
                _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
                run_result.update({key: test_bvalid})
            for k in hyperparameter_gnn.keys():
                if hasattr(cfg_model, k):
                    run_result[k] = getattr(cfg_model, k)
                if hasattr(cfg_score, k):
                    run_result[k] = getattr(cfg_score, k)
                elif hasattr(cfg.train, k):
                    run_result[k] = getattr(cfg.train, k)
                elif hasattr(cfg.optimizer, k):
                    run_result[k] = getattr(cfg.optimizer, k)
                    
            run_result.update({'epochs': cfg.train.epochs})
            
            print_logger.info(run_result)
            
            to_file = f'{cfg.data.name}_{cfg.model.type}_tune_result.csv'
            trainer.save_tune(run_result, to_file)
            save_results_to_file(run_result, cfg.model, cfg.out_dir)
            print_logger.info(f"runing time {time.time() - start_time}")
    
# statistic for all runs


if __name__ == "__main__":
    project_main()