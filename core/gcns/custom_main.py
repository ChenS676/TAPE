# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import *
import torch
import logging
import os.path as osp 

from torch_geometric import seed_everything
from torch_geometric.data.makedirs import makedirs
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (dump_cfg, 
                                             makedirs_rm_exist)
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from graphgps.train.opt_train import Trainer
from graphgps.network.custom_gnn import create_model
from data_utils.load import load_data_nc, load_data_lp
from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings

print("modules loaded")

if __name__ == "__main__":

    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    # Load args file
    
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    dump_cfg(cfg)

    # Set Pytorch environment
    torch.set_num_threads(cfg.run.num_threads)

    loggers = create_logger(args.repeat)
    
    splits, text = load_data_lp[cfg.data.name](cfg.data)
    
    # LLM: embeddings
    if cfg.llm.llm_embedding == True:
        print("Using LLM Embeddings")
        model_type = cfg.llm.model_type
        model_name = cfg.llm.model_name
        batch_size = cfg.llm.batch_size
        embeddings = use_pretrained_llm_embeddings(model_type, model_name, text, batch_size)
        for split in splits:
            splits[split].x = embeddings

    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)

        set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        
        print(splits)
        print(splits['train'].x.shape[1])
        cfg.model.in_channels = splits['train'].x.shape[1]
        model = create_model(cfg)

        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info(f'Num parameters: {cfg.params}')

        optimizer = create_optimizer(model, cfg)


        trainer = Trainer(FILE_PATH,
                    cfg,
                    model, 
                    optimizer,
                    splits,
                    run_id, 
                    args.repeat,
                    loggers)

        trainer.train()

    # statistic for all runs
    print('All runs:')
    
    result_dict = {}
    for key in loggers:
        print(key)
        _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
        result_dict.update({key: valid_test})

    trainer.save_result(result_dict)
