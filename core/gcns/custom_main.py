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
import argparse 

from graphgps.train.opt_train import Trainer
from graphgps.network.custom_gnn import create_model
from data_utils.load import load_data_lp
from utils import set_cfg, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger
import pprint

FILE_PATH = f'{get_git_repo_root_path()}/'

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/graphsage.yaml',
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
    parser.add_argument('--epochs', dest='epochs', type=int, required=True,
                        default=300,
                        help='data name')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()


# TODO add weight biases to report result
def project_main():
    
    args = parse_args()
    # Load args file

    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    dump_cfg(cfg)
    
    
    cfg.data.name = args.data
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epochs
    
    pprint.pprint(cfg)

    # Set Pytorch environment
    torch.set_num_threads(cfg.run.num_threads)

    loggers = create_logger(args.repeat)

    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        # Set configurations for each run
        custom_set_run_dir(cfg, cfg.wandb.name_tag)

        print_logger = set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)

        # load tag 
        splits, text, data = load_data_lp[cfg.data.name](cfg.data)

        cfg.model.in_channels = splits['train'].x.shape[1]
        model = create_model(cfg)

        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info(f'Num parameters: {cfg.params}')

        optimizer = create_optimizer(model, cfg)

        # LLM: finetuning
        if cfg.train.finetune: 
            model = init_model_from_pretrained(model, cfg.train.finetune,
                                               cfg.train.freeze_pretrained)

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

        run_result = dict.fromkeys(trainer.loggers.keys())
        for key in trainer.loggers.keys():
            # refer to calc_run_stats in Logger class
            _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
            run_result[key] = test_bvalid
        print(run_result)

        trainer.save_result(run_result)

    if args.repeat > 1:
        
        print('All runs:')
        result_dict = {}
        for key in loggers:
            print(key)
            _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
            result_dict.update({key: valid_test})

        trainer.save_result(result_dict)


if __name__ == "__main__":
    project_main()
