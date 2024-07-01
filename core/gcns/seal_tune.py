# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import copy
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import logging
import itertools
from tqdm import tqdm
import argparse
import wandb
import time
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
import scipy.sparse as ssp
from graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
          create_optimizer, config_device,  create_logger, custom_set_out_dir
from data_utils.load_data_nc import load_graph_cora, load_graph_pubmed, load_tag_arxiv23, load_graph_ogbn_arxiv
from graphgps.encoder.seal import get_pos_neg_edges, extract_enclosing_subgraphs, k_hop_subgraph, construct_pyg_graph, do_edge_split
from graphgps.config import (dump_cfg, dump_run_cfg)
from graphgps.train.seal_train import Trainer_SEAL
from graphgps.network.heart_gnn import DGCNN
from data_utils.seal_loader import (
    SEALDataset, 
    SEALDynamicDataset
)

FILE_PATH = f'{get_git_repo_root_path()}/'


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/seal.yaml',
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
    parser.add_argument('--wandb', dest='if_wandb', required=False, 
                        help='data name')
    parser.add_argument('--repeat', type=int, default=3,
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
        id = wandb.util.generate_id()
        cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}'
        custom_set_run_dir(cfg, cfg.wandb.name_tag)

        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)

        cfg = config_device(cfg)

        if cfg.data.name == 'pubmed':
            data = load_graph_pubmed(False)
        elif cfg.data.name == 'cora':
            data, _ = load_graph_cora(False)
        elif cfg.data.name == 'arxiv_2023':
            data, _ = load_tag_arxiv23()
        elif cfg.data.name == 'ogbn-arxiv':
            data = load_graph_ogbn_arxiv(False)
        splits = do_edge_split(copy.deepcopy(data), cfg.data.val_pct, cfg.data.test_pct)
        path = f'{os.path.dirname(__file__)}/seal_{cfg.data.name}'
        dataset = {
            'train': SEALDataset(
                path,
                data,
                splits,
                num_hops=cfg.model.num_hops,
                split='train',
                node_label=cfg.model.node_label,
                directed=not cfg.data.undirected,
            )
        }
        dataset['valid'] = SEALDataset(
            path,
            data,
            splits,
            num_hops=cfg.model.num_hops,
            split='valid',
            node_label=cfg.model.node_label,
            directed=not cfg.data.undirected,
        )
        dataset['test'] = SEALDataset(
            path,
            data,
            splits,
            num_hops=cfg.model.num_hops,
            split='test',
            node_label=cfg.model.node_label,
            directed=not cfg.data.undirected,
        )
        print_logger = set_printing(cfg)
        print_logger.info(
            f"The {cfg['data']['name']} graph {dataset['train'].data.x.shape[0]} is loaded on {dataset['train'].data.x.device}, \n Train: {2 * dataset['train'].data['edge_index'].shape[1]} samples,\n Valid: {2 * dataset['valid'].data['edge_index'].shape[1]} samples,\n Test: {2 * dataset['test'].data['edge_index'].shape[1]} samples")
        dump_cfg(cfg)

        hyperparameter_search = {'hidden_channels': [32, 64, 128, 256],
                                 "batch_size": [32, 64, 128, 256], "lr": [0.001, 0.0001]}

        print_logger.info(f"hypersearch space: {hyperparameter_search}")
        for hidden_channels, batch_size, lr in tqdm(itertools.product(*hyperparameter_search.values())):
            cfg.model.hidden_channels = hidden_channels
            cfg.train.batch_size =  batch_size
            cfg.optimizer.base_lr = lr
            print_logger.info(f"hidden: {hidden_channels}")
            print_logger.info(f"bs : {cfg.train.batch_size}, lr: {cfg.optimizer.base_lr}")

            start_time = time.time()

            model = DGCNN(cfg.model.hidden_channels, cfg.model.num_layers, cfg.model.max_z, cfg.model.k,
                          dataset['train'], False, use_feature=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.base_lr)

            logging.info(f"{model} on {next(model.parameters()).device}" )
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info(f'Num parameters: {cfg.params}')

            hyper_id = wandb.util.generate_id()
            cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}'
            custom_set_run_dir(cfg, cfg.wandb.name_tag)

            dump_run_cfg(cfg)
            print_logger.info(f"config saved into {cfg.run_dir}")
            print_logger.info(f'Run {run_id} with seed {seed} and split {split_index} on device {cfg.device}')

            trainer = Trainer_SEAL(FILE_PATH,
                                   cfg,
                                   model,
                                   None,
                                   optimizer,
                                   dataset,
                                   run_id,
                                   args.repeat,
                                   loggers,
                                   print_logger,
                                   cfg.device,
                                   args.if_wandb)

            trainer.train()

            run_result = {}
            for key in trainer.loggers.keys():
                # refer to calc_run_stats in Logger class
                _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
                run_result[key] = test_bvalid


            run_result.update({"hidden_channels": hidden_channels, "batch_size": batch_size, "lr": lr})
            print_logger.info(run_result)

            to_file = f'{cfg.data.name}_{cfg.model.type}_tune_result.csv'
            trainer.save_tune(run_result, to_file)

            print_logger.info(f"runing time {time.time() - start_time}")
            print_logger.info(f"runing time {time.time() - start_time}")
        
    # statistic for all runs


if __name__ == "__main__":
    project_main()
