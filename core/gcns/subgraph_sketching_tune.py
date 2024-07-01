import copy
import itertools
import os, sys

from torch_geometric.typing import torch_sparse
from torch_sparse import SparseTensor
from tqdm import tqdm
from torch_geometric.nn.conv.gcn_conv import gcn_norm


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphgps.network.subgraph_sketching import ElphHashes

import argparse
import time
import logging
import wandb
import torch
from functools import partial
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
          create_optimizer, config_device,  create_logger, custom_set_out_dir

from torch_geometric.data import InMemoryDataset, Dataset
from data_utils.load_data_nc import load_graph_cora, load_graph_pubmed, load_tag_arxiv23, load_graph_ogbn_arxiv
import scipy.sparse as ssp
from graphgps.config import (dump_cfg, dump_run_cfg)
from graphgps.network.subgraph_sketching import BUDDY, ELPH

from data_utils.load import load_data_lp
from graphgps.train.subgraph_sketching_train import Trainer_Subgraph_Sketching


def _generate_sign_features(data, edge_index, edge_weight, sign_k):
    """
    Generate features by preprocessing using the Scalable Inception Graph Neural Networks (SIGN) method
     https://arxiv.org/abs/2004.11198
    @param data: A pyg data object
    @param sign_k: the maximum number of times to apply the propagation operator
    @return:
    """
    try:
        num_nodes = data.x.size(0)
    except AttributeError:
        num_nodes = data.num_nodes
    edge_index, edge_weight = gcn_norm(  # yapf: disable
        edge_index, edge_weight.float(), num_nodes)
    if sign_k == 0:
        # for most datasets it works best do one step of propagation
        xs = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
    else:
        xs = [data.x]
        for _ in range(sign_k):
            x = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
            xs.append(x)
        xs = torch.cat(xs, dim=-1)
    return xs


def _preprocess_subgraph_features(num_nodes, edge_index, edges):
    """
    Handles caching of hashes and subgraph features where each edge is fully hydrated as a preprocessing step
    Sets self.subgraph_features
    @return:
    """
    elph_hashes = ElphHashes(cfg.model)
    hashes, cards = elph_hashes.build_hash_tables(num_nodes, edge_index)
    subgraph_features = elph_hashes.get_subgraph_features(edges, hashes, cards, cfg.train.batch_size)
    if cfg.model.floor_sf and subgraph_features is not None:
        subgraph_features[subgraph_features < 0] = 0
    if not cfg.model.use_zero_one and subgraph_features is not None:  # knock out the zero_one features (0,1) and (1,0)
        if cfg.model.max_hash_hops > 1:
            subgraph_features[:, [4, 5]] = 0
        if cfg.model.max_hash_hops == 3:
            subgraph_features[:, [11, 12]] = 0  # also need to get rid of (0, 2) and (2, 0)
    return subgraph_features
def hash_dataset(splits):
    data = splits.cpu()
    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    edge_index = data.edge_index.cpu()
    edge_weight = edge_weight.cpu()
    data.links = torch.cat([data['pos_edge_label_index'], data['neg_edge_label_index']], dim=1)
    data.A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])),
                            shape=(data.num_nodes, data.num_nodes))
    data.x = _generate_sign_features(data, edge_index, edge_weight, cfg.model.sign_k)
    data.subgraph_features = _preprocess_subgraph_features(data.num_nodes, data.edge_index, data.links.T)
    data.degrees = torch.tensor(data.A.sum(axis=0, dtype=float), dtype=torch.float).flatten()
    return data


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/seal.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        default='cora',
                        help='data name')
    parser.add_argument('--device', dest='device', required=True,
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=False,
                        default=100,
                        help='data name')
    parser.add_argument('--repeat', type=int, default=2,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()



if __name__ == "__main__":
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)

    torch.set_num_threads(cfg.num_threads)
    batch_sizes = [cfg.train.batch_size]

    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)


    for run_id, seed, split_index in zip(
                *run_loop_settings(cfg, args)):
        id = wandb.util.generate_id()
        custom_set_run_dir(cfg, run_id)
        set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)
        splits, text, data = load_data_lp[cfg.data.name](cfg.data)
        data.edge_index = torch.cat([splits['train']['pos_edge_label_index'],
                                        splits['train']['neg_edge_label_index']], dim=1)
        path = f'{os.path.dirname(__file__)}/{cfg.model.type}_{cfg.data.name}'
        print_logger = set_printing(cfg)
        print_logger.info(
            f"The {cfg['data']['name']} graph {splits['train']['x'].shape} is loaded on {splits['train']['x'].device},"
            f"\n Train: {2 * splits['train']['pos_edge_label'].shape[0]} samples,"
            f"\n Valid: {2 * splits['train']['pos_edge_label'].shape[0]} samples,"
            f"\n Test: {2 * splits['test']['pos_edge_label'].shape[0]} samples")
        dump_cfg(cfg)
        hyperparameter_search = {'hidden_channels': [128, 256, 512, 1024], 'num_layers': [2, 3],
                                 "batch_size": [512, 1024], "lr": [0.01, 0.001, 0.0001],
                                 'max_hash_hops': [2, 3], 'label_dropout': [0.1, 0.3, 0.5],
                                 'feature_dropout': [0.1, 0.3, 0.5], 'sign_dropout': [0.1, 0.3, 0.5],}
        print_logger.info(f"hypersearch space: {hyperparameter_search}")
        for hidden_channels, num_layers, batch_size, lr, max_hash_hops, label_dropout, feature_dropout, sign_dropout in tqdm(
                itertools.product(*hyperparameter_search.values())):
            cfg.model.hidden_channels = hidden_channels
            cfg.train.batch_size = batch_size
            cfg.optimizer.lr = lr
            cfg.model.num_layers = num_layers
            cfg.model.max_hash_hops = max_hash_hops
            cfg.model.label_dropout = label_dropout
            cfg.model.feature_dropout = feature_dropout
            cfg.model.sign_dropout = sign_dropout
            print_logger.info(
                f"hidden_channels: {hidden_channels}, num_layers: {num_layers}, batch_size: {batch_size}, lr: {lr}")
            start_time = time.time()
            if cfg.model.type == 'BUDDY':
                splits['train'] = hash_dataset(splits['train'])
                splits['valid'] = hash_dataset(splits['valid'])
                splits['test'] = hash_dataset(splits['test'])
                model = BUDDY(cfg.model, data.num_features, node_embedding=None)
            elif cfg.model.type == 'ELPH':
                model = ELPH(cfg.model, data.num_features, node_embedding=None)

            optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

            logging.info(f"{model} on {next(model.parameters()).device}")
            logging.info(cfg)
            cfg.params = params_count(model)
            logging.info(f'Num parameters: {cfg.params}')

            hyper_id = wandb.util.generate_id()
            cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}'
            custom_set_run_dir(cfg, cfg.wandb.name_tag)

            dump_run_cfg(cfg)
            print_logger.info(f"config saved into {cfg.run_dir}")
            print_logger.info(f'Run {run_id} with seed {seed} and split {split_index} on device {cfg.device}')

            # Execute experiment
            trainer = Trainer_Subgraph_Sketching(FILE_PATH,
                                  cfg,
                                  model,
                                  optimizer,
                                  data,
                                  splits,
                                  run_id,
                                  args.repeat,
                                  loggers,
                                  print_logger=print_logger,
                                  batch_size=batch_size)

            trainer.train()

            run_result = {}
            for key in trainer.loggers.keys():
                if trainer.loggers[key].results == [[], []]:
                    run_result[key] = None
                else:
                # refer to calc_run_stats in Logger class
                    _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
                    run_result[key] = test_bvalid

            run_result.update(
                {'hidden_channels': hidden_channels,' num_layers': num_layers, 'batch_size': batch_size,'lr': lr,
                    'max_hash_hops': max_hash_hops, 'label_dropout': label_dropout, 'feature_dropout': feature_dropout,
                    'sign_dropout': sign_dropout
                 })
            print_logger.info(run_result)

            to_file = f'{cfg.data.name}_{cfg.model.type}_tune_result.csv'
            trainer.save_tune(run_result, to_file)

            print_logger.info(f"runing time {time.time() - start_time}")




