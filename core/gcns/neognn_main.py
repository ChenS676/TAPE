import copy
import os, sys

from torch_sparse import SparseTensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import torch
from functools import partial

from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
    create_optimizer, config_device, \
    create_logger

from torch_geometric.data import InMemoryDataset, Dataset
from data_utils.load_data_nc import load_graph_cora, load_graph_pubmed, load_tag_arxiv23, load_graph_ogbn_arxiv
import scipy.sparse as ssp

from graphgps.network.neognn import NeoGNN, LinkPredictor
from data_utils.load import load_data_lp
from graphgps.train.neognn_train import Trainer_NeoGNN


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

    parser.add_argument('--repeat', type=int, default=2,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()

def ngnn_dataset(data, splits):
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.emb = torch.nn.Embedding(data.num_nodes, cfg.model.hidden_channels)
    edge_weight = torch.ones(edge_index.size(1), dtype=float)
    edge_index = edge_index.cpu()
    edge_weight = edge_weight.cpu()
    data.A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])),
                       shape=(data.num_nodes, data.num_nodes))
    return data



if __name__ == "__main__":
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    torch.set_num_threads(cfg.num_threads)
    batch_sizes = [cfg.train.batch_size]

    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)

    for batch_size in batch_sizes:
        for run_id, seed, split_index in zip(
                *run_loop_settings(cfg, args)):
            custom_set_run_dir(cfg, run_id)
            set_printing(cfg)
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            cfg = config_device(cfg)
            splits, text, data = load_data_lp[cfg.data.name](cfg.data)
            data.edge_index = splits['train']['pos_edge_label_index']
            data = ngnn_dataset(data, splits).to(cfg.device)
            path = f'{os.path.dirname(__file__)}/neognn_{cfg.data.name}'
            dataset = {}

            model = NeoGNN(cfg.model.hidden_channels, cfg.model.hidden_channels,
                           cfg.model.hidden_channels, cfg.model.num_layers,
                           cfg.model.dropout, args=cfg.model)

            predictor = LinkPredictor(cfg.model.hidden_channels, cfg.model.hidden_channels, 1,
                                      cfg.model.num_layers, cfg.model.dropout)

            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(data.emb.parameters()) +
                list(predictor.parameters()), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

            # Execute experiment
            trainer = Trainer_NeoGNN(FILE_PATH,
                                   cfg,
                                   model,
                                   predictor,
                                   optimizer,
                                   data,
                                   splits,
                                   run_id,
                                   args.repeat,
                                   loggers,
                                   print_logger=None,
                                   batch_size=batch_size)

            start = time.time()
            trainer.train()
            end = time.time()
            print('Training time: ', end - start)

        print('All runs:')

        result_dict = {}
        for key in loggers:
            print(key)
            _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
            result_dict[key] = valid_test

        trainer.save_result(result_dict)
