import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
from graphgps.utility.utils import get_git_repo_root_path
from typing import Dict
import numpy as np
import scipy.sparse as ssp
import json
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.transforms import RandomLinkSplit
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.preprocessing import normalize
from yacs.config import CfgNode as CN
from data_utils.dataset import CustomLinkDataset
from data_utils.load_data_nc import load_tag_cora, load_tag_pubmed, \
    load_tag_product, load_tag_ogbn_arxiv, load_tag_product, \
    load_tag_arxiv23, load_graph_cora, load_graph_pubmed, \
    load_graph_arxiv23, load_graph_ogbn_arxiv, load_text_cora, \
    load_text_pubmed, load_text_arxiv23, load_text_ogbn_arxiv, \
    load_text_product

from graphgps.utility.utils import get_git_repo_root_path, config_device, init_cfg_test
from graphgps.utility.utils import time_logger
from typing import Dict, Tuple, List, Union

FILE = 'core/dataset/ogbn_products_orig/ogbn-products.csv'
FILE_PATH = get_git_repo_root_path() + '/'


# arxiv_2023
def load_taglp_arxiv2023(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data, text = load_tag_arxiv23()
    undirected = data.is_directed()

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


def load_taglp_cora(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data, data_citeid = load_graph_cora(False)
    text = load_text_cora(data_citeid)
    # text = None
    undirected = data.is_directed()

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


def load_taglp_ogbn_arxiv(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data = load_graph_ogbn_arxiv(False)
    text = load_text_ogbn_arxiv()
    undirected = data.is_directed()

    cfg = config_device(cfg)

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


def get_edge_split(data: Data,
                   undirected: bool,
                   device: Union[str, int],
                   val_pct: float,
                   test_pct: float,
                   include_negatives: bool,
                   split_labels: bool):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        RandomLinkSplit(is_undirected=undirected,
                        num_val=val_pct,
                        num_test=test_pct,
                        add_negative_train_samples=include_negatives,
                        split_labels=split_labels),

    ])
    del data.adj_t, data.e_id, data.batch_size, data.n_asin, data.n_id
    train_data, val_data, test_data = transform(data)
    return {'train': train_data, 'valid': val_data, 'test': test_data}


def load_taglp_product(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data, text = load_tag_product()
    undirected = data.is_directed()

    cfg = config_device(cfg)

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


def load_taglp_pubmed(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data = load_graph_pubmed(False)
    text = load_text_pubmed()
    undirected = data.is_directed()

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


# TEST CODE
if __name__ == '__main__':
    args = init_cfg_test()
    print(args)
    data, text, __  = load_taglp_arxiv2023(args.data)
    print(data)
    print(type(text))
    data, text = load_taglp_cora(args.data)
    print(data)
    print(type(text))

    data, text = load_taglp_ogbn_arxiv(args.data)
    print(data)
    print(type(text))

    data, text = load_taglp_product(args.data)
    print(data)
    print(type(text))

    data, text = load_taglp_pubmed(args.data)
    print(data)
    print(type(text))