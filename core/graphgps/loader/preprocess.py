import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils.load import load_data_lp
import torch
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from typing import Tuple, List, Dict
from torch_geometric.data import Data, InMemoryDataset
from yacs.config import CfgNode as CN


def add_node_embed(data: Data, 
                    splits: Dict[str, Data],
                    cfg: CN) -> Tuple[Data, Dict[str, Data], torch.nn.Embedding, CN, torch.Tensor]:
    """preprocess node feat: embed """

    edge_index = data.edge_index.to(cfg.device)
    emb = None # here is your embedding
    node_num = data.num_nodes

    if hasattr(data, 'x') and data.x != None:
        x = data.x
        cfg.model.input_channels = x.size(1)
    else:
        emb = torch.nn.Embedding(node_num, cfg.model.hidden_channels)
        cfg.model.input_channels = cfg.model.hidden_channels

    if not hasattr(data, 'edge_weight'): 
        train_edge_weight = torch.ones(splits['train'].edge_index.shape[1])
        train_edge_weight = train_edge_weight.to(torch.float)

    data = T.ToSparseTensor()(data)

    if cfg.train.use_valedges_as_input:
        # in the previous setting we share the same train and valid 
        val_edge_index = splits['valid'].edge_index.t()
        val_edge_index = to_undirected(val_edge_index)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1).to(cfg.device)

        edge_weight = torch.ones(full_edge_index.shape[1]).to(cfg.device)
        train_edge_weight = torch.ones(splits['train'].edge_index.shape[1])

        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])

        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        print(data.full_adj_t)
        print(data.adj_t)
    else:
        data.full_adj_t = data.adj_t

    if emb != None:
        torch.nn.init.xavier_uniform_(emb.weight)
        
    return data, splits, emb, cfg, train_edge_weight