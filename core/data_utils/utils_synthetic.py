"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import os
import scipy
import gdown
import pandas as pd

from sklearn.model_selection import train_test_split

import torch

import dgl
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add


def to_adj_matrix(edge_index, num_nodes_1, num_nodes_2):
    return torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1)), (num_nodes_1, num_nodes_2))


def normalize_adj(edge_index, num_nodes=None, edge_weight=None, direction='sym', self_loops=True):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    if self_loops:
        fill_value = 1.
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    if direction == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif direction == 'row':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    elif direction == 'col':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = edge_weight * deg_inv[col]
    else:
        raise ValueError()

    return torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))

def uniform_negative_sampling(g, num_edges, exact=True):
    num_nodes = g.num_nodes()

    if exact:
        pos_row, pos_col = g.edges()
        neighbors = {n: set(pos_col[pos_row == n]) for n in range(num_nodes)}
        neg_edge_index = []
        while True:
            src, dst = np.random.randint(num_nodes, size=2)
            if dst not in neighbors[src] and src != dst:
                neg_edge_index.append([src, dst])
            if len(neg_edge_index) == num_edges:
                break
        neg_edge_index = torch.tensor(neg_edge_index).int().to(g.device).T
    else:
        neg_edge_index = torch.randint(0, num_nodes, (2, num_edges), device=g.device)

    return neg_edge_index[0], neg_edge_index[1]


def construct_dgl_graph(edge_index, num_nodes, features, symmetric=True):
    if symmetric:
        row, col = edge_index
        g = dgl.graph((torch.cat([row, col]), torch.cat([col, row])), num_nodes=num_nodes)
    else:
        g = dgl.graph(edge_index, num_nodes=num_nodes)
    g.ndata['feat'] = features
    return g


def split_edges(g, ratio, threshold=1e6, seed=None):
    assert len(ratio) == 3 and sum(ratio) == 1

    features = g.ndata['feat']
    num_nodes = g.number_of_nodes()
    row, col = g.edges()
    row, col = row[row < col], col[row < col]
    num_edges = len(row)

    train_idx, test_idx = train_test_split(np.arange(num_edges),
                                           test_size=ratio[2],
                                           random_state=seed)
    train_idx, valid_idx = train_test_split(train_idx,
                                            test_size=ratio[1] / (ratio[0] + ratio[1]),
                                            random_state=seed)

    train_g = construct_dgl_graph((row[train_idx], col[train_idx]), num_nodes, features)
    test_g = construct_dgl_graph((row[train_idx], col[train_idx]), num_nodes, features)

    exact = True if num_edges < threshold else False
    edge_index_valid = torch.cat([torch.stack((row[valid_idx], col[valid_idx])), 
                                  torch.stack(uniform_negative_sampling(g, len(valid_idx), exact=exact))], dim=-1)
    edge_index_test = torch.cat([torch.stack((row[test_idx], col[test_idx])), 
                                 torch.stack(uniform_negative_sampling(g, len(test_idx), exact=exact))], dim=-1)
    
    y_valid = torch.cat([torch.ones(len(valid_idx)), torch.zeros(len(valid_idx))]).to(g.device)
    y_test = torch.cat([torch.ones(len(test_idx)), torch.zeros(len(test_idx))]).to(g.device)

    return train_g, test_g, (edge_index_valid, y_valid), (edge_index_test, y_test)

    
