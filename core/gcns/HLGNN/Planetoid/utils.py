import math
from tqdm import tqdm
import random
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import (negative_sampling, add_self_loops, train_test_split_edges)
import torch_sparse
from torch_geometric.utils import degree
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from typing import Optional

# A_sym - symmetrical matrix
def symmetrical_matrix(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    adj_t = edge_index

    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1., dtype=dtype)
    if add_self_loops:
        adj_t = torch_sparse.fill_diag(adj_t, 1)

    deg = torch_sparse.sum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = torch_sparse.mul(deg_inv_sqrt.view(-1, 1), adj_t)
    adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

    return adj_t

# A_rs - Row-Stochastic Matrix
def row_stochastic_matrix(    
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    adj_t = edge_index

    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1., dtype=dtype)
    if add_self_loops:
        adj_t = torch_sparse.fill_diag(adj_t, 1)

    deg = torch_sparse.sum(adj_t, dim=1)
    deg_inv = deg.pow_(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
    # adj_t = torch_sparse.mul(deg_inv.view(-1, 1), adj_t) doesn't work!
    adj_t = torch_sparse.mul(adj_t, deg_inv.view(-1, 1))

    return adj_t

# A_cs - Column-Stochastic Matrix
def col_stochastic_matrix(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    adj_t = edge_index

    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1., dtype=dtype)
    if add_self_loops:
        adj_t = torch_sparse.fill_diag(adj_t, 1)

    deg = torch_sparse.sum(adj_t, dim=1)
    deg_inv = deg.pow_(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
    adj_t = torch_sparse.mul(adj_t, deg_inv.view(1, -1))

    return adj_t

def check_data_leakage(pos_edge_idx, neg_edge_idx):
    leakage = False

    pos_edge_idx_set = set(map(tuple, pos_edge_idx.t().tolist()))
    neg_edge_idx_set = set(map(tuple, neg_edge_idx.t().tolist()))

    if pos_edge_idx_set & neg_edge_idx_set:
        leakage = True
        print("Data leakage found between positive and negative samples.")
        raise Exception("Data leakage detected.")
    
    if not leakage:
        print("No data leakage found.")

def do_edge_split(data, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
        check_data_leakage(data.train_pos_edge_index, data.train_neg_edge_index)
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]
        check_data_leakage(data.train_pos_edge_index, data.train_neg_edge_index)
        check_data_leakage(data.val_pos_edge_index, data.val_neg_edge_index)
        check_data_leakage(data.test_pos_edge_index, data.test_neg_edge_index)
    
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge

def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0))

