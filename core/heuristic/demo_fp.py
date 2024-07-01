# This script evaluate the performance of node feature proximity 
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from collections import Counter
import matplotlib.pyplot  as plt 
import sys
import numpy as np


import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from core.graphgps.utility.utils import get_root_dir
from eval import evaluate_hits, evaluate_mrr, evaluate_auc
import pandas as pd 
from math import inf
import seaborn as sns
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

dir_path = get_root_dir()

def get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred, 
                     pos_val_pred, neg_val_pred):
   
    result = {}
    k_list = [1, 3, 10, 100]
   
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred)
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred)
    result_hit_test = evaluate_mrr(evaluator_hit, pos_test_pred, neg_test_pred)
    result_hit_val = evaluate_mrr(evaluator_hit, pos_val_pred, neg_val_pred)

    print(result_hit_test)
    print(result_hit_val)
    print(result_mrr_test)
    print(result_mrr_val)
    return 

def read_data(data_name, dir_path, filename):
    data_name = data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:
        dirpath = '/pfs/work7/workspace/scratch/cc7738-nlp_graph/HeaRT_Mao'
        path = dirpath + '/dataset/{}/{}_pos.txt'.format(data_name, split)

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                
            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    with open(f'{dir_path}/{data_name}/heart_valid_{filename}', "rb") as f:
        valid_neg = np.load(f)
        valid_neg = torch.from_numpy(valid_neg)
    with open(f'{dir_path}/{data_name}/heart_test_{filename}', "rb") as f:
        test_neg = np.load(f)
        test_neg = torch.from_numpy(test_neg)

    train_pos_tensor =  torch.transpose(torch.tensor(train_pos), 1, 0)

    valid_pos =  torch.transpose(torch.tensor(valid_pos), 1, 0)
    test_pos =  torch.transpose(torch.tensor(test_pos), 1, 0)
    
    valid_neg =  torch.tensor(valid_neg)
    test_neg =  torch.tensor(test_neg)

    valid_neg = torch.permute(valid_neg, (2, 0, 1))
    valid_neg = valid_neg.view(2,-1)

    test_neg = torch.permute(test_neg, (2, 0, 1))
    test_neg = test_neg.view(2,-1)
    
    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
          
    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    
    test_pos =  torch.tensor(test_pos)
    

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    feature_embeddings = torch.load(dir_path+ '/{}/{}'.format(data_name, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {}
    data['adj'] = adj
    data['train_pos'] = train_pos_tensor
    data['train_val'] = train_val

    data['valid_pos'] = valid_pos
    data['valid_neg'] = valid_neg
    data['test_pos'] = test_pos
    data['test_neg'] = test_neg

    data['x'] = feature_embeddings
    
    return  A, train_pos_tensor, valid_pos, test_pos, valid_neg, test_neg, train_pos, feature_embeddings


def get_hist(A, full_A, use_heuristic, data, num_nodes):

    # Assuming you have a tensor with node indices
    nodes = torch.arange(num_nodes)
    # Generate pairwise combinations of indices
    pairs = torch.combinations(nodes, r=2).T

    pos_test_pred = eval(use_heuristic)(A, pairs)
    
    data_df = pd.DataFrame({'size': pos_test_pred.numpy()})

    data_df_filtered = data_df[data_df['size'] != 0.0]
    
    plt.figure()
    bin_edges = [0, 1, 3, 10, 25, float('inf')]
    ax = sns.histplot(data=data_df_filtered, kde=False, stat='percent', discrete=True, 
                      color='blue')
    
    # Access the bin edges and heights
    bin_edges = ax.patches[0].get_x()*ax.patches[-1].get_x() + ax.patches[-1].get_width()
    heights = [patch.get_height() for patch in ax.patches]

    # Print bin edges and heights
    print("Bin Edges:", bin_edges)
    print("Heights:", heights)

    plt.title(f'{data}_{use_heuristic}_filtered')
    plt.xlim(1, 40) 
    plt.xlabel('Num of CN')  # Specify x-axis label
    plt.ylabel('Propotion')   # Specify y-axis label
    plt.savefig(f'{data}_{use_heuristic}_filtered.png')
    return 

def get_test_hist(A, test_pos, test_neg, use_heuristic, data, num_nodes):
    
    pos_test_pred = eval(use_heuristic)(A, test_pos)
    neg_test_pred = eval(use_heuristic)(A, test_neg)
    
    bin_edges = [0, 1, 3, 10, 25, float('inf')]
    
    pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    hist, bin_edges = np.histogram(pred, bins=bin_edges)
    
    hist = hist / hist.sum()
    plt.figure(figsize=(10, 8))
    plt.bar([1, 2, 3, 4, 5], hist)
    plt.title(f'{data}_{use_heuristic}_filtered')
    plt.xlabel('Num of CN')  
    plt.ylabel('Proportion')  
    dirpath = '/pfs/work7/workspace/scratch/cc7738-nlp_graph/HeaRT_Mao/benchmarking/HeaRT_small'
    plt.savefig(f'{dirpath}/{data}_{use_heuristic}_test_filtered.png')
    

    plt.figure(figsize=(10, 8))
    sns.barplot(x=[1, 2, 3, 4, 5], y=hist, color='skyblue')
    plt.xticks([0, 1, 2, 3, 4], ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('CN distribution', fontsize=24)
    plt.xlabel('Num of CN', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.savefig(f'{dirpath}/sns{data}_{use_heuristic}_test_filtered.png')
    # data_df = pd.DataFrame({'size': pred.numpy()})
    
    # plt.figure()
    # bin_edges = [0, 1, 3, 10, 25, float('inf')]
    # ax = sns.histplot(data=data_df, kde=False, stat='percent', discrete=True, binrange=(0, 40), 
    #                   color='blue')
    
    # # Access the bin edges and heights
    # bin_edges = ax.patches[0].get_x()*ax.patches[-1].get_x() + ax.patches[-1].get_width()
    # heights = [patch.get_height() for patch in ax.patches]

    # # Print bin edges and heights
    # print("Bin Edges:", bin_edges)
    # print("Heights:", heights)

    # plt.title(f'{data}_{use_heuristic}_filtered')
    # plt.xlabel('Num of CN')  # Specify x-axis label
    # plt.ylabel('Propotion')   # Specify y-axis label
    # plt.savefig(f'{data}_{use_heuristic}_test_filtered.png')

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance


def get_fp_prediction(data, test_pos, test_neg, args):
    test_pos, test_neg = test_pos.numpy().transpose(), test_neg.numpy().transpose()
    test_pos_pred, test_neg_pred = [], []

    distance_metric = {
        'cos': distance.cosine,
        'l2': distance.euclidean,
        'hamming': distance.hamming,
        'jaccard': distance.jaccard,
        'dice': distance.dice,
        'dot': lambda x, y: np.dot(x, y)
    }

    if args.distance not in distance_metric:
        raise ValueError("Invalid distance metric specified.")

    metric_function = distance_metric[args.distance]

    for ind in test_pos:
        metric_value = metric_function(data[ind[0]], data[ind[1]])
        test_pos_pred.append(metric_value)

    for ind_n in test_neg:
        metric_value = metric_function(data[ind_n[0]], data[ind_n[1]])
        test_neg_pred.append(metric_value)

    test_pos_pred, test_neg_pred = torch.tensor(np.asarray(test_pos_pred)), torch.tensor(np.asarray(test_neg_pred))
    return test_pos_pred, test_neg_pred




def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='pubmed')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--use_heuristic', type=str, default='FP')
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--distance', type=str, default='dot')
    parser.add_argument('--beta', type=float, default='0.005')

    args = parser.parse_args()

    # dataset = Planetoid('.', 'cora')

    A, train_pos, valid_pos, test_pos, valid_neg, test_neg, train_pos_list, data_embeddings  = read_data(args.data_name, args.input_dir, args.filename)
    
    node_num = A.shape[0]

    if args.use_valedges_as_input:
        print('use validation!!!')
        val_edge_index = valid_pos
        val_edge_index = to_undirected(val_edge_index)

        edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1)], dtype=int)

        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        

        full_A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                        shape=(node_num, node_num)) 
        print('nonzero values: ', full_A.nnz)
    else:
        full_A = A
        print('no validation!!!')
        print('nonzero values: ', full_A.nnz)

    use_heuristic = args.use_heuristic

    if args.use_heuristic != 'FP':
        get_test_hist(A, test_pos, test_neg, use_heuristic, args.data_name, node_num)
    else:
        test_pos_pred, test_neg_pred = get_fp_prediction(data_embeddings, test_pos, test_neg, args)
        valid_pos_pred, valid_neg_pred = get_fp_prediction(data_embeddings, valid_pos, valid_neg, args)
        evaluator_hit = Evaluator(name='ogbl-collab')
        evaluator_mrr = Evaluator(name='ogbl-citation2')

        results = get_metric_score(evaluator_hit, evaluator_mrr, test_pos_pred, test_neg_pred, valid_pos_pred, valid_neg_pred)
        print('heurisitic: ', args.use_heuristic)  


if __name__ == "__main__":
    main()