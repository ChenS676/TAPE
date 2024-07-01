import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization

from typing import Dict
import numpy as np
import scipy.sparse as ssp
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit
from heuristic.lsf import CN, AA, RA, InverseRA
from heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close, SymPPR
from textfeat.mlp_dot_product import pairwise_prediction
import matplotlib.pyplot as plt
from core.graphgps.visualization.adj import plot_coo_matrix, construct_sparse_adj
from core.graphgps.utility.utils import get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from heuristic.eval import get_metric_score

FILE_PATH = f'{get_git_repo_root_path()}/'


def eval_cora_mrr() -> None:
    """load text attribute graph in link predicton setting
    """

    dataset, data_cited, splits = get_cora_casestudy(undirected = True,
                                                include_negatives = True,
                                                val_pct = 0.15,
                                                test_pct = 0.05,
                                                split_labels = True)

    # ust test edge_index as full_A
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes

    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, 'test_edge_index.png')

    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

    # only for debug
    pos_test_index = splits['test'].pos_edge_label_index
    neg_test_index = splits['test'].neg_edge_label_index

    pos_m = construct_sparse_adj(pos_test_index)
    plot_coo_matrix(pos_m, f'test_pos_index.png')
    neg_m = construct_sparse_adj(neg_test_index)
    plot_coo_matrix(neg_m, f'test_neg_index.png')

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    result_dict = {}
    for use_heuristic in ['CN', 'AA', 'RA', 'InverseRA']:
        pos_test_pred, _ = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred, _ = eval(use_heuristic)(full_A, neg_test_index)

    #     result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
    #     result_dict.update({f'{use_heuristic}': result})

    # # 'shortest_path', 'katz_apro', 'katz_close', 'Ben_PPR'
    # for use_heuristic in ['Ben_PPR', 'SymPPR']:
    #     pos_test_pred, _ = eval(use_heuristic)(full_A, pos_test_index)
    #     neg_test_pred, _ = eval(use_heuristic)(full_A, neg_test_index)
    #     result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
    #     result_dict.update({f'{use_heuristic}': result})

    # for use_heuristic in ['shortest_path', 'katz_apro', 'katz_close']:
    #     pos_test_pred = eval(use_heuristic)(full_A, pos_test_index)
    #     neg_test_pred = eval(use_heuristic)(full_A, neg_test_index)
    #     result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)

    #     # calc mrr and hits@k
    #     result_dict.update({f'{use_heuristic}': result})

    for use_heuristic in ['pairwise_pred']:
        for dist in ['dot']:
            pos_test_pred = pairwise_prediction(dataset._data.x, pos_test_index, dist)
            neg_test_pred = pairwise_prediction(dataset._data.x, neg_test_index, dist)
            result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
            result_dict.update({f'{use_heuristic}_{dist}': result})

    return result_dict


def eval_cora_acc() -> None:
    
    dataset, data_cited, splits = get_cora_casestudy(undirected = True,
                                                include_negatives = True,
                                                val_pct = 0.15,
                                                test_pct = 0.05,
                                                split_labels = False
                                                )

    labels = splits['test'].edge_label
    test_index = splits['test'].edge_label_index
    
    test_edge_index = splits['test'].edge_index
    edge_weight = torch.ones(test_edge_index.size(1))
    num_nodes = dataset._data.num_nodes

    m = construct_sparse_adj(test_edge_index)
    plot_coo_matrix(m, f'cora_test_edge_index.png')
    
    A = ssp.csr_matrix((edge_weight.view(-1), (test_edge_index[0], test_edge_index[1])), shape=(num_nodes, num_nodes)) 

    result_acc = {}
    for use_lsf in ['CN', 'AA', 'RA', 'InverseRA']:
        scores, edge_index = eval(use_lsf)(A, test_index)
        
        plt.figure()
        plt.plot(scores)
        plt.plot(labels)
        plt.savefig(f'{use_lsf}.png')
        
        acc = torch.sum(scores == labels)/scores.shape[0]
        result_acc.update({f"{use_lsf}_acc" :acc})
        
            
    # 'shortest_path', 'katz_apro', 'katz_close', 'Ben_PPR'
    for use_gsf in ['Ben_PPR', 'SymPPR']:
        scores, edge_reindex = eval(use_gsf)(A, test_index)
        
        pred = torch.zeros(scores.shape)
        cutoff = 0.05
        thres = scores.max()*cutoff 
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        result_acc.update({f"{use_gsf}_acc" :acc})
    
    for use_gsf in ['shortest_path', 'katz_apro', 'katz_close']:
        scores = eval(use_gsf)(A, test_index)
        
        pred = torch.zeros(scores.shape)
        thres = scores.min()*10
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        print(f" {use_gsf}: acc: {acc}")

        result_acc.update({f"{use_gsf}_acc" :acc})

    for use_heuristic in ['pairwise_pred']:
        for dist in ['dot']:
            scores = pairwise_prediction(dataset._data.x, test_index, dist)
            test_pred = torch.zeros(scores.shape)
            cutoff = 0.25
            thres = scores.max()*cutoff 
            test_pred[scores <= thres] = 0
            test_pred[scores > thres] = 1
            acc = torch.sum(test_pred == labels)/labels.shape[0]
            
            plt.figure()
            plt.plot(test_pred)
            plt.plot(labels)
            plt.savefig(f'{use_heuristic}.png')
        
        result_acc.update({f"{use_heuristic}_acc" :acc})
        
    return result_acc
        
        
def parse_cora():
    # load original data from cora orig without text features
    path = f'{FILE_PATH}dataset/cora_orig/cora'
    idx_features_labels = np.genfromtxt(f"{path}.content", dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_cora_casestudy(undirected = True,
                        include_negatives = True,
                        val_pct = 0.15,
                        test_pct = 0.05,
                        split_labels = False
                        ) -> InMemoryDataset:
    # undirected = args.data.undirected
    # include_negatives = args.data.include_negatives
    # val_pct = args.data.val_pct
    # test_pct = args.data.test_pct
    # split_labels = args.data.split_labels
    
    data_X, data_Y, data_citeid, data_edges = parse_cora()

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('./dataset', data_name,
                        transform=T.NormalizeFeatures())

    data = dataset[0]
    # check is data has changed and try to return dataset
    x = torch.tensor(data_X).float()
    edge_index = torch.LongTensor(data_edges).long()
    y = torch.tensor(data_Y).long()
    num_nodes = len(data_Y)

    data = Data(x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
        node_attrs=x, 
        edge_attrs = None, 
        graph_attrs = None
    )        
    dataset._data = data

    undirected = data.is_undirected()

    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives, split_labels=split_labels)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}

    return dataset, data_citeid, splits


# main function 
if __name__ == "__main__":
    NAME = 'cora'
    result_acc = eval_cora_acc()
    result_mrr = eval_cora_mrr()

    for key, val in result_mrr.items():
        print(key, val)
    for key, val in result_acc.items():
        print(key, val)    

    root = f'{FILE_PATH}results'
    acc_file = f'{root}/{NAME}_acc.csv'
    mrr_file = f'{root}/{NAME}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    append_acc_to_excel(id, result_acc, acc_file, NAME, method='')
    append_mrr_to_excel(id, result_mrr, mrr_file, NAME, method='')
    