
import torch
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils.dataset import CustomPygDataset, CustomLinkDataset
from heuristic.lsf import CN, AA, RA, InverseRA
from heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close , SymPPR
from data_utils.load_data_lp import get_raw_text_pubmed, get_pubmed_lp
import matplotlib.pyplot as plt
from core.graphgps.visualization.adj import construct_sparse_adj, plot_coo_matrix
import scipy.sparse as ssp
from core.graphgps.utility.utils import get_git_repo_root_path
from typing import Dict
FILE_PATH = get_git_repo_root_path() + '/'


import torch
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from heuristic.eval import evaluate_auc, evaluate_hits, evaluate_mrr, get_metric_score, get_prediction
from core.graphgps.utility.utils import get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel
from textfeat.mlp_dot_product import pairwise_prediction
from data_utils.load_data_lp import get_raw_text_arxiv_2023_lp

def eval_arxiv_23_acc() -> Dict:
    dataset, _, splits = get_raw_text_arxiv_2023_lp(undirected = True,
                                                include_negatives = True,
                                                val_pct = 0.15,
                                                test_pct = 0.05,
                                                split_labels=False)
    print(dataset._data)
    
    test_split = splits['test']
    labels = test_split.edge_label
    test_index = test_split.edge_label_index
    
    edge_index = splits['test'].edge_index
    edge_weight = torch.ones(edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    result_acc = {}
    for use_lsf in ['CN', 'AA', 'RA']:
        scores, edge_index = eval(use_lsf)(A, test_index)
        
        plt.figure()
        plt.plot(scores)
        plt.plot(labels)
        plt.savefig(f'arxiv_23{use_lsf}.png')
        
        acc = torch.sum(scores == labels)/scores.shape[0]
        result_acc.update({f"{use_lsf}_acc" :acc})
        
    # for use_gsf in ['Ben_PPR']:
    #     scores, edge_reindex = eval(use_gsf)(A, test_index)
        
    #     # print(scores)
    #     # print(f" {use_heuristic}: accuracy: {scores}")
    #     pred = torch.zeros(scores.shape)
    #     cutoff = 0.05
    #     thres = scores.max()*cutoff 
    #     pred[scores <= thres] = 0
    #     pred[scores > thres] = 1
        
    #     acc = torch.sum(pred == labels)/labels.shape[0]
    #     result_acc.update({f"{use_gsf}_acc" :acc})
    
    # # , 'katz_close'
    # for use_gsf in ['shortest_path', 'katz_apro']:
    #     scores = eval(use_gsf)(A, test_index)
        
    #     pred = torch.zeros(scores.shape)
    #     thres = scores.min()*10
    #     pred[scores <= thres] = 0
    #     pred[scores > thres] = 1
        
    #     acc = torch.sum(pred == labels)/labels.shape[0]
    #     result_acc.update({f"{use_gsf}_acc" :acc})

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

def eval_cora_mrr() -> None:
    """load text attribute graph in link predicton setting

    """

    dataset, text, splits = get_raw_text_arxiv_2023(undirected = True,
                                                include_negatives = True,
                                                val_pct = 0.15,
                                                test_pct = 0.05,
                                                split_labels=True)
    
    # ust test edge_index as full_A
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')
    
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
    # , 'InverseRA'
    for use_heuristic in ['CN', 'AA', 'RA']:
        pos_test_pred, _ = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred, _ = eval(use_heuristic)(full_A, neg_test_index)
        
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
        result_dict.update({f'{use_heuristic}': result})
        
    # , 'SymPPR'
    for use_heuristic in ['Ben_PPR']:
        pos_test_pred, _ = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred, _ = eval(use_heuristic)(full_A, neg_test_index)
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
        result_dict.update({f'{use_heuristic}': result})
    
    #  'katz_close'
    for use_heuristic in ['shortest_path', 'katz_apro']:
        pos_test_pred = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred = eval(use_heuristic)(full_A, neg_test_index)
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)

        # calc mrr and hits@k
        result_dict.update({f'{use_heuristic}': result})

    for use_heuristic in ['pairwise_pred']:
        for dist in ['dot']:
            pos_test_pred = pairwise_prediction(dataset._data.x, pos_test_index, dist)
            neg_test_pred = pairwise_prediction(dataset._data.x, neg_test_index, dist)
            result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
            result_dict.update({f'{use_heuristic}_{dist}': result})
            
    return result_dict

if __name__ == "__main__":
    name = 'arxiv_2023'
    result_acc = eval_arxiv_23_acc()
    print(result_acc)
    result_mrr = eval_cora_mrr()
    print(result_mrr)
        
    root = FILE_PATH + 'results'
    acc_file = root + f'/{name}_acc.csv'
    mrr_file = root + f'/{name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    
    import wandb
    id = wandb.util.generate_id() 
    append_acc_to_excel(id, result_acc, acc_file, name, method='')
    append_mrr_to_excel(id, result_mrr, mrr_file, name, method='')
    