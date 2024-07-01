# TODO: Not tested error 
import torch_geometric.transforms as T
import torch
import pandas as pd
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_geometric.transforms import RandomLinkSplit
from data_utils.dataset import CustomPygDataset, CustomLinkDataset
from heuristic.lsf import CN, AA, RA, InverseRA
from heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close , SymPPR
import matplotlib.pyplot as plt
from core.graphgps.visualization.adj import construct_sparse_adj, plot_coo_matrix, plot_pos_neg_adj
import scipy.sparse as ssp

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from eval import evaluate_auc, evaluate_hits, evaluate_mrr, get_metric_score, get_prediction
FILE_PATH = get_git_repo_root_path() + '/'
from core.graphgps.utility.utils import get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel


def get_raw_text_products(use_text=False, seed=0,
                            undirected = True,
                            include_negatives = True,
                            val_pct = 0.15,
                            test_pct = 0.05,
                            split_labels=True):
    
    root_path = get_git_repo_root_path()
    data = torch.load(root_path + '/dataset/ogbn_products_orig/ogbn-products_subset.pt')
    text = pd.read_csv(root_path + '/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric().to_torch_sparse_coo_tensor().coalesce().indices()
    
    if not use_text:
        text = None

    dataset = CustomLinkDataset('./dataset', 'ogbn-products', transform=T.NormalizeFeatures())
    dataset._data = data
    del dataset._data.n_id, dataset._data.adj_t, dataset._data.e_id

    undirected = data.is_undirected()
    undirected = True
    include_negatives = True
    val_pct = 0.15
    test_pct = 0.05
    
    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives, 
                                split_labels=split_labels)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    
    return dataset, text, splits



def eval_ogbn_products_acc(name='ogbn_products'):
    dataset, text, splits = get_raw_text_products(use_text=False, seed=0,
                                                  undirected = True,
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
    for use_heuristic in ['CN', 'AA', 'RA']:
        scores, edge_index = eval(use_heuristic)(A, test_index)
        
        plt.figure()
        plt.plot(scores)
        plt.plot(labels)
        plt.savefig(f'{name}_{use_heuristic}.png')
        
        acc = torch.sum(scores == labels)/scores.shape[0]
        result_acc.update({f"{name}_{use_heuristic}_acc" :acc})
        
    for use_heuristic in ['Ben_PPR']:
        scores, edge_reindex = eval(use_heuristic)(A, test_index)
        
        # print(scores)
        # print(f" {use_heuristic}: accuracy: {scores}")
        pred = torch.zeros(scores.shape)
        cutoff = 0.05
        thres = scores.max()*cutoff 
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        result_acc.update({f"{name}_{use_heuristic}_acc" :acc})
    
    # , 'katz_close'
    for use_heuristic in ['shortest_path', 'katz_apro']:
        scores = eval(use_heuristic)(A, test_index)
        
        pred = torch.zeros(scores.shape)
        thres = scores.min()*10
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        result_acc.update({f"{name}_{use_heuristic}_acc" :acc})
    
    
    return result_acc


def eval_heuristic_mrr_hits(name='ogbn_products'):
    """eval heuristic using mrr and hits from ogb.evaluator"""

    dataset, text, splits = get_raw_text_products(undirected = True,
                                                include_negatives = True,
                                                val_pct = 0.15,
                                                test_pct = 0.05,
                                                split_labels=True)
    
    # ust test edge_index as full_A
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, f'{name}_full_cedge_index.png')
    
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

    # only for debug
    pos_test_index = splits['test'].pos_edge_label_index
    neg_test_index = splits['test'].neg_edge_label_index
    
    pos_m = construct_sparse_adj(pos_test_index)
    plot_coo_matrix(pos_m, f'{name}_pos_index.png')
    neg_m = construct_sparse_adj(neg_test_index)
    plot_coo_matrix(neg_m, f'{name}_neg_index.png')
    
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

    return result_dict


if __name__ == "__main__":

    name = 'ogbn-products'
    result_acc = eval_ogbn_products_acc(name)
    print(result_acc)
    result_mrr = eval_heuristic_mrr_hits(name)
    print(result_mrr)
        
    root = FILE_PATH + 'results'
    acc_file = root + f'/{name}_acc.csv'
    mrr_file = root + f'/{name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    
    append_acc_to_excel(result_acc, acc_file, name)
    append_mrr_to_excel(result_mrr, mrr_file) 