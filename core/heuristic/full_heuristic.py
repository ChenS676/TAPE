import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization

import csv
import time
import torch
import numpy as np
import scipy.sparse as ssp
import matplotlib.pyplot as plt
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops

from data_utils.load import load_graph_lp
from heuristic.eval import get_metric_score
from heuristic.lsf import CN, AA, RA, InverseRA
from heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close, SymPPR
from data_utils.load_data_lp import get_edge_split
from data_utils.lcc import find_scc_direc, use_lcc_direc, use_lcc
from data_utils.graph_stats import plot_coo_matrix, construct_sparse_adj
from graphgps.utility.utils import get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel

FILE_PATH = f'{get_git_repo_root_path()}/'
NAME = 'pubmed'

def eval_cora_mrr(data, splits) -> dict:
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = data.num_nodes

    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, 'test_edge_index.png')

    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

    pos_test_index = splits['test'].pos_edge_label_index
    neg_test_index = splits['test'].neg_edge_label_index

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    result_dict = {}
    time_dict = {}
    # 'CN', 'AA', 'RA', 'katz_apro', 'katz_close', 'Ben_PPR', 'SymPPR'
    for use_heuristic in ['CN', 'AA', 'RA', 'katz_apro', 'katz_close', 'Ben_PPR', 'SymPPR']:
        start = time.time()
        pos_test_pred, _ = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred, _ = eval(use_heuristic)(full_A, neg_test_index)
        end = time.time()
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
        result_dict.update({f'{use_heuristic}': result})
        time_dict.update({f'{use_heuristic}': end - start})

    return result_dict, time_dict


def eval_cora_acc(data, splits) -> dict:
    labels = splits['test'].edge_label
    test_index = splits['test'].edge_label_index
    test_edge_index = splits['test'].edge_index
    edge_weight = torch.ones(test_edge_index.size(1))
    num_nodes = data.num_nodes

    m = construct_sparse_adj(test_edge_index)
    plot_coo_matrix(m, f'cora_test_edge_index.png')
    A = ssp.csr_matrix((edge_weight.view(-1), (test_edge_index[0], test_edge_index[1])), shape=(num_nodes, num_nodes)) 

    result_acc = {}
    # 'CN', 'AA', 'RA', 'katz_apro', 'katz_close', 'Ben_PPR', 'SymPPR'
    for use_heuristic in ['CN', 'AA', 'RA', 'katz_apro', 'katz_close', 'Ben_PPR', 'SymPPR']:
        scores, edge_index = eval(use_heuristic)(A, test_index)
        
        plt.figure()
        plt.plot(scores)
        plt.plot(labels)
        plt.savefig(f'{use_heuristic}.png')
        
        acc = torch.sum(scores == labels)/scores.shape[0]
        result_acc.update({f"{use_heuristic}_acc" :acc})
        
    return result_acc


def run_evaluation(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if NAME == 'cora':
        data, _ = load_graph_lp[NAME](False)
    elif NAME == 'pubmed':
        data = load_graph_lp[NAME](False)
    elif NAME == 'arxiv_2023':
        data = load_graph_lp[NAME]()

    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data, lcc, _ = use_lcc(data)

    splits = get_edge_split(data, True, 'cpu', 0.15, 0.05, True, False)
    result_acc = eval_cora_acc(data, splits)
    splits = get_edge_split(data, True, 'cpu', 0.15, 0.05, True, True)
    result_mrr, result_time = eval_cora_mrr(data, splits)

    return result_acc, result_mrr, result_time

def aggregate_results(results_list):
    aggregated = {}
    for key in results_list[0]:
        if isinstance(results_list[0][key], dict):
            aggregated[key] = {sub_key: [] for sub_key in results_list[0][key]}
            for result in results_list:
                for sub_key, value in result[key].items():
                    aggregated[key][sub_key].append(value)
        else:
            aggregated[key] = []
            for result in results_list:
                aggregated[key].append(result[key])
    
    return aggregated

def calculate_means_variances(aggregated_results):
    means = {}
    variances = {}
    for key, values in aggregated_results.items():
        if isinstance(values, dict):
            means[key] = {}
            variances[key] = {}
            for sub_key, sub_values in values.items():
                means[key][sub_key] = np.mean(sub_values)
                print(sub_values)
                variances[key][sub_key] = np.var(sub_values, ddof=1)
        else:
            means[key] = np.mean(values)
            variances[key] = np.var(values, ddof=1)
    return means, variances

if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]
    
    all_results_acc = []
    all_results_mrr = []
    all_time = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        result_acc, result_mrr, result_time = run_evaluation(seed)
        all_results_acc.append(result_acc)
        all_results_mrr.append(result_mrr)
        all_time.append(result_time)
        
        root = f'{FILE_PATH}results/heuristic/{NAME}'
        acc_file = f'{root}/{NAME}_acc.csv'
        mrr_file = f'{root}/{NAME}_mrr.csv'
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        
        run_id = f'{NAME}_{seed}'
        append_acc_to_excel(run_id, result_acc, acc_file, NAME, method='')
        append_mrr_to_excel(run_id, result_mrr, mrr_file, NAME, method='')
    
    # Aggregate results across seeds
    aggregated_acc = aggregate_results(all_results_acc)
    aggregated_mrr = aggregate_results(all_results_mrr)
    aggregated_time = aggregate_results(all_time)
    
    # Calculate means and variances
    means_acc, variances_acc = calculate_means_variances(aggregated_acc)
    means_mrr, variances_mrr = calculate_means_variances(aggregated_mrr)
    means_time, _ = calculate_means_variances(aggregated_time)
    
    root = os.path.join(FILE_PATH, f'results/heuristic/{NAME}')
    summary_file = os.path.join(root, f'{NAME}_summary_results.csv')
    
    with open(summary_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Time', 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@20', 'Hits@50', 'Hits@100', 'MRR', 
                        'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 'mrr_hit20', 'mrr_hit50', 'mrr_hit100', 'AUC', 'AP', 'ACC'])
        
        keys = ["Hits@1", "Hits@3", "Hits@10", "Hits@20", "Hits@50", "Hits@100", "MRR", 
                "mrr_hit1", "mrr_hit3", "mrr_hit10", "mrr_hit20", "mrr_hit50", "mrr_hit100", 
                "AUC", "AP", "ACC"]

        for alg in ['CN', 'AA', 'RA', 'katz_apro', 'katz_close', 'Ben_PPR', 'SymPPR']:
            row = [f"{alg}"] + [f'{means_time[f"{alg}"]:.6f}'] + [f'{means_mrr[alg][key] * 100:.2f} ± {variances_mrr[alg][key] * 100:.2f}' for key in keys if key != 'ACC'] + [f'{means_acc[f"{alg}_acc"] * 100:.2f} ± {variances_acc[f"{alg}_acc"] * 100:.2f}']

            writer.writerow(row)
