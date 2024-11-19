# -*- coding: utf-8 -*-
import torch
import numpy as np
from HLGNN.OGB.negative_sample import global_neg_sample, global_perm_neg_sample, local_neg_sample
from torch_geometric.utils import (negative_sampling, add_self_loops, train_test_split_edges)
import random
import math 
import re
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def get_pos_neg_edges(split, split_edge, edge_index=None, num_nodes=None, neg_sampler_name=None, num_neg=None):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge']
    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        pos_edge = torch.stack([source, target]).t()

    if split == 'train':
        if neg_sampler_name == 'local':
            neg_edge = local_neg_sample(
                pos_edge,
                num_nodes=num_nodes,
                num_neg=num_neg)
        elif neg_sampler_name == 'global':            
            neg_edge = global_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
        else:
            neg_edge = global_perm_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
    else:
        if 'edge' in split_edge['train']:
            neg_edge = split_edge[split]['edge_neg']
        elif 'source_node' in split_edge['train']:
            target_neg = split_edge[split]['target_node_neg']
            neg_per_target = target_neg.size(1)
            neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                    target_neg.view(-1)]).t()
    return pos_edge, neg_edge

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


def evaluate_hits(evaluator, pos_val_pred, neg_val_pred,
                  pos_test_pred, neg_test_pred):
    results = {}
    for K in [1, 3, 10, 20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def eval_mrr(y_pred_pos, y_pred_neg):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''

    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)

    hits20_list = (ranking_list <= 20).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)

    return { 'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@20_list': hits20_list,
                'hits@50_list': hits50_list,
                'hits@10_list': hits10_list,
                'hits@100_list': hits100_list,
                'mrr_list': mrr_list}

def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred,
                 pos_test_pred, neg_test_pred):
    # print(f"Shape of pos_val_pred: {pos_val_pred.shape}")
    # print(f"Shape of neg_val_pred: {neg_val_pred.shape}")

    # neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    # neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    
    mrr_output = eval_mrr(pos_val_pred, neg_val_pred)
    # test_mrr = eval_mrr(pos_test_pred, neg_test_pred)
    
    mrr=mrr_output['mrr_list'].mean().item()
    mrr_hit1 = mrr_output['hits@1_list'].mean().item()
    mrr_hit3 = mrr_output['hits@3_list'].mean().item()
    mrr_hit10 = mrr_output['hits@10_list'].mean().item()

    mrr_hit20 = mrr_output['hits@20_list'].mean().item()
    mrr_hit50 = mrr_output['hits@50_list'].mean().item()
    mrr_hit100 = mrr_output['hits@100_list'].mean().item()


    valid_mrr = round(mrr, 4)
    valid_mrr_hit1 = round(mrr_hit1, 4)
    valid_mrr_hit3 = round(mrr_hit3, 4)
    valid_mrr_hit10 = round(mrr_hit10, 4)

    valid_mrr_hit20 = round(mrr_hit20, 4)
    valid_mrr_hit50 = round(mrr_hit50, 4)
    valid_mrr_hit100 = round(mrr_hit100, 4)
    
    results = {}
    results['mrr_hit1'] = valid_mrr_hit1
    results['mrr_hit3'] = valid_mrr_hit3
    results['mrr_hit10'] = valid_mrr_hit10

    results['MRR'] = valid_mrr

    results['mrr_hit20'] = valid_mrr_hit20
    results['mrr_hit50'] = valid_mrr_hit50
    results['mrr_hit100'] = valid_mrr_hit100

    return results

def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    # test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    
    valid_auc = round(valid_auc, 4)
    # test_auc = round(test_auc, 4)

    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    # test_ap = average_precision_score(test_true, test_pred)
    
    valid_ap = round(valid_ap, 4)
    # test_ap = round(test_ap, 4)
    
    results['AP'] = valid_ap

    return results

def acc(pos_pred, neg_pred):
        hard_thres = (max(torch.max(pos_pred).item(), torch.max(neg_pred).item()) + min(torch.min(pos_pred).item(), torch.min(neg_pred).item())) / 2

        # Initialize predictions with zeros and set ones where condition is met
        y_pred = torch.zeros_like(pos_pred)
        y_pred[pos_pred >= hard_thres] = 1

        # Do the same for negative predictions
        neg_y_pred = torch.ones_like(neg_pred)
        neg_y_pred[neg_pred <= hard_thres] = 0

        # Concatenate the positive and negative predictions
        y_pred = torch.cat([y_pred, neg_y_pred], dim=0)

        # Initialize ground truth labels
        pos_y = torch.ones_like(pos_pred)
        neg_y = torch.zeros_like(neg_pred)
        y = torch.cat([pos_y, neg_y], dim=0)
        y_logits = torch.cat([pos_pred, neg_pred], dim=0)
        # Calculate accuracy    
        return (y == y_pred).float().mean().item()
    
def gcn_normalization(adj_t):
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def adj_normalization(adj_t):
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t
    return adj_t


def generate_neg_dist_table(num_nodes, adj_t, power=0.75, table_size=1e8):
    table_size = int(table_size)
    adj_t = adj_t.set_diag()
    node_degree = adj_t.sum(dim=1).to(torch.float)
    node_degree = node_degree.pow(power)

    norm = float((node_degree).sum())  # float is faster than tensor when visited
    node_degree = node_degree.tolist()  # list has fastest visit speed
    sample_table = np.zeros(table_size, dtype=np.int32)
    p = 0
    i = 0
    for j in range(num_nodes):
        p += node_degree[j] / norm
        while i < table_size and float(i) / float(table_size) < p:
            sample_table[i] = j
            i += 1
    sample_table = torch.from_numpy(sample_table)
    return sample_table

def extract_final_test_results(block):
    pattern = r'Final Test:\s*([\d.]+)\s*±\s*([\d.]+)'
    match = re.search(pattern, block)
    if match:
        return match.groups()
    return None

def read_blocks_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    blocks = content.strip().split('\n\n')
    return blocks

def extract_dataset_name(block, emb_name):
    pattern = r'Dataset:\s*(\w+)'
    match = re.search(pattern, block)
    if match:
        dataset_name = match.group(1) + '_' + emb_name
        return dataset_name
    return "Unknown"

def save_to_csv(results, filename='HLGNN/OGB/metrics_and_weights/final_test_results.csv'):
    header = ['Dataset', 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@20', 'Hits@50', 'Hits@100', 'MRR', 
              'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 'mrr_hit20', 'mrr_hit50', 'mrr_hit100', 'AUC', 
              'AP', 'ACC']
    
    # existing_rows = set()
    # with open(filename, mode='r', newline='') as file:
    #     reader = csv.reader(file)
    #     existing_rows = {tuple(row) for row in reader}
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # if not existing_rows:
        writer.writerow(header)
        
        for result in results:
            # if tuple(result) not in existing_rows:
            writer.writerow(result)

def process_blocks(blocks, name_emb):
    final_test_results = []
    for i in range(0, len(blocks), 16):
        dataset_name = extract_dataset_name(blocks[i], name_emb)
        res = [dataset_name]
        for j in range(16):
            result = extract_final_test_results(blocks[i + j])
            if result:
                res.append(f"{result[0]} ± {result[1]}")
            else:
                res.append("N/A")
        final_test_results.append(res)
    return final_test_results

def do_csv(file_name, name_emb):
    blocks = read_blocks_from_file(file_name)
    # Process blocks and save results to CSV
    final_test_results = process_blocks(blocks, name_emb)
    
    save_to_csv(final_test_results)

    print("Final test results have been saved to 'final_test_results.csv'")
