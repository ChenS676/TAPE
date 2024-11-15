import argparse
import torch
from load import load_data_lp
from load_data_lp import token_statistic
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--name', type=str, required=False, default='cora')
    parser.add_argument('--undirected', type=bool, required=False, default=True)
    parser.add_argument('--include_negatives', type=bool, required=False, default=True)
    parser.add_argument('--val_pct', type=float, required=False, default=0.15)
    parser.add_argument('--test_pct', type=float, required=False, default=0.05)
    parser.add_argument('--split_labels', type=bool, required=False, default=True)
    parser.add_argument('--device', type=str, required=False, default='cpu')
    return parser.parse_args()

def check_data_leakage(splits, log):
    sets = ['train', 'valid', 'test']
    leakage = False

    # Extract indices
    train_pos_index = set(map(tuple, splits['train'].pos_edge_label_index.t().tolist()))
    train_neg_index = set(map(tuple, splits['train'].neg_edge_label_index.t().tolist()))
    valid_pos_index = set(map(tuple, splits['valid'].pos_edge_label_index.t().tolist()))
    valid_neg_index = set(map(tuple, splits['valid'].neg_edge_label_index.t().tolist()))
    test_pos_index = set(map(tuple, splits['test'].pos_edge_label_index.t().tolist()))
    test_neg_index = set(map(tuple, splits['test'].neg_edge_label_index.t().tolist()))

    # Check for leakage
    if train_pos_index & valid_pos_index:
        log.write("Data leakage found between train and valid positive samples.\n")
        leakage = True
    if train_pos_index & test_pos_index:
        log.write("Data leakage found between train and test positive samples.\n")
        leakage = True
    if valid_pos_index & test_pos_index:
        log.write("Data leakage found between valid and test positive samples.\n")
        leakage = True
    if train_neg_index & valid_neg_index:
        log.write("Data leakage found between train and valid negative samples.\n")
        leakage = True
    if train_neg_index & test_neg_index:
        log.write("Data leakage found between train and test negative samples.\n")
        leakage = True
    if valid_neg_index & test_neg_index:
        log.write("Data leakage found between valid and test negative samples.\n")
        leakage = True

    if not leakage:
        log.write("No data leakage found.\n")

def check_self_loops(data, log):
    self_loops = (data.edge_index[0] == data.edge_index[1]).nonzero(as_tuple=False)
    if self_loops.size(0) > 0:
        log.write("Self-loops found.\n")
    else:
        log.write("No self-loops found.\n")

def check_edges_completeness(splits, data, log):
    rate = 2 * (float(splits['train']['pos_edge_label_index'].size(1) +
                     splits['test']['pos_edge_label_index'].size(1) +
                     splits['valid']['pos_edge_label_index'].size(1))) / data.edge_index.size(1)
    log.write(f"Edges completeness rate: {rate:.4f}\n")

def check_is_symmetric(edge_index):
    src, dst = edge_index
    num_edges = edge_index.size(1)

    reverse_edges = torch.stack([dst, src], dim=0)

    edge_set = set()
    for i in range(num_edges):
        edge = (src[i].item(), dst[i].item())
        edge_set.add(edge)
    for i in range(num_edges):
        reverse_edge = (reverse_edges[0, i].item(), reverse_edges[1, i].item())
        if reverse_edge not in edge_set:
            return False

    return True

if __name__ == "__main__":
    args = parse_args()
    args.split_index = [0.8, 0.15, 0.05]

    with open("dataset_report.txt", "w") as log:
        for dataset in ['computers', 'photo']:
            log.write(f"\n\n\nChecking dataset {dataset} :\n")
            args.name = dataset
            splits, text, data = load_data_lp[dataset](args)

            check_data_leakage(splits, log)
            check_self_loops(data, log)
            check_edges_completeness(splits, data, log)
            
            token_statistic([dataset], log)
            log.write(f"Is data.edge_index symmetric? {check_is_symmetric(data.edge_index)}\n")
            log.write(f"Is splits['test']['pos_edge_label_index'] symmetric? {check_is_symmetric(splits['test']['pos_edge_label_index'])}\n")
            log.write(f"Is splits['valid']['pos_edge_label_index'] symmetric? {check_is_symmetric(splits['valid']['pos_edge_label_index'])}\n")
            log.write(f"Is splits['train']['pos_edge_label_index'] symmetric? {check_is_symmetric(splits['train']['pos_edge_label_index'])}\n")
            log.write(f"Is splits['test']['neg_edge_label_index'] symmetric? {check_is_symmetric(splits['test']['neg_edge_label_index'])}\n")
            log.write(f"Is splits['valid']['neg_edge_label_index'] symmetric? {check_is_symmetric(splits['valid']['neg_edge_label_index'])}\n")
            log.write(f"Is splits['train']['neg_edge_label_index'] symmetric? {check_is_symmetric(splits['train']['neg_edge_label_index'])}\n")
            log.write(f"Is splits['test']['edge_index'] symmetric? {check_is_symmetric(splits['test']['edge_index'])}\n")
            log.write(f"Is splits['valid']['edge_index'] symmetric? {check_is_symmetric(splits['valid']['edge_index'])}\n")
            log.write(f"Is splits['train']['edge_index'] symmetric? {check_is_symmetric(splits['train']['edge_index'])}\n")

    print("Report has been saved to 'dataset_report.txt'.")
