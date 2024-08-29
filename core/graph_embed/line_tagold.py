
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import argparse
import numpy as np
import scipy.sparse as ssp
import matplotlib.pyplot as plt
import torch
import networkx as nx
from sklearn.linear_model import LogisticRegression
from ogb.linkproppred import Evaluator

# Custom imports
from ge.models.line_tf import LINE
# from ge.models.line_torch import LINE_torch
from tune_utils import save_parameters
from heuristic.eval import get_metric_score
from data_utils.load import load_graph_lp as data_loader
from data_utils.graph_stats import plot_coo_matrix, construct_sparse_adj
from data_utils.load_data_lp import get_edge_split
from graphgps.utility.utils import set_cfg, append_acc_to_excel, append_mrr_to_excel
from torch_geometric.utils import to_scipy_sparse_matrix
from graph_embed.tune_utils import get_git_repo_root_path
import wandb

def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')
    return parser.parse_args()

def set_device():
    """Set the PyTorch device based on availability."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(0)
        return 'cuda'
    return 'cpu'

def prepare_data(cfg):
    """Load and prepare the dataset."""
    dataset, _ = data_loader[cfg.data.name](cfg)
    undirected = dataset.is_undirected()
    splits = get_edge_split(dataset, undirected, cfg.data.device, cfg.data.split_index[1], cfg.data.split_index[2], cfg.data.include_negatives, cfg.data.split_labels)
    return dataset, splits

def create_graph(splits):
    """Construct the graph from the dataset splits."""
    full_edge_index = splits['test'].edge_index
    # full_edge_weight = torch.ones(full_edge_index.size(1))
    # num_nodes = splits['train'].num_nodes  # Assuming num_nodes is available in splits

    # m = construct_sparse_adj(full_edge_index)
    # plot_coo_matrix(m, f'test_edge_index.png')
    
    # full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 
    adj = to_scipy_sparse_matrix(full_edge_index)
    G = nx.from_scipy_sparse_array(adj)
    return G

def train_line_model(G, cfg, device):
    """Train the LINE model on the graph."""
    model = LINE(G, embedding_size=96, order='all', lr=0.01)
    # model = LINE_torch(G, embedding_size=96, order='all', lr=0.00001, device=device)
    start_time = time.time()
    epochs = 1
    model.train(batch_size=2048, epochs=epochs, verbose=2)
    end_time = time.time()
    save_parameters(os.path.join(get_git_repo_root_path() + '/TAPE/', 'results'), model, start_time, end_time, epochs)
    return model

def evaluate_model(model, splits, cfg):
    """Evaluate the trained model using logistic regression."""
    embed = model.get_embeddings()

    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    X_train = np.multiply(embed[X_train_index][:, 1], embed[X_train_index][:, 0])
    
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    X_test = np.multiply(embed[X_test_index][:, 1], embed[X_test_index][:, 0])

    clf = LogisticRegression(solver='lbfgs', max_iter=cfg.model.line.max_iter, multi_class='auto')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_test)
    acc = clf.score(X_test, y_test)
    
    return y_pred, acc, y_test

def save_results(y_pred, y_test, acc, cfg):
    """Save the evaluation results."""
    method = cfg.model.type
    
    plt.figure()
    plt.plot(y_pred, label='pred')
    plt.plot(y_test, label='test')
    plt.savefig(f'{method}_pred.png')

    results_acc = {f'{method}_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    pos_pred = pos_test_pred[:, 1]
    neg_pred = neg_test_pred[:, 1]
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
    result_mrr['ACC'] = acc
    results_mrr = {f'{method}_mrr': result_mrr}

    root = os.path.join(get_git_repo_root_path(), 'results')
    acc_file = os.path.join(root, f'{cfg.data.name}_acc.csv')
    mrr_file = os.path.join(root, f'{cfg.data.name}_mrr.csv')

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    
    wandb_id = wandb.util.generate_id()
    append_acc_to_excel(wandb_id, results_acc, acc_file, cfg.data.name, method)
    append_mrr_to_excel(wandb_id, results_mrr, mrr_file, cfg.data.name, method)

    print("Results saved.")

def main():
    """Main execution function."""
    args = parse_args()
    cfg = set_cfg(get_git_repo_root_path() + '/', args.cfg)
    
    torch.set_num_threads(cfg.num_threads)
    device = set_device()
    dataset, splits = prepare_data(cfg)
    G = create_graph(splits)
    model = train_line_model(G, cfg, device)
    
    y_pred, acc, y_test = evaluate_model(model, splits, cfg)
    save_results(y_pred, y_test, acc, cfg)

if __name__ == "__main__":
    main()
