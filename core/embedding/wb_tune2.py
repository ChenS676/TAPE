
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import pprint
import numpy as np
import scipy.sparse as ssp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from numba.typed import List
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.graphgym.config import cfg
from embedding.node2vec_tagplus import node2vec, data_loader
from yacs.config import CfgNode as CN
from heuristic.eval import get_metric_score
from core.graphgps.utility.utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel,
)
import wandb 
from ogb.linkproppred import Evaluator
from core.embedding.tune_utils import (
    set_cfg,
    parse_args,
    load_sweep_config, 
    initialize_config, 
    param_tune_acc_mrr,
    FILE_PATH,
    wandb_record_files
)


# Constants
FILE_PATH = get_git_repo_root_path() + '/'

def set_cfg(file_path, args):
    with open(file_path + args.cfg_file, "r") as f:
        return CN.load_cfg(f)

def load_sweep_config(file_path):
    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)

def print_args(args):
    print(args)




import argparse

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default = "core/configs/cora/gae.yaml",
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default = "core/configs/cora/gae_sweep.yaml",
                        help='The configuration file path.')
    
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()


def process_edge_index(full_edge_index):
    print("full_edge_index", full_edge_index.shape)
    return to_scipy_sparse_matrix(full_edge_index)

def condition(p, q, method, data):
    # for arxiv  p < q for others p > q
    if method == 'node2vec':
        if data == 'cora' or data == 'pubmed':
            return p > q
        if data == 'arxiv_2023':
            return p < q

    if method == 'deepwalk':
        if p == q:
            return True 
        else:
            return False
    


def perform_node2vec_embedding(adj, config, splits, method):
    pprint.pprint(config)
    walk_length = config.wl
    num_walks = config.num_walks
    # for deepwalk p = q = 1
    p = 1
    q = 1

    embed_size = cfg.model.node2vec.embed_size
    num_neg_samples = cfg.model.node2vec.num_neg_samples
    workers = cfg.model.node2vec.workers

    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    print("X_train_index range", X_train_index.max(), X_train_index.min())
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    
    if condition(p, q,  method, cfg.data.name): # for arxiv  p < q for others p > q
        embed = node2vec(workers,
                         adj,
                         embedding_dim=embed_size,
                         walk_length=walk_length,
                         walks_per_node=num_walks,
                         num_neg_samples=num_neg_samples,
                         p=p,
                         q=q
                         )
        print(f"embed.shape: {embed.shape}")

        if X_train_index.max() < embed.shape[0]:
            X_train = embed[X_train_index]
            X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
            X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
            X_test = embed[X_test_index]
            X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))

            return X_train, y_train, X_test, y_test

    return None

def train_and_evaluate_logistic_regression(id, X_train, y_train, X_test, y_test, max_iter, method):
    clf = LogisticRegression(solver='lbfgs', max_iter=max_iter, multi_class='auto')
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print("acc", acc)

    y_pred = clf.predict_proba(X_test)
    results_acc = {'node2vec_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    pos_pred = pos_test_pred[:, 1]
    neg_pred = neg_test_pred[:, 1]
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
    results_mrr = {'node2vec_mrr': result_mrr}

    root = FILE_PATH + 'results'
    acc_file = root + f'/{cfg.data.name}_acc.csv'
    mrr_file = root + f'/{cfg.data.name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    append_acc_to_excel(id, results_acc, acc_file, cfg.data.name, method)
    append_mrr_to_excel(id, results_mrr, mrr_file, method)

    print(results_acc, '\n', results_mrr)
    return acc


args = parse_args()

print_args(args)

SWEEP_FILE_PATH = FILE_PATH + args.sweep_file
sweep_config = load_sweep_config(SWEEP_FILE_PATH)

cfg = initialize_config(FILE_PATH, args)

_, _, splits = data_loader[cfg.data.name](cfg)

global X_train_index, y_train, X_test_index, X_test

def run_experiment(config=None):
    id = wandb.util.generate_id()
    run = wandb.init(id=id, config=config, settings=wandb.Settings(_service_wait=300), save_code=True)

    wandb_config = wandb.config
    
    wandb.log(dict(wandb_config))
    full_edge_index = splits['test'].edge_index
    adj = process_edge_index(full_edge_index)

    method = 'deepwalk'
    embedding_results = perform_node2vec_embedding(adj, wandb_config, splits, method)
    if embedding_results:
        X_train, y_train, X_test, y_test = embedding_results
        acc = train_and_evaluate_logistic_regression(id, X_train, y_train, X_test, y_test, cfg.model.node2vec.max_iter, method)
        run.log({"score": acc})
        run.log_code("../", include_fn=wandb_record_files)

    else:
        run.log({"score": 0})


sweep_id = wandb.sweep(sweep=sweep_config, project=f"deepwalk-sweep-{cfg.data.name}")
wandb.agent(sweep_id, run_experiment, count=60)

