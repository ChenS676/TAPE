import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import scipy.sparse as ssp
import torch
import argparse
import wandb
import time
import matplotlib.pyplot as plt
import networkx as nx
import csv
from torch.utils.tensorboard import SummaryWriter
from sklearn.neural_network import MLPClassifier
from ogb.linkproppred import Evaluator
from graph_embed.ge.models import Struc2Vec
from tune_utils import save_parameters
from data_utils.load_data_lp import get_edge_split
from data_utils.load import load_graph_lp as data_loader
<<<<<<< HEAD
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel
from heuristic.eval import get_metric_score

# Set the file path for the project
FILE_PATH = get_git_repo_root_path() + '/'
=======
from graphgps.utility.utils import (
    set_cfg,
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel
)
from ge.models.struc2vec import Struc2Vec
from networkx import from_scipy_sparse_matrix as from_scipy_sparse_array
>>>>>>> 89ecc007765bc968264719cad8b571269a77729f

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

def setup_device():
    """Sets up the device for computation."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(0)
        return 'cuda'
    return 'cpu'

def preprocess_data(cfg):
    """Preprocesses the data according to the configuration."""
    if cfg.data.name == 'cora':
        dataset, _ = data_loader[cfg.data.name](cfg)
    elif cfg.data.name == 'pubmed':
        dataset = data_loader[cfg.data.name](cfg)
    elif cfg.data.name == 'arxiv_2023':
        dataset = data_loader[cfg.data.name]()
        
    undirected = dataset.is_undirected()
    splits = get_edge_split(
        dataset, undirected, cfg.data.device, cfg.data.split_index[1],
        cfg.data.split_index[2], cfg.data.include_negatives, cfg.data.split_labels
    )
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset.num_nodes
    
<<<<<<< HEAD
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), 
                             (full_edge_index[0], full_edge_index[1])), 
                             shape=(num_nodes, num_nodes))
    adj = full_A
    G = nx.from_scipy_sparse_array(adj)
=======
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes))
    adj = to_scipy_sparse_matrix(full_edge_index)
    G = from_scipy_sparse_array(adj)
>>>>>>> 89ecc007765bc968264719cad8b571269a77729f
    
    return dataset, splits, G, full_A

def train_model(G, cfg):
    """Trains the Struc2Vec model."""
    struc2vec_params = cfg.model.struc2vec
    model = Struc2Vec(
        G, walk_length=struc2vec_params.wl, num_walks=struc2vec_params.num_walks, workers=20, verbose=40,
        data=cfg.data.name, reuse=False, temp_path='./temp_path'
    )
    start = time.time()
    model.train(embed_size=struc2vec_params.embed_size, window_size=struc2vec_params.window_size, workers=20, epochs=struc2vec_params.epoch)
    end = time.time()
    embeddings = model.get_embeddings()
    root = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}')
    params = [root, model, start, end, cfg.model.struc2vec.epoch]
    
    npz_file = os.path.join(root, f'{cfg.data.name}_embeddings.npz')
    if isinstance(embeddings, dict):
        embeddings_str_keys = {str(key): value for key, value in embeddings.items()}
        np.savez(npz_file, **embeddings_str_keys)
    else:
        np.savez(npz_file, embeddings=embeddings)
    
    return model, embeddings, params

def evaluate_model(embed, splits, cfg, params):
    """Evaluates the model using logistic regression."""
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    X_train = np.multiply(embed[X_train_index[:, 1]], embed[X_train_index[:, 0]])
    
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    X_test = np.multiply(embed[X_test_index[:, 1]], embed[X_test_index[:, 0]])
    
    start_eval = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200).fit(X_train, y_train)
    end_eval = time.time()
    y_pred = clf.predict_proba(X_test)
    acc = clf.score(X_test, y_test)
    
    root, model, start_train, end_train, epochs = params
    save_parameters(root, model, start_train, end_train, epochs, start_eval, end_eval, f"{cfg.data.name}_model_parameters.csv")
    
    plt.figure()
    plt.plot(y_pred, label='pred')
    plt.plot(y_test, label='test')
    plt.savefig('struc2vec_pred.png')
    
    return acc, y_pred, y_test

def calculate_mrr(y_pred, y_test):
    """Calculates the Mean Reciprocal Rank (MRR) for predictions."""
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    pos_pred = pos_test_pred[:, 1]
    neg_pred = neg_test_pred[:, 1]
    
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
    return result_mrr

if __name__ == "__main__":
    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg)

    device = setup_device()
    torch.set_num_threads(cfg.num_threads)
    
    seeds = [1, 2, 3, 4, 5]
    mrr_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        run_name = f'seed_{seed}'
        writer = SummaryWriter(log_dir=os.path.join(FILE_PATH, 'runs', cfg.data.name, run_name))

        # Preprocess and train the model
        dataset, splits, G, full_A = preprocess_data(cfg)
        model, embed, params = train_model(G, cfg)
        
        # Evaluate the model
        acc, y_pred, y_test = evaluate_model(embed, splits, cfg, params)
        result_mrr = calculate_mrr(y_pred, y_test)
        result_mrr['ACC'] = acc
        
        # Save the results
        root = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}', run_name)
        os.makedirs(root, exist_ok=True)

        acc_file = os.path.join(root, f'{cfg.data.name}_acc.csv')
        mrr_file = os.path.join(root, f'{cfg.data.name}_mrr.csv')

        run_id = wandb.util.generate_id()
        append_acc_to_excel(run_id, {'struc2vec_acc': acc}, acc_file, cfg.data.name, cfg.model.type)
        append_mrr_to_excel(run_id, {'struc2vec_mrr': result_mrr}, mrr_file, cfg.data.name, cfg.model.type)

        # Save MRR results for further processing
        mrr_results.append(result_mrr)

        writer.close()

    # Calculate mean and variance across all seeds
    columns = {key: [d[key] for d in mrr_results] for key in mrr_results[0]}
    means = {key: np.mean(values) for key, values in columns.items()}
    variances = {key: np.var(values, ddof=1) for key, values in columns.items()}

    # Save the aggregated results
    root = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}')
    mrr_file = os.path.join(root, f'{cfg.data.name}_gr_emb_res.csv')
    
    with open(mrr_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@20', 'Hits@50', 'Hits@100', 'MRR', 
                        'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 'mrr_hit20', 'mrr_hit50', 'mrr_hit100', 'AUC', 'AP', 'ACC'])
        
        keys = ["Hits@1", "Hits@3", "Hits@10", "Hits@20", "Hits@50", "Hits@100", "MRR", 
                "mrr_hit1", "mrr_hit3", "mrr_hit10", "mrr_hit20", "mrr_hit50", "mrr_hit100", 
                "AUC", "AP", "ACC"]

        row = [f"{run_id}_{cfg.data.name}"] + [f'{means.get(key, 0) * 100:.2f} ± {variances.get(key, 0) * 100:.2f}' for key in keys]
        writer.writerow(row)
    
    # Calculate mean across all model parameters
    file_path = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}/{cfg.data.name}_model_parameters.csv')
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)

        data = []
        for row in reader:
            data.append([float(value) for value in row[1:]])

        rows = np.array(data)

    means = np.mean(rows, axis=0)
    mean_row = ['Mean'] + [f'{mean:.6f}' for mean in means]
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(mean_row)