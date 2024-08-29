import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import scipy.sparse as ssp
import torch
import wandb
import time
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from ogb.linkproppred import Evaluator
import csv
from torch.utils.tensorboard import SummaryWriter  # Импортируем TensorBoard

from graph_embed.ge.models import Node2Vec
from tune_utils import save_parameters
from data_utils.load_data_lp import get_edge_split
from data_utils.load import load_graph_lp as data_loader
from data_utils.lcc import construct_sparse_adj
from data_utils.graph_stats import plot_coo_matrix
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel, parse_args
from heuristic.eval import get_metric_score

# Set the file path for the project
FILE_PATH = get_git_repo_root_path() + '/'

def eval_embed(embed, splits, visual=True, params=None, writer=None):
    """Trains the classifier and returns predictions."""
    embed = np.asarray(list(embed.values()))
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label

    # Compute dot products for training and testing
    X_train = np.multiply(embed[X_train_index[:, 1]], embed[X_train_index[:, 0]])
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    X_test = np.multiply(embed[X_test_index[:, 1]], embed[X_test_index[:, 0]])

    # Train classifier
    start_eval = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200).fit(X_train, y_train)
    end_eval = time.time()
    y_pred = clf.predict(X_test)
    acc = clf.score(X_test, y_test)

    root, model, start_train, end_train, epochs = params
    save_parameters(root, model, start_train, end_train, epochs, start_eval, end_eval, f"{cfg.data.name}_model_parameters.csv")
    
    # Логируем метрику accuracy в TensorBoard
    if writer:
        writer.add_scalar('Accuracy/test', acc)

    # Evaluate predictions
    results_acc = {'node2vec_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
    result_mrr['ACC'] = acc
    results_mrr = {'node2vec_mrr': result_mrr}

    # Логируем метрику MRR в TensorBoard
    if writer:
        writer.add_scalar('MRR/test', result_mrr['MRR'])

    # Visualization
    if visual:
        tsne = TSNE(n_components=2, random_state=0, perplexity=100, n_iter=300)
        node_pos = tsne.fit_transform(X_test)

        color_dict = {'0.0': 'r', '1.0': 'b'}
        colors = [color_dict[str(label)] for label in y_test.tolist()]
        plt.figure()
        for idx in range(len(node_pos)):
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c=colors[idx])
        plt.legend()
        plt.savefig('cora_node2vec.png')

        # Логируем t-SNE изображение в TensorBoard
        if writer:
            writer.add_figure('t-SNE', plt.gcf())

    return y_pred, results_acc, results_mrr, y_test

def eval_mrr_acc(cfg, writer=None) -> None:
    """Loads the graph data and evaluates embeddings using MRR and accuracy."""
    if cfg.data.name == 'cora':
        dataset, _ = data_loader[cfg.data.name](cfg)
    elif cfg.data.name == 'pubmed':
        dataset = data_loader[cfg.data.name](cfg)
    elif cfg.data.name == 'arxiv_2023':
        dataset = data_loader[cfg.data.name]()
        
    undirected = dataset.is_undirected()
    splits = get_edge_split(dataset, undirected, cfg.data.device,
                            cfg.data.split_index[1], cfg.data.split_index[2],
                            cfg.data.include_negatives, cfg.data.split_labels)

    # Create the full adjacency matrix from test edges
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset.num_nodes
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), 
                             (full_edge_index[0], full_edge_index[1])), 
                             shape=(num_nodes, num_nodes))

    # Visualize the test edge adjacency matrix
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, 'test_edge_index.png')

    # Extract Node2Vec parameters from config
    node2vec_params = cfg.model.node2vec
    G = nx.from_scipy_sparse_array(full_A, create_using=nx.Graph())

    model = Node2Vec(G, walk_length=node2vec_params.walk_length, 
                        num_walks=node2vec_params.num_walks,
                        p=node2vec_params.p, 
                        q=node2vec_params.q, 
                        workers=node2vec_params.workers, 
                        use_rejection_sampling=node2vec_params.use_rejection_sampling)
    
    start = time.time()

    model.train(embed_size=node2vec_params.embed_size,
                window_size=node2vec_params.window_size, 
                iter=node2vec_params.max_iter,
                epochs=node2vec_params.epoch)
    end = time.time()
    
    embeddings = model.get_embeddings()
    root = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}')
    params = [root, model, start, end, node2vec_params.epoch]
    
    # Save embeddings to npz file
    npz_file = os.path.join(root, f'{cfg.data.name}_embeddings.npz')
    embeddings_str_keys = {str(key): value for key, value in embeddings.items()}
    np.savez(npz_file, **embeddings_str_keys)

    return eval_embed(embeddings, splits, params=params, writer=writer)

if __name__ == "__main__":
    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    seeds = [1, 2, 3, 4, 5]
    
    mrr_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        run_name = f'seed_{seed}'
        writer = SummaryWriter(log_dir=os.path.join(FILE_PATH, 'runs', cfg.data.name, run_name))

        # Run the evaluation
        y_pred, results_acc, results_mrr, y_test = eval_mrr_acc(cfg, writer=writer)

        # Save the results
        root = os.path.join(FILE_PATH, f'results/graph_emb/{cfg.data.name}/{cfg.model.type}', run_name)
        os.makedirs(root, exist_ok=True)

        acc_file = os.path.join(root, f'{cfg.data.name}_acc.csv')
        mrr_file = os.path.join(root, f'{cfg.data.name}_mrr.csv')

        run_id = wandb.util.generate_id()
        append_acc_to_excel(run_id, results_acc, acc_file, cfg.data.name, cfg.model.type)
        append_mrr_to_excel(run_id, results_mrr, mrr_file, cfg.data.name, cfg.model.type)

        mrr_results.append(results_mrr['node2vec_mrr'])

        writer.close()


    columns = {key: [d[key] for d in mrr_results] for key in mrr_results[0]}

    means = {key: np.mean(values) for key, values in columns.items()}
    variances = {key: np.var(values, ddof=1) for key, values in columns.items()}

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