import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import scipy.sparse as ssp
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import to_scipy_sparse_matrix
from ogb.linkproppred import Evaluator
from yacs.config import CfgNode as CN
from heuristic.eval import get_metric_score
from data_utils.load import load_data_lp as data_loader
from lpda.adjacency import plot_coo_matrix, construct_sparse_adj
from core.graphgps.utility.utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel
)

import uuid
from ge.classify import read_node_label, Classifier
from ge import Struc2Vec
import itertools
import wandb



if __name__ == "__main__":

    FILE_PATH = get_git_repo_root_path() + '/'

    cfg_file = FILE_PATH + "core/configs/arxiv_2023/struc2vec.yaml"
    # # Load args file
    with open(cfg_file, "r") as f:
        cfg = CN.load_cfg(f)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    if torch.cuda.is_available():
        # Get the number of available CUDA devices
        num_cuda_devices = torch.cuda.device_count()

        if num_cuda_devices > 0:
            # Set the first CUDA device as the active device
            torch.cuda.set_device(0)
            device = 'cuda'
    else:
        device = 'cpu'
    
    dataset, data_cited, splits = data_loader[cfg.data.name](cfg)
    # Access individual parameters
    max_iter = cfg.model.struc2vec.max_iter
    
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')
    
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    result_dict = {}
    
    adj = to_scipy_sparse_matrix(full_edge_index)

    G = nx.from_scipy_sparse_array(adj)
    
    # three parameters
    model = Struc2Vec(G, 
                      walk_length=10, 
                      num_walks = 80, 
                      workers=20, 
                      verbose=40, 
                      data=cfg.data.name, 
                      reuse=False, 
                      temp_path=f'./temp_path')
    
    model.train(embed_size=128, 
                window_size=5, 
                workers=20)

    embed = model.get_embeddings()

    # Load the array back from the npz file
    # embed = np.load('core/Embedding/structure_arxiv_2023_thomasha.npz')['my_array']

    print(f"embedding size {embed.shape}")

    # embedding method 
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    # dot product
    X_train = embed[X_train_index]
    X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    # dot product 
    X_test = embed[X_test_index]
    X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
    

    clf = LogisticRegression(solver='lbfgs', max_iter=max_iter, multi_class='auto')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_test)

    acc = clf.score(X_test, y_test)

    plt.figure()
    plt.plot(y_pred, label='pred')
    plt.plot(y_test, label='test')
    plt.savefig('node2vec_pred.png')
        
    results_acc = {'node2vec_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    pos_pred = pos_test_pred[:, 1]
    neg_pred = neg_test_pred[:, 1]
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
    results_mrr = {'node2vec_mrr': result_mrr}
    print(results_acc, results_mrr)


    root = FILE_PATH + 'results'
    acc_file = root + f'/{cfg.data.name}_acc.csv'
    mrr_file = root +  f'/{cfg.data.name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    
    id = wandb.util.generate_id()
    append_acc_to_excel(id, results_acc, acc_file, cfg.data.name, 'struc2vec')
    append_mrr_to_excel(id, results_mrr, mrr_file, 'struc2vec')
    


