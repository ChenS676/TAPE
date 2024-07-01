import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np 
import scipy.sparse as ssp
import wandb
import torch
from pathlib import Path 
import matplotlib.pyplot as plt 
from IPython import embed 
from scipy.spatial import distance
from sklearn.neural_network import MLPClassifier
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from heuristic.eval import get_metric_score
from data_utils.load import data_loader_nc
from embedding.tune_utils import parse_args, param_tune_acc_mrr 
from core.graphgps.utility.utils import get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel, set_cfg
from core.graphgps.visualization.adj import construct_sparse_adj, plot_coo_matrix, plot_pos_neg_adj

method = 'nonlinear_mlp'
    
FILE_PATH = get_git_repo_root_path() + '/'

distance_metric = {
    'cos': distance.cosine,
    'l2': distance.euclidean,
    'hamming': distance.hamming,
    'jaccard': distance.jaccard,
    'dice': distance.dice,
    'dot': lambda x, y: np.dot(x, y)
}


def pairwise_prediction(data, test_index, distance):
    """predict the link existence using the pairwise textual similarity"""
    test_index = test_index.numpy().transpose()
    test_pred = []

    if distance not in distance_metric:
        raise ValueError("Invalid distance metric specified.")

    metric_function = distance_metric[distance]

    for ind in test_index:
        metric_value = metric_function(data[ind[0]], data[ind[1]])
        test_pred.append(metric_value)

    test_pred = torch.tensor(np.asarray(test_pred))
    return test_pred


def create_node_feat(edge_label_index, train_node_feat):
    node_feat = torch.zeros(edge_label_index.shape[0], train_node_feat.shape[1]*2)
    for i, (srt, trg) in enumerate(edge_label_index):
        node_feat[i, :] = torch.cat((train_node_feat[srt], train_node_feat[trg]), 0)
    
    return node_feat
        
def MLP_as_compat(splits, max_iter=10000, method='node_sim', predict='soft', data=None):
    # train loop
    
    X_train_feat = create_node_feat(splits['train'].edge_label_index.T, splits['train'].x)
    
    y_train = splits['train'].edge_label
    
    pos_test_index = splits['test'].edge_label_index[:, splits['test'].edge_label == 1].T
    neg_test_index = splits['test'].edge_label_index[:, splits['test'].edge_label == 0].T

    X_pos_test_feat = create_node_feat(pos_test_index, splits['test'].x) 
    X_neg_test_feat = create_node_feat(neg_test_index, splits['test'].x) 
    
    clf = MLPClassifier(hidden_layer_sizes=(256,), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001, 
                        batch_size='auto', 
                        learning_rate_init=0.001, 
                        power_t=0.5, 
                        max_iter=max_iter, 
                        shuffle=True, 
                        random_state=None, 
                        tol=0.0001, 
                        verbose=False, 
                        warm_start=False, 
                        momentum=0.9, 
                        nesterovs_momentum=True, 
                        early_stopping=False, 
                        validation_fraction=0.1, 
                        epsilon=1e-08, 
                        n_iter_no_change=10, 
                        max_fun=15000).fit(X_train_feat, y_train)

    
    if predict == 'hard':
        y_pos_pred = torch.tensor(clf.predict(X_pos_test_feat))
        y_neg_pred = torch.tensor(clf.predict(X_neg_test_feat))
    elif predict == 'soft':
        y_pos_pred = torch.tensor(clf.predict_proba(X_pos_test_feat)[:, 1])
        y_neg_pred = torch.tensor(clf.predict_proba(X_neg_test_feat)[:, 1])
    
    test_pos_label = splits['test'].edge_label[splits['test'].edge_label == 1]
    test_neg_label = splits['test'].edge_label[splits['test'].edge_label == 0]
    
    acc_pos = clf.score(X_pos_test_feat, test_pos_label)
    acc_neg = clf.score(X_neg_test_feat, test_neg_label)
    results_acc = {f'{method}_acc': (acc_pos+ acc_neg)/2}
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    weight_matrices = clf.coefs_
    
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.imshow(weight_matrices[0].T)
    id = wandb.util.generate_id()
    plt.savefig(f'MLP_{cfg.data.name}visual_{id}.png')

    results_acc.update(get_metric_score(evaluator_hit, evaluator_mrr, y_pos_pred, y_neg_pred))
    return results_acc


def eval_node_sim_acc(cfg) -> None:
    """load text attribute graph in link predicton setting
    """
    dataset, data_cited, splits = data_loader_nc[cfg.data.name](cfg)
    
    # ust test edge_index as full_A
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')
    
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

    # only for debug
    test_index = splits['test'].edge_label_index
    
    pos_m = construct_sparse_adj(test_index)
    plot_coo_matrix(pos_m, f'test_pos_index.png')
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    result_dict = {}

    pos_test_index = splits['test'].edge_label_index[:, splits['test'].edge_label == 1]
    neg_test_index = splits['test'].edge_label_index[:, splits['test'].edge_label == 0]
    
    for method in ['pairwise_pred']:
        for dist in ['dot']:
            pos_test_pred = pairwise_prediction(dataset._data.x, pos_test_index, dist)
            neg_test_pred = pairwise_prediction(dataset._data.x, neg_test_index, dist)
            result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
            result_dict.update({f'{method}_{dist}': result})

    result_dict = MLP_as_compat(splits, max_iter=10000, predict='soft', data=cfg.data.name)
    

    root = FILE_PATH + 'results/node_sim/'
    acc_file = root + f'{cfg.data.name}_acc_{method}.csv'

    if not os.path.exists(root):
        Path(root).mkdir(parents=True, exist_ok=False)
    
    id = wandb.util.generate_id()
    param_tune_acc_mrr(id, result_dict, acc_file, cfg.data.name, method)
    
    return result_dict


if __name__ == "__main__":
    args = parse_args()
    # Load args file
    
    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)


    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    result = eval_node_sim_acc(cfg)
    print(result)
