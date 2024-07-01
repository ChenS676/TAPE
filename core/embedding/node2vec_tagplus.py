# Standard library imports
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Third-party library imports
import numba
import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from IPython import embed
from joblib import Parallel, delayed

# External module imports
import torch
import matplotlib.pyplot as plt
from ogb.linkproppred import Evaluator
from yacs.config import CfgNode as CN
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import to_scipy_sparse_matrix
import itertools 
import scipy.sparse as ssp
from data_utils.load import load_data_lp as data_loader
from heuristic.eval import get_metric_score
from lpda.adjacency import plot_coo_matrix, construct_sparse_adj
from core.graphgps.utility.utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel
)

FILE_PATH = get_git_repo_root_path() + '/'

data_loader = {
    'cora': get_cora_casestudy,
    'pubmed': get_pubmed_casestudy,
    'arxiv_2023': get_raw_text_arxiv_2023
}


def set_cfg(args):
    with open(args.cfg_file, "r") as f:
        cfg = CN.load_cfg(f)
    return cfg

def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
    
def node2vec(workers, 
            adj, 
            embedding_dim, 
            walk_length, 
            walks_per_node,
            num_neg_samples, 
            p, 
            q, 
            ):
    """
    参数说明
    -------------
    adj : 图的邻接矩阵
    embedding_dim : 图嵌入的维度
    walk_length : 随机游走的长度
    walks_per_node : 每个节点采样多少个随机游走
    workers: word2vec模型使用的线程数量
    window_size: word2vec模型中使用的窗口大小
    num_neg_samples : 负样本的数量
    p: node2vec的p参数
    q: node2vec的q参数
    """
    if workers == 1:
        adj = sp.csr_matrix(adj)
        walks = _n2v_random_walk_iterator(adj.indptr,
                                    adj.indices,
                                    walk_length,
                                    walks_per_node,
                                    p,
                                    q)
        
    else:
    
        results = Parallel(n_jobs=workers, verbose=0, )(
            delayed(sample_n2v_random_walks)(adj, walk_length, num, p=p, q=q) for num in
            partition_num(walks_per_node, workers))
        walks = np.asarray(list(itertools.chain(*results)))
            
        walks = [list(map(str, walk)) for walk in walks]
    
    
    model = Word2Vec(walks, 
                     vector_size=embedding_dim, 
                     negative=num_neg_samples, 
                     compute_loss=True,
                     workers = workers,
                    #  epochs=epoch, 
                    #  sg=sg, 
                    #  hs=hs,
                    #  window=window, 
                    #  min_count=min_count, 
                    #  shrink_windows=shrink_window
                     )  

    embedding = model.wv.vectors[np.fromiter(map(int, model.wv.index_to_key), np.int32).argsort()] 
    
    return embedding


    
def sample_n2v_random_walks(adj, walk_length, walks_per_node, p, q):
    """
    返回值的类型
    -------
    walks : np.ndarray, shape [num_walks * num_nodes, walk_length]
        采样后的随机游走
    """
    adj = sp.csr_matrix(adj)
    random_walks = _n2v_random_walk(adj.indptr,
                                    adj.indices,
                                    walk_length,
                                    walks_per_node,
                                    p,
                                    q)
    return random_walks 


@numba.jit(nopython=True)
def _n2v_random_walk_iterator(indptr,
                    indices,
                    walk_length,
                    walks_per_node,
                    p,
                    q):
    N = len(indptr) - 1 # 节点数量

    for _ in range(walks_per_node):
        for n_iter in range(N):
            walk = np.zeros(walk_length+1, dtype=np.int32)
            walk[0] = n_iter   

            visited_set = None
            for il in range(walk_length):
                # v_iter 's neighbors
                neighbors = indices[indptr[n_iter]:indptr[n_iter+1]]
                
                sample_idx_arr = np.zeros(len(neighbors)+1, dtype=np.int32)
                sample_prob_arr = np.zeros(len(neighbors)+1, dtype=np.float32)
                
                for i, samples in enumerate(neighbors):
                    sample_idx_arr[i] = samples
                
                    if visited_set is not None:
                        if samples in visited_set:
                            sample_prob_arr[i] = 1
                        else:
                            sample_prob_arr[i] = 1/q
                    else:
                        sample_prob_arr[i] = 1/q 

                visited_set = neighbors.copy()
                sample_idx_arr[-1] = n_iter
                sample_prob_arr[-1] = 1/p
                    
                sample_prob_arr = sample_prob_arr / np.sum(sample_prob_arr)
                n_iter = random_choice(sample_idx_arr, sample_prob_arr)
                
                walk[il+1] = n_iter
            yield walk 


def _n2v_random_walk(indptr,
                    indices,
                    walk_length,
                    walks_per_node,
                    p,
                    q):
    N = len(indptr) - 1 # 节点数量
    final_walk = []
    for _ in range(walks_per_node):
        for n_iter in range(N):
            walk = np.zeros(walk_length+1, dtype=np.int32)
            walk[0] = n_iter   

            visited_set = None
            for il in range(walk_length):
                # v_iter 's neighbors
                neighbors = indices[indptr[n_iter]:indptr[n_iter+1]]
                
                sample_idx_arr = np.zeros(len(neighbors)+1, dtype=np.int32)
                sample_prob_arr = np.zeros(len(neighbors)+1, dtype=np.float32)
                
                for i, samples in enumerate(neighbors):
                    sample_idx_arr[i] = samples
                
                    if visited_set is not None:
                        if samples in visited_set:
                            sample_prob_arr[i] = 1
                        else:
                            sample_prob_arr[i] = 1/q
                    else:
                        sample_prob_arr[i] = 1/q 

                visited_set = neighbors.copy()
                sample_idx_arr[-1] = n_iter
                sample_prob_arr[-1] = 1/p
                    
                sample_prob_arr = sample_prob_arr / np.sum(sample_prob_arr)
                n_iter = random_choice(sample_idx_arr, sample_prob_arr)
                
                walk[il+1] = n_iter
            
            final_walk.append(walk)
        return np.array(final_walk)
            
@numba.jit(nopython=True)
def random_choice(arr: np.int64, p):
    """
    params
    ----------
    arr : 1d tensor
    p : probability of each element in arr
    
    返回值
    -------
    samples : sampled samples 
    adopted from blog https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
    """
    return arr[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]




# main function 
if __name__ == "__main__":

    # args = parse_args()
    # # # Load args file

    # cfg = set_cfg(args)
    # cfg.merge_from_list(args.opts)
    
    # # Set Pytorch environment
    # torch.set_num_threads(cfg.num_threads)

    FILE_PATH = get_git_repo_root_path() + '/'

    cfg_file = FILE_PATH + "core/configs/pubmed/node2vec.yaml"
    # # Load args file
    with open(cfg_file, "r") as f:
        cfg = CN.load_cfg(f)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    dataset, data_cited, splits = data_loader[cfg.data.name](cfg)
    
    # ust test edge_index as full_A
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')
    
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    result_dict = {}
    # Access individual parameters
    walk_length = cfg.model.node2vec.walk_length
    walks_per_node = cfg.model.node2vec.num_walks
    p = cfg.model.node2vec.p
    q = cfg.model.node2vec.q
    workers = cfg.model.node2vec.workers
    use_rejection_sampling = cfg.model.node2vec.use_rejection_sampling
    embed_size = cfg.model.node2vec.embed_size
    ws = cfg.model.node2vec.window_size
    iter = cfg.model.node2vec.iter
    num_neg_samples = cfg.model.node2vec.num_neg_samples 
    window = cfg.model.node2vec.window
    min_count = cfg.model.node2vec.min_count
    shrink_window = cfg.model.node2vec.shrink_window
    epoch = cfg.model.node2vec.epoch
    sg = cfg.model.node2vec.sg 
    hs = cfg.model.node2vec.hs
    
    # G = nx.from_scipy_sparse_matrix(full_A, create_using=nx.Graph())
    adj = to_scipy_sparse_matrix(full_edge_index)

    embed = node2vec(workers, 
                    adj, 
                    embedding_dim=embed_size, 
                    walk_length=walk_length,
                    walks_per_node=walks_per_node,
                    num_neg_samples=num_neg_samples,
                    p=p, 
                    q=q,
                    window=window, 
                    epoch=epoch, 
                    sg=sg, 
                    hs=hs,
                    min_count=min_count,
                    shrink_window=shrink_window)
    
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
    
    
    clf = LogisticRegression(solver='lbfgs', max_iter=iter, multi_class='auto')
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
    append_acc_to_excel(results_acc, acc_file, cfg.data.name)
    append_mrr_to_excel(results_mrr, mrr_file)
    

