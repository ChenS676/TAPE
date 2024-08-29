import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import networkx as nx
from matplotlib import pyplot
import numpy as np
import torch
import os
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_torch_coo_tensor
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import matspy as spy  # https://github.com/alugowski/matspy
import math
import argparse
import os.path as osp
import numpy as np
from ogb.linkproppred import PygLinkPropPredDataset
import random 
from tqdm import tqdm 
import timeit 
import time 
import pandas as pd 

import networkx as nx
from torch_geometric.utils import to_undirected 
from torch_geometric.data import Data
from typing import Dict, Tuple, List, Union
from yacs.config import CfgNode as CN

from graphgps.utility.utils import get_git_repo_root_path, config_device, init_cfg_test
from data_utils.load import load_data_lp
from data_utils.load_data_lp import load_taglp_citationv8, load_graph_citationv8
from data_utils.lcc import use_lcc, get_largest_connected_component
from networkx import from_scipy_sparse_matrix as from_scipy_sparse_array

# adopted from [Google Research - GraphWorld](https://github.com/google-research/graphworld/tree/main/src/graph_world/metrics)
# inspired by https://dl.acm.org/doi/epdf/10.1145/3633778

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.5f} seconds")
        return result
    return wrapper


def plot_adjacency_matrix(G: nx.graph, name: str) -> None:
    """
    Plot the adjacency matrix of a networkx graph.

    Parameters:
    - G: nx.Graph, input graph
    - name: str, output file name

    adopted from  https://stackoverflow.com/questions/22961541/python-matplotlib-plot-sparse-matrix-pattern

    """
    adjacency_matrix = nx.to_numpy_array(G)
    # , dtype=np.bool, nodelist=node_order
    # Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5))  # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none"
                  )
    pyplot.savefig(f'{name}')


def draw_adjacency_matrix(adj: np.array, name: str) -> None:
    """
    Plot the adjacency matrix of a numpy array.

    Parameters:
    - adj: np.array, adjacency matrix
    - name: str, output file name
    """
    fig = pyplot.figure(figsize=(5, 5))  # in inches
    pyplot.imshow(adj,
                  cmap="Greys",
                  interpolation="none")
    pyplot.savefig(f'{name}')



def plot_adj_sparse():
    """plot the adjacency matrix of a sparse matrix"""
    raise NotImplementedError


def plot_coo_matrix(m: coo_matrix, name: str):
    """
    Plot the COO matrix.

    Parameters:
    - m: coo_matrix, input COO matrix
    - name: str, output file name
    """

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m.col, m.row, 's', color='black', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(name)
    return ax


def coo_tensor_to_coo_matrix(coo_tensor: SparseTensor):
    coo = coo_tensor.coo()
    row_indices = coo[0].numpy()
    col_indices = coo[1].numpy()
    values = coo[2].numpy()
    shape = coo_tensor.sizes()

    # Create a scipy coo_matrix
    return coo_matrix((values, (row_indices, col_indices)), shape=shape)



def plot_pos_neg_adj(m_pos: coo_matrix, m_neg: coo_matrix, name: str):
    """
    Plot the COO matrix.

    Parameters:
    - m: coo_matrix, input COO matrix
    - name: str, output file name
    """

    if not isinstance(m_pos, coo_matrix):
        m_pos = coo_matrix(m_pos)
    if not isinstance(m_neg, coo_matrix):
        m_neg = coo_matrix(m_neg)

    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m_neg.col, m_neg.row, 's', color='black', ms=1)
    ax.plot(m_pos.col, m_pos.row, 's', color='blue', ms=1)
    #     ax.set_xlim(0, m_neg.shape[1])
    #     ax.set_ylim(0, m_neg.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(name)
    return ax

# @time_function
def construct_sparse_adj(edge_index) -> coo_matrix:
    """
    Construct a sparse adjacency matrix from an edge index.

    Parameters:
    - edge_index: np.array or tuple, edge index
    """
    # Resource: https://stackoverflow.com/questions/22961541/python-matplotlib-plot-sparse-matrix-pattern

    if type(edge_index) == tuple:
        edge_index = np.concatenate([[edge_index[0].numpy()],
                                     [edge_index[1].numpy()]], axis=0)
    elif type(edge_index) != np.ndarray:
        edge_index.numpy()

    if edge_index.shape[0] > edge_index.shape[1]:
        edge_index = edge_index.T

    rows, cols = edge_index[0, :], edge_index[1, :]
    vals = np.ones_like(rows)
    shape = (edge_index.max() + 1, edge_index.max() + 1)
    m = coo_matrix((vals, (rows, cols)), shape=shape)
    return m


def _gini_coefficient(array: np.ndarray) -> float:
  """Computes the Gini coefficient of a 1-D input array."""
  if array.size == 0:  # pylint: disable=g-explicit-length-test  (numpy arrays have no truth value)
    return 0.0
  array = array.astype(np.float32)
  array += np.finfo(np.float32).eps
  array = np.sort(array)
  n = array.shape[0]
  index = np.arange(1, n + 1)
  return np.sum((2 * index - n  - 1) * array) / (n * np.sum(array))


# @time_function
def _degree_heterogeneity(graph):
    degrees = [degree for _, degree in graph.degree()]
    average_degree = sum(degrees) / len(degrees)
    variance_degree = sum((degree - average_degree) ** 2 for degree in degrees) / len(degrees)
    return math.log10(math.sqrt(variance_degree) / average_degree)


# @time_function
def _avg_degree2(G, avg_degree_dict={}):
    avg_degree_list = []
    for index in range(max(G.nodes) + 1):
        degree = 0
        adj_list = list(G.neighbors(index))
        if adj_list != []:
            for neighbors in adj_list:
                try:
                    degree += avg_degree_dict[neighbors]
                except:
                    degree += 0
            avg_degree_list.append(degree / len(adj_list))

    return np.array(avg_degree_list).mean()


# @time_function
def _avg_degree(G):
    avg_deg = nx.average_neighbor_degree(G)
    # Calculate the sum of all values
    total_sum = sum(avg_deg.values())

    # Calculate the total number of values
    num_values = len(avg_deg)

    # Calculate the average value
    average_value = total_sum / num_values
    return average_value, avg_deg


def _gini_coefficient(array: np.ndarray) -> float:
  """Computes the Gini coefficient of a 1-D input array."""
  if array.size == 0:  # pylint: disable=g-explicit-length-test  (numpy arrays have no truth value)
    return 0.0
  array = array.astype(np.float32)
  array += np.finfo(np.float32).eps
  array = np.sort(array)
  n = array.shape[0]
  index = np.arange(1, n + 1)
  return np.sum((2 * index - n  - 1) * array) / (n * np.sum(array))


def _power_law_estimate(degrees: np.ndarray) -> float:
  degrees = degrees + 1.0
  n = degrees.shape[0]
  return 1.0 + n / np.sum(np.log(degrees / np.min(degrees)))


# @time_function
def _avg_cluster(G, name, scale):
    # name pwc_large scale 100
    
    if name in ['cora', 'pwc_small', 'arxiv_2023', 'pubmed']:
        avg_cluster = nx.average_clustering(G)
    else:
        avg_cluster =  []
        for i, n in tqdm(enumerate(G.nodes())):
            if i % scale == 0:
                avg_cluster.append(nx.clustering(G, n))
    return avg_cluster 

# @time_function
def _avg_shortest_path(G, name, scale):
    
    if name in ['cora', 'pwc_small', 'arxiv_2023']:
        avg_st = nx.average_shortest_path_length(G)
    else:
        all_avg_shortest_paths = []
        for _ in tqdm(range(scale)):
            n1, n2 = random.choices(list(G.nodes()), k=2)
            length = nx.shortest_path_length(G, source=n1, target=n2)
            all_avg_shortest_paths.append(length)
            avg_st = np.array(all_avg_shortest_paths).mean()
    return avg_st


def plot_cc_dist(G: nx.Graph, name: str) -> None:
    connected_components = list(nx.connected_components(G))
    component_sizes = sorted([len(component) for component in connected_components])

    print(f"Number of connected components: {len(connected_components)}")
    print(f"Sizes of connected components: {component_sizes[:10]}")

    plt.figure(figsize=(10, 6))
    plt.plot(component_sizes, '^', markersize=3)
    plt.title('Distribution of Number of Nodes in Each Connected Component')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f"{name}_cc_dist.png")
    

def _degrees(graph: nx.Graph) -> np.ndarray:
  """Returns degrees of the input graph."""
  return np.array([d for _, d in graph.degree()]).astype(np.float32)


def _counts(graph: nx.Graph) -> Dict[str, float]:
  """Returns a dict of count statistics on a graph.

  Arguments:
    graph: a networkx Graph object.
  Returns:
    dict with the following keys and values:
      num_nodes: count of nodes in graph
      num_edges: number of edges in graph
      edge_density: num_edges / {num_nodes choose 2}
  """
  num_nodes = float(graph.number_of_nodes())
  num_edges = float(graph.number_of_edges()) * 2.0  # count both directions
  edge_density = 0.0
  if num_nodes > 1.0:
    edge_density = num_edges / num_nodes / (num_nodes - 1.0)
  return {'num_nodes': num_nodes, 'num_edges': num_edges,
          'edge_density': edge_density}


def _diameter(graph: nx.Graph) -> float:
  """Computes diameter of the graph."""
  if graph.number_of_nodes() == 0:
    return 0.0
  if not nx.is_connected(graph):
    return np.inf
  return float(nx.diameter(graph))



def _largest_connected_component_size(graph: nx.Graph) -> float:
  """Computes the relative size of the largest graph connected component."""
  if graph.number_of_nodes() == 0:
    return 0.0
  if graph.number_of_nodes() == 1:
    return 1.0
  components = nx.connected_components(graph)
  return np.max(list(map(len, components))) / graph.number_of_nodes()


def graph_metrics_nx(graph: nx.Graph, name: str, use_lcc: bool) -> Dict[str, float]:
    """Computes graph metrics on a networkx graph object.

    Arguments:
        graph: networkx graph.
    Returns:
        dict from metric names to metric values.
    """
    result = {'name': f"{name}_{use_lcc}"}
    result.update(_counts(graph))
    degrees = _degrees(graph)
    result['degree_gini'] = _gini_coefficient(degrees)
    
    avg_degree_G, avg_degree_dict = _avg_degree(G)
    avg_degree_G2 = _avg_degree2(graph, avg_degree_dict)
    result['avg_deg'] = avg_degree_G
    result['avg_deg2'] = avg_degree_G2

    result['deg_heterogeneity'] = _degree_heterogeneity(graph)
    result['avg_shortest_path'] = _avg_shortest_path(graph, name, 1000) if nx.is_connected(G) else np.inf
    
    if name in ['pubmed', 'pwc_medium', 'ogbn_arxiv', 'pwc_large', 'ogbn-arxiv', 'citationv8']:
        print(name)
        result['approximate_diameter'] = np.inf 
    else:
        result['approximate_diameter'] = _diameter(graph)
        result['num_triangles'] = float(
        np.sum(list(nx.triangles(graph).values())) / 3.0)
        
    if graph.number_of_nodes() == 0:  # avoid np.mean of empty slice
        result['avg_degree'] = 0.0
        return result
    
    result['avg_degree'] = float(np.mean(degrees))
    core_numbers = np.array(list(nx.core_number(graph).values()))
    result['coreness_eq_1'] = float(np.mean(core_numbers == 1))
    result['coreness_geq_2'] = float(np.mean(core_numbers >= 2))
    result['coreness_geq_5'] = float(np.mean(core_numbers >= 5))
    result['coreness_geq_10'] = float(np.mean(core_numbers >= 10))
    result['coreness_gini'] = float(_gini_coefficient(core_numbers))
    result['avg_cc'] = float(np.mean(list(nx.clustering(graph).values())))
    result['transitivity'] = float(nx.transitivity(graph))
    
    result['cc_size'] = float(_largest_connected_component_size(graph))
    result['power_law_estimate'] = _power_law_estimate(degrees)
    return result


# small_data = ['pwc_small', 'cora', 'arxiv_2023']
# medium_data = ['pwc_medium', 'pubmed']
# large_data = ['citationv8', 'pwc_large']

def plot_all_cc_dist(G, name):
    
    if not nx.is_connected(G):
        print("Graph is not connected.")
        
    else:
        print(f"Graph {name} is connected.")
    plot_cc_dist(G, f"original_{name}")

    
if __name__ == '__main__':

    cfg = init_cfg_test()
    cfg.device = 'cpu'
    cfg = config_device(cfg)

    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--scale', dest='scale', type=int, required=False,
                        help='data name')
    args = parser.parse_args()
    scale = 100
    scale = args.scale 

    plot_cc = False
    graph_metrics = True
    
    gc = []
    for name in ['pwc_medium']:  # 'arxiv_2023', 'pwc_medium', 'ogbn-arxiv', 'pwc_large', 'citationv8', 
        print(f"------ Dataset {name}------")
        
        splits, text, data = load_data_lp[name](cfg.data, False)
        
        start_time = time.time()
        m = construct_sparse_adj(data.edge_index.numpy())
        G = from_scipy_sparse_array(m)
        print(f"Time taken to create graph: {time.time() - start_time} s")
        
        if  plot_cc:
            plot_all_cc_dist(G, name)
        
        if graph_metrics:
            gc.append(graph_metrics_nx(G, name, False))
            print(gc)
            
            gc = pd.DataFrame(gc)
            gc.to_csv(f'{name}_all_graph_metric_False.csv', index=False)
        
        gc = []
        if name in ['cora', 'arxiv_2023', 'citationv8']:
            
            splits, text, data = load_data_lp[name](cfg.data, True)
            
            start_time = time.time()
            m = construct_sparse_adj(data.edge_index.numpy())
            G = from_scipy_sparse_array(m)
            print(f"Time taken to create graph: {time.time() - start_time} s")
            
            if  plot_cc:
                plot_all_cc_dist(G, name)
            
            if graph_metrics:
                gc.append(graph_metrics_nx(G, name, False))
                print(gc)

    gc = pd.DataFrame(gc)
    gc.to_csv(f'{name}_all_graph_metric_True.csv', index=False)
