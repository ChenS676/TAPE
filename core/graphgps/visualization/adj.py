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
from ogb.nodeproppred import NodePropPredDataset
from scipy.sparse import csc_array

from data_utils.load import load_data_nc

import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import matspy as spy  # https://github.com/alugowski/matspy
import math
import argparse


def calculate_heterogeneity(graph):
    degrees = [degree for node, degree in graph.degree()]
    average_degree = sum(degrees) / len(degrees)
    variance_degree = sum((degree - average_degree) ** 2 for degree in degrees) / len(degrees)
    return math.log10(math.sqrt(variance_degree) / average_degree)


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


import os.path as osp


def compare_adj(data_name, data_edges):
    """_summary_

    Args:
          data_name (_type_): _description_
          data_edges (_type_): _description_
    """

    path = osp.join(osp.dirname(osp.realpath(__file__)), './dataset')
    dataset = Planetoid(path, data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    # check if loaded dataset is the same as Planetoid
    sorted_data_edges = np.sort(data_edges, axis=0)
    sorted_data_index = np.sort(data.edge_index.numpy(), axis=0)
    are_datasets_equal = np.array_equal(sorted_data_edges, sorted_data_index)

    print(sorted_data_edges)
    print(sorted_data_index)
    print(f'Are datasets equal? {are_datasets_equal}')
    # original Planetoid dataset
    adj = to_torch_coo_tensor(data.edge_index)
    adj = adj.to_dense().numpy()
    G = nx.from_numpy_array(adj, create_using=nx.Graph)

    plot_adjacency_matrix(G, f'{data_name}_data.edge_index.png')

    # TAG
    adj = to_torch_coo_tensor(torch.tensor(data_edges))
    adj = adj.to_dense().numpy()
    plot_adjacency_matrix(G, f'{data_name}_data_edges.png')


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
from torch_sparse import SparseTensor

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


def avg_degree2(G, avg_degree_dict={}):
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


def avg_degree(G):
    avg_deg = nx.average_neighbor_degree(G)
    # Calculate the sum of all values
    total_sum = sum(avg_deg.values())

    # Calculate the total number of values
    num_values = len(avg_deg)

    # Calculate the average value
    average_value = total_sum / num_values
    return average_value, avg_deg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        help='Dataset name.')
    args = parser.parse_args()

    scale = 100000
    name_list = ['ogbn-arxiv', 'arxiv_2023', 'cora', 'pubmed']

    for name in name_list:

        if name == 'ogbn-products':
            dataset = NodePropPredDataset(name)
            edge_index = dataset[0][0]['edge_index']

            # edge index to sparse matrix
            edge_index = edge_index[:, ::scale]
            m = construct_sparse_adj(edge_index)

            G = nx.from_scipy_sparse_array(m)
            plot_coo_matrix(m, f'{name}_data_edges.png')

            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"{name}_data_edges_spy.png", bbox_inches='tight')

            data, text = load_data_nc[name]()
            m = construct_sparse_adj(data.edge_index.coo())
            plot_coo_matrix(m, f'{name}_data_index.png')

            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"{name}_data_index_spy.png", bbox_inches='tight')

            heterogeneity = calculate_heterogeneity(G)
            print(f"{name}, heterogeneity: {heterogeneity}. num_node: {dataset[0].num_node}")

        if name == 'ogbn-arxiv':
            dataset = NodePropPredDataset(name)
            edge_index = dataset[0][0]['edge_index']

            m = construct_sparse_adj(edge_index[:, ::2])
            G = nx.from_scipy_sparse_array(m)

            plot_coo_matrix(m, f'{name}_data_edges.png')

            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"{name}_data_edges_spy.png", bbox_inches='tight')

            heterogeneity = calculate_heterogeneity(G)
            num_nodes = dataset[0][0]['num_nodes']
            num_edges = dataset[0][0]['edge_index'].shape[1]
            avg_degree_arithmetic = int(num_edges / num_nodes)
            avg_degree_G, avg_degree_dict = avg_degree(G)
            avg_degree_G2 = avg_degree2(G, avg_degree_dict)
            print(f"{name}, heterogeneity: {heterogeneity}. num_node: {num_nodes}, num_edges: {num_edges}, \
                      avg degree arithmetic {avg_degree_arithmetic},  \
                      avg degree G {avg_degree_G}, avg degree G2 {avg_degree_G2}, clustering {nx.average_clustering(G)}.")

        if name == 'arxiv_2023':
            data, text = load_data_nc[name]()
            m = construct_sparse_adj(data.edge_index.numpy())
            G = nx.from_scipy_sparse_array(m)

            plot_coo_matrix(m, f'{name}_data_index.png')

            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"{name}_data_index_spy.png", bbox_inches='tight')

            heterogeneity = calculate_heterogeneity(G)
            num_nodes = data.num_nodes
            num_edges = data.edge_index.shape[1]
            avg_degree_arithmetic = int(num_edges / num_nodes)
            avg_degree_G, avg_degree_dict = avg_degree(G)
            avg_degree_G2 = avg_degree2(G, avg_degree_dict)
            print(f"{name}, heterogeneity: {heterogeneity}. num_node: {num_nodes}, num_edges: {num_edges}, \
                      avg degree arithmetic {avg_degree_arithmetic},  \
                      avg degree G {avg_degree_G}, avg degree G2 {avg_degree_G2}, clustering {nx.average_clustering(G)}.")

        if name == 'pubmed':
            data, text = load_data_nc[name](use_mask=False)
            G = nx.from_scipy_sparse_array(m)

            compare_adj(name, data.edge_index.numpy())
            m = construct_sparse_adj(data.edge_index.numpy())
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"{name}_data_index_spy.png", bbox_inches='tight')

            heterogeneity = calculate_heterogeneity(G)
            num_nodes = data.num_nodes
            num_edges = data.edge_index.shape[1]
            avg_degree_arithmetic = int(num_edges / num_nodes)
            avg_degree_G, avg_degree_dict = avg_degree(G)
            avg_degree_G2 = avg_degree2(G, avg_degree_dict)
            print(f"{name}, heterogeneity: {heterogeneity}. num_node: {num_nodes}, num_edges: {num_edges}, \
                      avg degree arithmetic {avg_degree_arithmetic},  \
                      avg degree G {avg_degree_G}, avg degree G2 {avg_degree_G2}, clustering {nx.average_clustering(G)}.")