import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import logging
from itertools import product
from network.gsaint import GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler
from torch_geometric.graphgym.cmd_args import parse_args
from utility.utils import set_cfg, parse_args, get_git_repo_root_path, create_logger
from data_utils.load import load_data_lp
from splits.neighbor_loader import NeighborLoader

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph
import matplotlib.patches as mpatches

def to_network_full_graph(data):
    G = nx.Graph()
    row, col = data.edge_index.numpy()
    G.add_edges_from(zip(row, col), color='black')

    node_color = {node: 'lightgray' for node in range(data.num_nodes)}

    pos_row, pos_col = data.pos_edge_label_index.numpy()
    for u, v in zip(pos_row, pos_col):
        if G.has_edge(u, v):
            G[u][v]['color'] = 'green'
            node_color[u] = 'green'
            node_color[v] = 'green'

    neg_row, neg_col = data.neg_edge_label_index.numpy()
    for u, v in zip(neg_row, neg_col):
        if not G.has_edge(u, v):
            node_color[u] = 'red' if node_color[u] != 'green' and node_color[u] != 'lightgray' else node_color[u]
            node_color[v] = 'red' if node_color[v] != 'green' and node_color[u] != 'lightgray' else node_color[v]

    return G, node_color

def to_network(subgraph_nodes, test_data):
    # Extract the subgraph using the indices from subgraph_nodes
    sub_edge_index, mapping = subgraph(subgraph_nodes, test_data.edge_index, relabel_nodes=True)
    
    G = nx.Graph()

    node_color = {}
    pos_nodes_1 = test_data.pos_edge_label_index.tolist()[0]
    pos_nodes_2 = test_data.pos_edge_label_index.tolist()[1]
    neg_nodes_1 = test_data.neg_edge_label_index.tolist()[0]
    neg_nodes_2 = test_data.neg_edge_label_index.tolist()[1]
    
    for idx, original_idx in enumerate(subgraph_nodes.tolist()):
        if original_idx in pos_nodes_1 or original_idx in pos_nodes_2:
            node_color[idx] = 'green'
        elif original_idx in neg_nodes_1 or original_idx in neg_nodes_2:
            node_color[idx] = 'red'
        else:
            node_color[idx] = 'lightgray'

    for u, v in sub_edge_index.t().tolist():
        if node_color[u] == 'green' and node_color[v] == 'green':
            condition_u = test_data.edge_index[0] == subgraph_nodes[u]
            condition_v = test_data.edge_index[1] == subgraph_nodes[v]
            if torch.any(condition_u & condition_v):
                G.add_edge(u, v)
        else:
            G.add_node(u)
            G.add_node(v)

    return G, node_color

def to_network_ns(subgraph, test_data):
    # Extract the subgraph using the indices from subgraph_nodes
    sub_edge_index = subgraph.edge_index
    
    G = nx.Graph()

    node_color = {}
    pos_nodes_1 = test_data.pos_edge_label_index.tolist()[0]
    pos_nodes_2 = test_data.pos_edge_label_index.tolist()[1]
    neg_nodes_1 = test_data.neg_edge_label_index.tolist()[0]
    neg_nodes_2 = test_data.neg_edge_label_index.tolist()[1]
    
    for idx, original_idx in enumerate(subgraph.n_id):
        if original_idx in pos_nodes_1 or original_idx in pos_nodes_2:
            node_color[idx] = 'green'
        elif original_idx in neg_nodes_1 or original_idx in neg_nodes_2:
            node_color[idx] = 'red'
        else:
            node_color[idx] = 'lightgray'

    for u, v in sub_edge_index.t().tolist():
        if node_color[u] == 'green' and node_color[v] == 'green':
            # print(u, v)
            # print(test_data.edge_index[0], subgraph.n_id[u])
            # condition_u = test_data.edge_index[0] == subgraph[u]
            # condition_v = test_data.edge_index[1] == subgraph[v]
            # if torch.any(condition_u & condition_v):
            G.add_edge(u, v)
        else:
            G.add_node(u)
            G.add_node(v)

    return G, node_color

def plot_graph(G, node_color, idx, name):
    # Set up the plot
    plt.figure(figsize=(15, 15))

    # Determine positions for all nodes using the Spring layout
    pos = nx.spring_layout(G, seed=42)  # Using a seed for reproducibility

    # Extract node colors from the node_color dictionary
    colors = [node_color[node] for node in G.nodes()]

    # Draw the nodes with colors based on the node_color mapping
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=10)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edge_color='gray')  # Assuming a generic color for all edges

    # Create a legend with custom handles
    legend_handles = [
        mpatches.Patch(color='red', label='Negative Nodes'),
        mpatches.Patch(color='green', label='Positive Nodes'),
        mpatches.Patch(color='lightgray', label='Potential Candidates')
    ]
    plt.legend(handles=legend_handles)

    # Save the figure, incorporating the index to distinguish files
    plt.savefig(f'pictures/sampled_subgraph_{idx}_{name}.png')
    plt.close()  # Close the figure to free up memory

def get_loader_RW(data, batch_size, walk_length, num_steps, sample_coverage):
    return GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=walk_length, num_steps=num_steps, sample_coverage=sample_coverage)

def get_loader_ES(data, batch_size, num_steps, sample_coverage):
    return GraphSAINTEdgeSampler(data, batch_size=batch_size, num_steps=num_steps, sample_coverage=sample_coverage)

def get_loader_NS(data, batch_size, num_steps, sample_coverage):
    return GraphSAINTNodeSampler(data, batch_size=batch_size, num_steps=num_steps, sample_coverage=sample_coverage)

def get_loader_Neighbor_Sampler(data, batch_size_sampler, num_neighbors, num_hops):
    return NeighborLoader(
                data=data,
                num_neighbors=[num_neighbors] * num_hops, # tune
                input_nodes=None,
                subgraph_type='bidirectional', # Check it more
                disjoint=False,
                temporal_strategy='uniform', # Check about "last"
                is_sorted=False, # Broke for me everything if is_sorted=True
                                  # After adding this parameter my lists:
                                  # edge_index, pos_edge_label, pos_edge_label_index, neg_edge_label, neg_edge_label_index are EMPTY
                neighbor_sampler=None,
                directed=False,
                replace=False,
                shuffle=False,
                batch_size=batch_size_sampler # tune
            )
    
if __name__ == "__main__":
    print(torch.cuda.is_available())
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    # torch.set_num_threads(cfg.num_threads)
    # Best params: {'batch_size': 64, 'walk_length': 10, 'num_steps': 30, 'sample_coverage': 100, 'accuracy': 0.82129}
    
    # GSAINT
    batch_sizes = [2, 4, 8, 16, 32, 128, 256, 512, 1024]
    walk_lengths = [10]#[10, 15, 20]
    num_steps = [30]#[10, 20, 30]
    sample_coverages = [100]#[50, 100, 150]
    samplers = [get_loader_RW]#, get_loader_ES]
    
    # NS
    # batch_size_samplers = [2, 4, 8]
    # num_neighborss = [10]
    # num_hopss = [5]

    loggers = create_logger(args.repeat)
    for batch_size, walk_length, num_steps, sample_coverage, sampler in product(batch_sizes, walk_lengths, num_steps, sample_coverages, samplers):
    # for batch_size_sampler, num_neighbors, num_hops in product(batch_size_samplers, num_neighborss, num_hopss):
        splits, _, _ = load_data_lp[cfg.data.name](cfg.data)
        
        Sampler = sampler(splits['test'], 
                    batch_size=batch_size,  # batch_size < 32 lead to very sparce graph
                    walk_length=walk_length, 
                    num_steps=num_steps, 
                    sample_coverage=sample_coverage)
        # G, node_color = to_network_full_graph(splits['train'])
        # plot_graph(G, node_color, batch_size, 'cora')
        for idx in range(len(Sampler)-29):
            G, node_color = to_network(Sampler[idx][0], splits['test'])
            plot_graph(G, node_color, batch_size, sampler.__name__)

        # Sampler = get_loader_Neighbor_Sampler(splits['test'],  batch_size_sampler, num_neighbors, num_hops)
        # print(batch_size_sampler)
        # for subgraph in Sampler:
        #     G, node_color = to_network_ns(subgraph, splits['test'])
        #     plot_graph(G, node_color, batch_size_sampler, get_loader_Neighbor_Sampler.__name__) 
        #     break       
            
# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     FILE_PATH = f'{get_git_repo_root_path()}/'

#     args = parse_args()
#     cfg = set_cfg(FILE_PATH, args.cfg_file)
#     cfg.merge_from_list(args.opts)

#     torch.set_num_threads(cfg.num_threads)
#     # Best params: {'batch_size': 64, 'walk_length': 10, 'num_steps': 30, 'sample_coverage': 100, 'accuracy': 0.82129}
#     batch_sizes = [16, 32, 128, 256, 512, 1024]
#     walk_lengths = [10]#[10, 15, 20]
#     num_steps = [30]#[10, 20, 30]
#     sample_coverages = [100]#[50, 100, 150]
#     samplers = [get_loader_NS, get_loader_RW]#, get_loader_ES]
    
#     best_acc = 0
#     best_params = {}
#     flag = True
#     loggers = create_logger(args.repeat)
#     for batch_size, walk_length, num_steps, sample_coverage, sampler in product(batch_sizes, walk_lengths, num_steps, sample_coverages, samplers):
    
#             splits, text = load_data_lp[cfg.data.name](cfg.data)
#             if flag:
#                 lst_args = cfg.model.type.split('_')
#                 cfg.model.type = lst_args[1].upper() # Convert to upper cas
#                 flag=False
            
#             if sampler.__name__ == 'get_loader_RW':
#                 Sampler = sampler(splits['train'], 
#                             batch_size=batch_size,  # batch_size < 32 lead to very sparce graph
#                             walk_length=walk_length, 
#                             num_steps=num_steps, 
#                             sample_coverage=sample_coverage)
#             else:
#                 Sampler = sampler(splits['train'], 
#                             batch_size=batch_size,
#                             num_steps=num_steps, 
#                             sample_coverage=sample_coverage)
            
#             G, node_color = to_network_full_graph(splits['train'])
#             plot_graph(G, node_color, batch_size, 'cora')
#             for idx in range(len(Sampler)-29):
#                 G, node_color = to_network(Sampler[idx][0], splits['test'])
#                 plot_graph(G, node_color, batch_size, sampler.__name__)

            