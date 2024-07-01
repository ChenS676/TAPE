import seaborn as sns
import matplotlib.pyplot as plt 
import torch
import numpy as np
import pandas as pd

def get_hist(A, full_A, use_heuristic, data, num_nodes):

    # Assuming you have a tensor with node indices
    nodes = torch.arange(num_nodes)
    # Generate pairwise combinations of indices
    pairs = torch.combinations(nodes, r=2).T

    pos_test_pred = eval(use_heuristic)(A, pairs)
    
    data_df = pd.DataFrame({'size': pos_test_pred.numpy()})

    data_df_filtered = data_df[data_df['size'] != 0.0]
    
    plt.figure()
    bin_edges = [0, 1, 3, 10, 25, float('inf')]
    ax = sns.histplot(data=data_df_filtered, kde=False, stat='percent', discrete=True, 
                      color='blue')
    
    # Access the bin edges and heights
    bin_edges = ax.patches[0].get_x()*ax.patches[-1].get_x() + ax.patches[-1].get_width()
    heights = [patch.get_height() for patch in ax.patches]

    # Print bin edges and heights
    print("Bin Edges:", bin_edges)
    print("Heights:", heights)

    plt.title(f'{data}_{use_heuristic}_filtered')
    plt.xlim(1, 40) 
    plt.xlabel('Num of CN')  # Specify x-axis label
    plt.ylabel('Propotion')   # Specify y-axis label
    plt.savefig(f'{data}_{use_heuristic}_filtered.png')
    return 

def get_test_hist(A, test_pos, test_neg, use_heuristic, data, num_nodes):
    
    pos_test_pred = eval(use_heuristic)(A, test_pos)
    neg_test_pred = eval(use_heuristic)(A, test_neg)
    
    bin_edges = [0, 1, 3, 10, 25, float('inf')]
    
    pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    hist, bin_edges = np.histogram(pred, bins=bin_edges)
    
    hist = hist / hist.sum()
    plt.figure(figsize=(10, 8))
    plt.bar([1, 2, 3, 4, 5], hist)
    plt.title(f'{data}_{use_heuristic}_filtered')
    plt.xlabel('Num of CN')  
    plt.ylabel('Proportion')  
    dirpath = '/pfs/work7/workspace/scratch/cc7738-nlp_graph/HeaRT_Mao/benchmarking/HeaRT_small'
    plt.savefig(f'{dirpath}/{data}_{use_heuristic}_test_filtered.png')
    

    plt.figure(figsize=(10, 8))
    sns.barplot(x=[1, 2, 3, 4, 5], y=hist, color='skyblue')
    plt.xticks([0, 1, 2, 3, 4], ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('CN distribution', fontsize=24)
    plt.xlabel('Num of CN', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.savefig(f'{dirpath}/sns{data}_{use_heuristic}_test_filtered.png')
    # data_df = pd.DataFrame({'size': pred.numpy()})
    
    # plt.figure()
    # bin_edges = [0, 1, 3, 10, 25, float('inf')]
    # ax = sns.histplot(data=data_df, kde=False, stat='percent', discrete=True, binrange=(0, 40), 
    #                   color='blue')
    
    # # Access the bin edges and heights
    # bin_edges = ax.patches[0].get_x()*ax.patches[-1].get_x() + ax.patches[-1].get_width()
    # heights = [patch.get_height() for patch in ax.patches]

    # # Print bin edges and heights
    # print("Bin Edges:", bin_edges)
    # print("Heights:", heights)

    # plt.title(f'{data}_{use_heuristic}_filtered')
    # plt.xlabel('Num of CN')  # Specify x-axis label
    # plt.ylabel('Propotion')   # Specify y-axis label
    # plt.savefig(f'{data}_{use_heuristic}_test_filtered.png')
