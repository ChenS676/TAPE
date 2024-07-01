import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit
from data_utils.dataset import CustomLinkDataset

import torch
from core.graphgps.utility.utils import get_git_repo_root_path


FILE_PATH = get_git_repo_root_path() + '/'

def get_raw_text_arxiv_2023(args):
    """
    Retrieves raw text data related to ArXiv 2023.

    Args:
        args: A namespace containing data-related arguments.

    Returns:
        dataset: CustomLinkDataset object with ArXiv 2023 data.
        text: List of strings containing titles and abstracts of ArXiv 2023 papers.
        splits: Dictionary containing train, validation, and test data splits.
    
    Refs:
        refer to load_arxiv_2023.py, load_ogbn_arxiv.py
    """
    
    undirected = args.data.undirected
    include_negatives = args.data.include_negatives
    val_pct = args.data.val_pct
    test_pct = args.data.test_pct
    split_labels = args.data.split_labels
    
    data = torch.load(FILE_PATH + 'core/dataset/arxiv_2023/graph.pt') # Missed folder name core/
    
    # data.edge_index = data.adj_t.to_symmetric()
    text = None

    df = pd.read_csv(FILE_PATH + 'core/dataset/arxiv_2023_orig/paper_info.csv') # Missed folder name core/
    text = []
    for ti, ab in zip(df['title'], df['abstract']):
        text.append(f'Title: {ti}\nAbstract: {ab}')
        # text.append((ti, ab))
        
    dataset = CustomLinkDataset('./generated_dataset', 'arxiv_2023', transform=T.NormalizeFeatures())
    dataset._data = data
    
    undirected = data.is_undirected()
    
    transform = RandomLinkSplit(is_undirected=undirected, 
                                num_val=val_pct,
                                num_test=test_pct,
                                add_negative_train_samples=include_negatives, 
                                split_labels=split_labels)
    
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    

    return dataset, text, splits


