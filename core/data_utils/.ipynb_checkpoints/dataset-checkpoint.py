import torch
import os.path as osp
import torch
from torch_geometric.data import Data, Dataset
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)


# Create torch dataset
class CustomPygDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

    
class CustomLinkDataset(Dataset):
    def __init__(self,  root, name, transform=None, pre_transform=None):
        super(CustomLinkDataset, self).__init__(root, transform, pre_transform)
        self.name  = name
        self.root = root
        
    @property
    def raw_file_names(self):
        # If you have raw files to download, specify their names here
        return []

    @property
    def processed_file_names(self):
        # If you have processed files, specify their names here
        return []

    def download(self):
        # Download raw data here
        pass

    def process(self):
        # Process raw data and save it to processed files
        pass

    def len(self):
        # Return the number of graphs in the dataset
        return 1

    def get(self, idx):
        # Load and return the graph at index idx
        pass
