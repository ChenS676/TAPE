import os
import sys
from os.path import abspath, dirname, join
sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))
# standard library imports
import torch
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN

from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from graphgps.network.gsaint import GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler, GraphSAINTEdgeSampler
import torch.nn.functional as F

# external 
from embedding.tune_utils import param_tune_acc_mrr, mvari_str2csv, save_parmet_tune
from heuristic.eval import get_metric_score
from utils import config_device, savepred, Logger
from typing import Dict, Tuple

# Understand, whu is it work
from graphgps.train.opt_train import Trainer


class Trainer_Saint(Trainer):
    def __init__(self, 
                 FILE_PATH: str, 
                 cfg: CN, 
                 model: torch.nn.Module, 
                 emb: torch.nn.Module,
                 data: Data,
                 optimizer: torch.optim.Optimizer, 
                 splits: Dict[str, Data], 
                 run: int, 
                 repeat: int,
                 loggers: Logger, 
                 print_logger: None,  # Ensure this is correctly defined and passed
                 device: torch.device,
                 gsaint=None,
                 batch_size_sampler=None, 
                 walk_length=None, 
                 num_steps=None, 
                 sample_coverage=None):
        # Correctly pass all parameters expected by the superclass constructor
        super().__init__(FILE_PATH, cfg, model, emb, data, optimizer, splits, run, repeat, loggers, print_logger, device)
        
        self.device = device 
        self.print_logger = print_logger                
        self.model = model.to(self.device)
        self.emb = emb
        self.data = data.to(self.device)
        self.model_name = cfg.model.type 
        self.data_name = cfg.data.name
        
        self.FILE_PATH = FILE_PATH 
        self.epochs = cfg.train.epochs
        
        # GSAINT splitting
        if gsaint is not None:
            device_cpu = torch.device('cpu')
            self.test_data  = GraphSAINTRandomWalkSampler(splits['test'].to(device_cpu), batch_size_sampler, walk_length, num_steps, sample_coverage)
            self.train_data = GraphSAINTRandomWalkSampler(splits['train'].to(device_cpu), batch_size_sampler, walk_length, num_steps, sample_coverage)
            self.valid_data = GraphSAINTRandomWalkSampler(splits['valid'].to(device_cpu), batch_size_sampler, walk_length, num_steps, sample_coverage)
        else:
            self.test_data  = splits['test'].to(self.device)
            self.train_data = splits['train'].to(self.device)
            self.valid_data = splits['valid'].to(self.device)

        self.optimizer  = optimizer
    
    def global_to_local(self, edge_label_index, node_idx):

        # Make dict where key: local indexes, value: global indexes
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_idx.tolist())}

        # Create new local edge indexes
        edge_indices = [
            torch.tensor([global_to_local.get(idx.item(), -1) for idx in edge_label_index[0]], dtype=torch.long),
            torch.tensor([global_to_local.get(idx.item(), -1) for idx in edge_label_index[1]], dtype=torch.long)
        ]

        local_indices = torch.stack(edge_indices, dim=0)

        # Since we are going through the entire list of positive/negative indices, 
        # some edges in the subgraph will be marked -1, so we delete them
        valid_indices = (local_indices >= 0).all(dim=0)
        local_indices = local_indices[:, valid_indices]

        return local_indices
    
    def _train_gae(self):
        self.model.train()
        total_loss = total_examples = 0
        for subgraph in self.train_data:
            self.optimizer.zero_grad()
            subgraph = subgraph.to(self.device)

            z = self.model.encoder(subgraph.x, subgraph.edge_index)

            local_pos_indices = self.global_to_local(subgraph.pos_edge_label_index, subgraph.node_index)
            local_neg_indices = self.global_to_local(subgraph.neg_edge_label_index, subgraph.node_index)

            loss = self.model.recon_loss(z, local_pos_indices, local_neg_indices)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * subgraph.num_nodes
            total_examples += subgraph.num_nodes
        
        return total_loss / total_examples
    
    def _train_vgae(self):
        self.model.train()
        total_loss = total_examples = 0
        for subgraph in self.train_data:
            self.optimizer.zero_grad()
            subgraph = subgraph.to(self.device)

            z = self.model(subgraph.x, subgraph.edge_index)

            local_pos_indices = self.global_to_local(subgraph.pos_edge_label_index, subgraph.node_index)
            local_neg_indices = self.global_to_local(subgraph.neg_edge_label_index, subgraph.node_index)

            loss = self.model.recon_loss(z, local_pos_indices, local_neg_indices)
            loss += (1 / subgraph.num_nodes) * self.model.kl_loss()

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * subgraph.num_nodes
            total_examples += subgraph.num_nodes