import os
import sys
from os.path import abspath, dirname, join
sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

import torch
import time
import wandb 
import numpy as np
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from yacs.config import CfgNode as CN
from tqdm import tqdm 
from torch_geometric.data import Data
from typing import Dict, Tuple
from torch_geometric.transforms.virtual_node import VirtualNode

from embedding.tune_utils import mvari_str2csv, save_parmet_tune
from heuristic.eval import get_metric_score
from graphgps.utility.utils import config_device, Logger
from graphgps.train.opt_train import Trainer
from torch.utils.data import DataLoader
from graphgps.splits.neighbor_loader import NeighborLoader
from graphgps.utility.NS_utils import degree
from torch_sparse import SparseTensor
import pandas as pd


report_step = {
    'cora': 2, #100,
    'pubmed': 5,
    'arxiv_2023': 5, #100,
    'ogbn-arxiv': 1,
    'ogbn-products': 1,
}

def data_loader(data, batch_size_sampler, num_neighbors, num_hops):
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

class Trainer_NS(Trainer):
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
                 print_logger: None, 
                 device: int,
                 if_wandb: bool,
                 batch_size_sampler: int,
                 num_neighbors: int,
                 num_hops: int,
                 virtual_node_flag: bool):
        
        self.device = device
        self.if_wandb = if_wandb
        self.model = model.to(self.device)
        self.run_dir = cfg.run_dir
        self.emb = emb

        if if_wandb:
            self.step = 0
        # params
        self.model_name = cfg.model.type 
        self.data_name = cfg.data.name
        self.FILE_PATH = FILE_PATH 
        self.name_tag = cfg.wandb.name_tag
        self.epochs = cfg.train.epochs
        self.batch_size = cfg.train.batch_size
        
        self.test_data  = data_loader(splits['test'],  batch_size_sampler, num_neighbors, num_hops)
        self.train_data = data_loader(splits['train'], batch_size_sampler, num_neighbors, num_hops)
        self.valid_data = data_loader(splits['valid'], batch_size_sampler, num_neighbors, num_hops)
        self.virtual_node_flag = virtual_node_flag # Turn on/off the adding of Virtual Node in graph

        self.data = data
        self.optimizer = optimizer
        self.loggers = loggers
        self.print_logger = print_logger
        self.report_step = report_step[cfg.data.name]
        self.model_types = ['VGAE', 'GAE', 'GAT', 'GraphSage']
        
        self.train_func = {model_type: self._train_heart for model_type in self.model_types}
        self.test_func = {model_type: self._eval_heart  for model_type in self.model_types}
        self.evaluate_func = {model_type: self._eval_heart  for model_type in self.model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        
        self.run = run
        self.repeat = repeat
        self.results_rank = {}
        self.run_result = {}

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
    
    def is_disjoint(self, edge_index, num_nodes):
        node_degree = degree(edge_index[0], num_nodes) + degree(edge_index[1], num_nodes)
        return (node_degree == 0).any()

    def _train_heart(self):
        pos_train_weight = None

        for subgraph in self.train_data:  # Drop the last incomplete batch if dataset size is not divisible by batch size
            
            if self.emb is None: 
                x = subgraph.x
                emb_update = 0
            else: 
                x = self.emb.weight
                emb_update = 1
            
            self.optimizer.zero_grad()
            
            if self.virtual_node_flag:
                if self.is_disjoint(subgraph.edge_index, subgraph.num_nodes):
                   transform = VirtualNode()
                   subgraph = transform(subgraph)
            
            num_nodes = x.size(0)    
            batch_edge_index = subgraph.edge_index.to(self.device)
            x = x.to(self.device) 
            
            pos_edges = self.global_to_local(subgraph.pos_edge_label_index.to('cpu'), subgraph.n_id)
            neg_edges = self.global_to_local(subgraph.neg_edge_label_index.to('cpu'), subgraph.n_id)
            
            pos_edges = pos_edges.to(self.device)
            neg_edges = neg_edges.to(self.device)

            if self.model_name == 'VGAE':
                h = self.model(x, batch_edge_index)
                loss = self.model.recon_loss(h, pos_edges, neg_edges)
                loss = loss + (1 / num_nodes) * self.model.kl_loss()
            elif self.model_name in ['GAE', 'GAT', 'GraphSage', 'GAT_Variant', 
                                 'GCN_Variant', 'SAGE_Variant', 'GIN_Variant']:
                h = self.model.encoder(x, batch_edge_index)
                loss = self.model.recon_loss(h, pos_edges, neg_edges)                
            loss.backward()
            
            if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()           
        
        #self.scheduler.step()
        return loss.item() 
    
    @torch.no_grad()
    def test_edge(self, h, edge_index):
        preds = []
        edge_index = edge_index.t()
    
        # sourcery skip: no-loop-in-tests
        for perm in DataLoader(range(edge_index.size(0)), self.batch_size):
            edge = edge_index[perm].t()
            preds += [self.model.decoder(h[edge[0]], h[edge[1]]).cpu()]

        return torch.cat(preds, dim=0)


    @torch.no_grad()
    def _eval_heart(self, graph: Data):
        self.model.eval()

        for subgraph in graph:
            if self.virtual_node_flag:
                if self.is_disjoint(subgraph.edge_index, subgraph.num_nodes):
                   transform = VirtualNode()
                   subgraph = transform(subgraph)
   
            pos_edge_label_index = self.global_to_local(subgraph.pos_edge_label_index.to('cpu'), subgraph.n_id)
            neg_edge_label_index = self.global_to_local(subgraph.neg_edge_label_index.to('cpu'), subgraph.n_id)

            batch_edge_index = subgraph.edge_index.to(self.device)
            x = subgraph.x.to(self.device) 

            if self.model_name == 'VGAE':
                z = self.model(x, batch_edge_index)
                
            elif self.model_name in ['GAE', 'GAT', 'GraphSage', 'GAT_Variant', 
                                    'GCN_Variant', 'SAGE_Variant', 'GIN_Variant']:
                z = self.model.encoder(x, batch_edge_index)
            
            pos_pred = self.test_edge(z, pos_edge_label_index.to(self.device))
            neg_pred = self.test_edge(z, neg_edge_label_index.to(self.device))
            
            acc = self._acc(pos_pred, neg_pred)
            
            pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
            result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred.squeeze(), neg_pred.squeeze())
            result_mrr.update({'ACC': round(acc, 5)})
            return result_mrr
    
    @torch.no_grad()
    def save_eval_edge_pred(self, h, edge_index):
        preds = []
        edge_index = edge_index.t()
        edge_index_list = []
        # sourcery skip: no-loop-in-tests
        for perm  in DataLoader(range(edge_index.size(0)), self.batch_size):
            edge = edge_index[perm].t()
            edge_index_list.append(edge)
            preds += [self.model.decoder(h[edge[0]], h[edge[1]]).cpu()]

        return torch.cat(preds, dim=0), torch.cat(edge_index_list, dim=1)
    
    @torch.no_grad()
    def _save_err_heart(self, graph: Data, mode):
        self.model.eval()

        for subgraph in graph:
            if self.model_name == 'VGAE':
                z = self.model(subgraph.x, subgraph.edge_index)
                
            elif self.model_name in ['GAE', 'GAT', 'GraphSage', 'GAT_Variant', 
                                    'GCN_Variant', 'SAGE_Variant', 'GIN_Variant']:
                z = self.model.encoder(subgraph.x, subgraph.edge_index)
            
            pos_edge_label_index = self.global_to_local(subgraph.pos_edge_label_index, subgraph.n_id)
            neg_edge_label_index = self.global_to_local(subgraph.neg_edge_label_index, subgraph.n_id)

            pos_pred, pos_edge_index = self.save_eval_edge_pred(z, pos_edge_label_index)
            neg_pred, neg_edge_index = self.save_eval_edge_pred(z, neg_edge_label_index)
            
            self._acc_error_save(pos_pred, pos_edge_index, neg_pred, neg_edge_index, mode)

    
    def _acc_error_save(self, 
                        pos_pred: torch.tensor, 
                        pos_edge_index: torch.tensor, 
                        neg_pred: torch.tensor, 
                        neg_edge_index: torch.tensor, 
                        mode: str) -> None:
        
        hard_thres = (max(torch.max(pos_pred).item(), torch.max(neg_pred).item()) + min(torch.min(pos_pred).item(), torch.min(neg_pred).item())) / 2

        y_pred = torch.zeros_like(pos_pred)
        y_pred[pos_pred >= hard_thres] = 1

        neg_y_pred = torch.ones_like(neg_pred)
        neg_y_pred[neg_pred <= hard_thres] = 0

        # Concatenate the positive and negative predictions
        y_pred = torch.cat([y_pred, neg_y_pred], dim=0)

        # Initialize ground truth labels
        pos_y = torch.ones_like(pos_pred)
        neg_y = torch.zeros_like(neg_pred)
        gr = torch.cat([pos_y, neg_y], dim=0)

        data = {
            "edge_index0": torch.cat([pos_edge_index, neg_edge_index], dim=1)[0].cpu(),
            "edge_index1": torch.cat([pos_edge_index, neg_edge_index], dim=1)[1].cpu(),
            "pred": y_pred.squeeze().cpu(),
            "gr": gr.squeeze().cpu(),
        }
        df = pd.DataFrame(data)
        df.to_csv(f'{self.run_dir}/{self.data_name}_{mode}pred_gr_last_epoch.csv', index=False)

    def train(self):  
        for epoch in range(1, self.epochs + 1):
            loss = self.train_func[self.model_name]()
            print(f"Epoch: {epoch}, Loss: {loss}")
            if self.if_wandb:
                wandb.log({"Epoch": epoch}, step=self.step)
                wandb.log({'loss': loss}, step=self.step) 
                #wandb.log({"lr": self.scheduler.get_lr()}, step=self.step)
            if epoch % int(self.report_step) == 0:

                self.results_rank = self.merge_result_rank()
                
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_train: {loss:.4f}, AUC: {self.results_rank["AUC"][0]:.4f}, AP: {self.results_rank["AP"][0]:.4f}, MRR: {self.results_rank["MRR"][0]:.4f}, Hit@10 {self.results_rank["Hits@10"][0]:.4f}')
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_valid: {loss:.4f}, AUC: {self.results_rank["AUC"][1]:.4f}, AP: {self.results_rank["AP"][1]:.4f}, MRR: {self.results_rank["MRR"][1]:.4f}, Hit@10 {self.results_rank["Hits@10"][1]:.4f}')               
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_test: {loss:.4f}, AUC: {self.results_rank["AUC"][2]:.4f}, AP: {self.results_rank["AP"][2]:.4f}, MRR: {self.results_rank["MRR"][2]:.4f}, Hit@10 {self.results_rank["Hits@10"][2]:.4f}')               
                    
                for key, result in self.results_rank.items():
                    self.loggers[key].add_result(self.run, result)
                    
                    train_hits, valid_hits, test_hits = result
                    self.print_logger.info(
                        f'Run: {self.run + 1:02d}, Key: {key}, '
                        f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_hits:.2f}, Valid: {100 * valid_hits:.2f}, Test: {100 * test_hits:.2f}%')
                    
                    if self.if_wandb:
                        wandb.log({f"Metrics/train_{key}": train_hits}, step=self.step)
                        wandb.log({f"Metrics/valid_{key}": valid_hits}, step=self.step)
                        wandb.log({f"Metrics/test_{key}": test_hits}, step=self.step)
                    
                self.print_logger.info('---')
                
            if self.if_wandb:
                self.step += 1
    
    
    def finalize(self):
        for _ in range(1):
            start_train = time.time() 
            self.train_func[self.model_name]()
            self.run_result['train_time'] = time.time() - start_train
            self.evaluate_func[self.model_name](self.test_data)
            self.run_result['eval_time'] = time.time() - start_train
        
        self._save_err_heart(self.test_data, 'test')
        self._save_err_heart(self.valid_data, 'valid')
        self._save_err_heart(self.train_data,  'train')


