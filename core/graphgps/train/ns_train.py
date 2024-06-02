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
from graphgps.splits.neighbor_loader import NeighborLoader
from graphgps.utility.NS_utils import degree


report_step = {
    'cora': 100,
    'pubmed': 1,
    'arxiv_2023': 100,
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
                 batch_size_sampler: int,
                 num_neighbors: int,
                 num_hops: int):
        
        self.device = device
        self.model = model.to(self.device)
        self.emb = emb

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

        self.data = data
        self.optimizer = optimizer
        self.loggers = loggers
        self.print_logger = print_logger
        self.report_step = report_step[cfg.data.name]
        model_types = ['VGAE', 'GAE', 'GAT', 'GraphSage']
        self.train_func = {model_type: self._train_gae if model_type in ['GAE', 'GAT', 'GraphSage', 'GNNStack'] else self._train_vgae for model_type in model_types}
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate if model_type in ['GAE', 'GAT', 'GraphSage', 'GNNStack'] else self._evaluate_vgae for model_type in model_types}
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

    def _train_gae(self):
        self.model.train()
        total_loss = total_examples = 0
        for subgraph in self.train_data:
            self.optimizer.zero_grad()
            
            #if self.is_disjoint(subgraph.edge_index, subgraph.num_nodes):
            #    transform = VirtualNode()
            #    subgraph = transform(subgraph)

            subgraph = subgraph.to(self.device)

            z = self.model.encoder(subgraph.x, subgraph.edge_index)
            
            local_pos_indices = self.global_to_local(subgraph.pos_edge_label_index, subgraph.n_id)
            local_neg_indices = self.global_to_local(subgraph.neg_edge_label_index, subgraph.n_id)
            
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

            #if self.is_disjoint(subgraph.edge_index, subgraph.num_nodes):
            #    transform = VirtualNode()
            #    subgraph = transform(subgraph)

            subgraph = subgraph.to(self.device)

            z = self.model(subgraph.x, subgraph.edge_index)

            local_pos_indices = self.global_to_local(subgraph.pos_edge_label_index, subgraph.n_id)
            local_neg_indices = self.global_to_local(subgraph.neg_edge_label_index, subgraph.n_id)

            loss = self.model.recon_loss(z, local_pos_indices, local_neg_indices)
            loss += (1 / subgraph.num_nodes) * self.model.kl_loss()

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * subgraph.num_nodes
            total_examples += subgraph.num_nodes
        
        return total_loss / total_examples

    @torch.no_grad()
    def _evaluate(self, test_data):
        self.model.eval()
        accumulated_metrics = []

        for subgraph in test_data:
            #if self.is_disjoint(subgraph.edge_index, subgraph.num_nodes):
            #   transform = VirtualNode()
            #   subgraph = transform(subgraph)

            subgraph = subgraph.to(self.device)

            local_pos_indices = self.global_to_local(subgraph.pos_edge_label_index, subgraph.n_id)
            local_neg_indices = self.global_to_local(subgraph.neg_edge_label_index, subgraph.n_id)
            
            z = self.model.encoder(subgraph.x, subgraph.edge_index)
            pos_pred = self.model.decoder(z, local_pos_indices)
            neg_pred = self.model.decoder(z, local_neg_indices)
            y_pred = torch.cat([pos_pred, neg_pred], dim=0)

            hard_thres = (y_pred.max() + y_pred.min())/2

            pos_y = z.new_ones(local_pos_indices.size(1))
            neg_y = z.new_zeros(local_neg_indices.size(1)) 
            y = torch.cat([pos_y, neg_y], dim=0)
            
            y_pred[y_pred >= hard_thres] = 1
            y_pred[y_pred < hard_thres] = 0
            acc = torch.sum(y == y_pred) / len(y)

            pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
            result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
            result_mrr.update({'acc': round(acc.item(), 5)})
            accumulated_metrics.append(result_mrr)

        # Aggregate results from accumulated_metrics
        aggregated_results = {}
        for result in accumulated_metrics:
            for key, value in result.items():
                if key in aggregated_results:
                    aggregated_results[key].append(value)
                else:
                    aggregated_results[key] = [value]

        # Calculate average results
        averaged_results = {key: sum(values) / len(values) for key, values in aggregated_results.items()}

        return averaged_results

    @torch.no_grad()
    def _evaluate_vgae(self, test_data):
        self.model.eval()
        accumulated_metrics = []
        for subgraph in test_data:
#            if self.is_disjoint(subgraph.edge_index, subgraph.num_nodes):
#                transform = VirtualNode()
#                subgraph = transform(subgraph)

            subgraph = subgraph.to(self.device)

            local_pos_indices = self.global_to_local(subgraph.pos_edge_label_index, subgraph.n_id)
            local_neg_indices = self.global_to_local(subgraph.neg_edge_label_index, subgraph.n_id)
            
            z = self.model(subgraph.x, subgraph.edge_index)
            pos_pred = self.model.decoder(z, local_pos_indices)
            neg_pred = self.model.decoder(z, local_neg_indices)
            y_pred = torch.cat([pos_pred, neg_pred], dim=0)

            hard_thres = (y_pred.max() + y_pred.min())/2

            pos_y = z.new_ones(local_pos_indices.size(1))
            neg_y = z.new_zeros(local_neg_indices.size(1)) 
            y = torch.cat([pos_y, neg_y], dim=0)
            
            y_pred[y_pred >= hard_thres] = 1
            y_pred[y_pred < hard_thres] = 0
            acc = torch.sum(y == y_pred) / len(y)
            
            pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
            result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
            result_mrr.update({'acc': round(acc.item(), 5)})
            accumulated_metrics.append(result_mrr)

        # Aggregate results from accumulated_metrics
        aggregated_results = {}
        for result in accumulated_metrics:
            for key, value in result.items():
                if key in aggregated_results:
                    aggregated_results[key].append(value)
                else:
                    aggregated_results[key] = [value]

        # Calculate average results
        averaged_results = {key: sum(values) / len(values) for key, values in aggregated_results.items()}

        return averaged_results
    
    def train(self):  
        
        for epoch in range(1, self.epochs + 1):
            loss = self.train_func[self.model_name]()
            print(epoch, ': ', loss)
            # wandb.log({'loss': loss, 'epoch': epoch}) #if self.if_wandb else None
            if epoch % 25 == 0: #int(self.report_step) == 0:

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
                    # wandb.log({f"Metrics/train_{key}": train_hits})
                    # wandb.log({f"Metrics/valid_{key}": valid_hits})
                    # wandb.log({f"Metrics/test_{key}": test_hits})
                    
                self.print_logger.info('---')

        for _ in range(1):
            start_train = time.time() 
            self.train_func[self.model_name]()
            self.run_result['train_time'] = time.time() - start_train
            self.evaluate_func[self.model_name](self.test_data)
            self.run_result['eval_time'] = time.time() - start_train


