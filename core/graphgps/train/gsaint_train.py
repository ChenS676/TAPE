import os
import sys
from os.path import abspath, dirname, join
sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))
# standard library imports
import torch
import time
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from typing import Dict, Tuple
import torch.nn.functional as F

# external 
from embedding.tune_utils import param_tune_acc_mrr, mvari_str2csv, save_parmet_tune
from heuristic.eval import get_metric_score
from graphgps.utility.utils import config_device, savepred, Logger
from graphgps.train.opt_train import Trainer
from graphgps.network.gsaint import GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler, GraphSAINTEdgeSampler

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
        
        return total_loss / len(self.train_data)
    
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
        
        return total_loss / len(self.train_data)      

    @torch.no_grad()
    def _evaluate(self, data_loader: Data):
        self.model.eval()
        accumulated_metrics = []

        for data in data_loader:
            data = data.to(self.device)

            local_pos_indices = self.global_to_local(data.pos_edge_label_index, data.node_index)
            local_neg_indices = self.global_to_local(data.neg_edge_label_index, data.node_index)
            
            z = self.model.encoder(data.x, data.edge_index)
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
    def _evaluate_vgae(self, data_loader):
        self.model.eval()
        accumulated_metrics = []

        for data in data_loader:
            data = data.to(self.device)

            local_pos_indices = self.global_to_local(data.pos_edge_label_index, data.node_index)
            local_neg_indices = self.global_to_local(data.neg_edge_label_index, data.node_index)
            
            z = self.model(data.x, data.edge_index)
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
            # wandb.log({'loss': loss, 'epoch': epoch}) #if self.if_wandb else None
            if epoch % 100 == 0: #int(self.report_step) == 0:

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
