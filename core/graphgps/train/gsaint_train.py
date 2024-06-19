import os
import sys
from os.path import abspath, dirname, join
sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))
# standard library imports
import torch
import time
import wandb
import pandas as pd
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from typing import Dict, Tuple
import torch.nn.functional as F

# external 
from embedding.tune_utils import param_tune_acc_mrr, mvari_str2csv, save_parmet_tune
from heuristic.eval import get_metric_score
from graphgps.utility.utils import config_device, Logger
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
                 if_wandb: bool,
                 gsaint=None):
        # Correctly pass all parameters expected by the superclass constructor
        super().__init__(FILE_PATH, cfg, model, emb, data, optimizer, splits, run, repeat, loggers, print_logger, device)
        
        self.device = device 
        self.if_wandb = if_wandb
        self.print_logger = print_logger                
        self.model = model.to(self.device)
        self.emb = emb
        self.data = data.to(self.device)
        self.model_name = cfg.model.type 
        self.data_name = cfg.data.name
        
        report_step = {
            'cora': 10, #100,
            'pubmed': 5,
            'arxiv_2023': 5, #100,
            'ogbn-arxiv': 1,
            'ogbn-products': 1,
            'synthetic': 2,
        }
        self.report_step = report_step[cfg.data.name]
        
        self.FILE_PATH = FILE_PATH 
        self.epochs = cfg.train.epochs
        
        # GSAINT splitting
        if gsaint is not None:
            self.test_data  = gsaint(splits['test'].to('cpu'),  cfg.sampler.gsaint.batch_size_sampler, cfg.sampler.gsaint.walk_length, cfg.sampler.gsaint.num_steps, cfg.sampler.gsaint.sample_coverage)
            self.train_data = gsaint(splits['train'].to('cpu'), cfg.sampler.gsaint.batch_size_sampler, cfg.sampler.gsaint.walk_length, cfg.sampler.gsaint.num_steps, cfg.sampler.gsaint.sample_coverage)
            self.valid_data = gsaint(splits['valid'].to('cpu'), cfg.sampler.gsaint.batch_size_sampler, cfg.sampler.gsaint.walk_length, cfg.sampler.gsaint.num_steps, cfg.sampler.gsaint.sample_coverage)
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
        total_loss = 0
        for subgraph in self.train_data:
            self.optimizer.zero_grad()

            x = subgraph.x.to(self.device)
            edge_index = subgraph.edge_index.to(self.device)
            z = self.model.encoder(x, edge_index)

            local_pos_indices = self.global_to_local(subgraph.pos_edge_label_index, subgraph.node_index)
            local_neg_indices = self.global_to_local(subgraph.neg_edge_label_index, subgraph.node_index)

            loss = self.model.recon_loss(z, local_pos_indices.to(self.device), local_neg_indices.to(self.device))

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * subgraph.num_nodes
        
        return total_loss / len(self.train_data)
    
    def _train_vgae(self):
        self.model.train()
        total_loss = 0
        for subgraph in self.train_data:
            self.optimizer.zero_grad()
            
            x = subgraph.x.to(self.device)
            edge_index = subgraph.edge_index.to(self.device)
            z = self.model.encoder(x, edge_index)

            local_pos_indices = self.global_to_local(subgraph.pos_edge_label_index, subgraph.node_index)
            local_neg_indices = self.global_to_local(subgraph.neg_edge_label_index, subgraph.node_index)

            loss = self.model.recon_loss(z, local_pos_indices.to(self.device), local_neg_indices.to(self.device))
            loss += (1 / subgraph.num_nodes) * self.model.kl_loss()

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * subgraph.num_nodes
        
        return total_loss / len(self.train_data)      

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
    def _evaluate(self, data_loader: Data):
        self.model.eval()
        accumulated_metrics = []

        for data in data_loader:

            local_pos_indices = self.global_to_local(data.pos_edge_label_index, data.node_index)
            local_neg_indices = self.global_to_local(data.neg_edge_label_index, data.node_index)
            
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            z = self.model.encoder(x, edge_index)
            
            pos_pred = self.test_edge(z, local_pos_indices.to(self.device))
            neg_pred = self.test_edge(z, local_neg_indices.to(self.device))
            
            acc = self._acc(pos_pred, neg_pred)

            pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
            result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
            result_mrr.update({'ACC': round(acc, 5)})
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

            local_pos_indices = self.global_to_local(data.pos_edge_label_index, data.node_index)
            local_neg_indices = self.global_to_local(data.neg_edge_label_index, data.node_index)
            
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            z = self.model.encoder(x, edge_index)
            
            pos_pred = self.test_edge(z, local_pos_indices.to(self.device))
            neg_pred = self.test_edge(z, local_neg_indices.to(self.device))
            
            acc = self._acc(pos_pred, neg_pred)

            pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
            result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
            result_mrr.update({'ACC': round(acc, 5)})
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
            x = subgraph.x.to(self.device)
            edge_index = subgraph.edge_index.to(self.device)
            if self.model_name == 'VGAE':
                z = self.model(x, edge_index)
                
            elif self.model_name in ['GAE', 'GAT', 'GraphSage', 'GAT_Variant', 
                                    'GCN_Variant', 'SAGE_Variant', 'GIN_Variant']:
                z = self.model.encoder(x, edge_index)
            
            pos_edge_label_index = self.global_to_local(subgraph.pos_edge_label_index, subgraph.n_id)
            neg_edge_label_index = self.global_to_local(subgraph.neg_edge_label_index, subgraph.n_id)

            pos_pred, pos_edge_index = self.save_eval_edge_pred(z, pos_edge_label_index.to(self.device))
            neg_pred, neg_edge_index = self.save_eval_edge_pred(z, neg_edge_label_index.to(self.device))
            
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
    

        for _ in range(1):
            start_train = time.time() 
            self.train_func[self.model_name]()
            self.run_result['train_time'] = time.time() - start_train
            self.evaluate_func[self.model_name](self.test_data)
            self.run_result['eval_time'] = time.time() - start_train
        
        self._save_err_heart(self.test_data, 'test')
        self._save_err_heart(self.valid_data, 'valid')
        self._save_err_heart(self.train_data,  'train')
