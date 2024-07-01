
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import wandb 
from typing import Dict, Tuple
from graphgps.train.opt_train import Trainer
from ogb.linkproppred import Evaluator
from torch_geometric.data import Data
from graphgps.utility.utils import config_device, Logger
from yacs.config import CfgNode as CN
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from heuristic.eval import get_metric_score


class Trainer_TFIDF():
    def __init__(self, 
                 FILE_PATH: str, 
                 cfg: CN, 
                 model: torch.nn.Module, 
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 test_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, 
                 run: int, 
                 repeat: int,
                 loggers: Logger, 
                 print_logger: None, 
                 device: int,
                 writer: SummaryWriter):

        
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.if_wandb = True
        # params
        self.model_name = cfg.embedder.type 
        self.data_name = cfg.data.name
        self.FILE_PATH = FILE_PATH 
        self.name_tag = cfg.wandb.name_tag
        self.epochs = cfg.train.epochs
        self.batch_size = cfg.train.batch_size
        
        self.optimizer = optimizer
        self.loggers = loggers
        self.print_logger = print_logger
        self.writer = writer
        self.criterion = nn.CrossEntropyLoss()
        
        report_step = {
                'cora': 1,
                'pubmed': 1,
                'arxiv_2023': 1,
                'ogbn-arxiv': 1,
                'ogbn-products': 1,
        }

        self.report_step = report_step[cfg.data.name]
        
        self.embed_types = ['tfidf', 'word2vec']

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        
        self.run = run
        self.repeat = repeat
        self.results_rank = {}
        self.run_result = {}
        self.train_func = {
            'tfidf': self._train_tfidf,
            # 'word2vec': self.train_word2vec
        }
        self.evaluate_func = {
            'tfidf': self._eval_tfidf,
            # 'word2vec': self.test_word2vec
        }


    def train(self):  
        for epoch in range(1, self.epochs + 1):
            loss = self.train_func[self.model_name]()
            
            self.writer.add_scalar("Loss/train", loss, epoch)

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
                        self.writer.add_scalar(f"Metrics/Train/{key}", result[0], epoch)
                        self.writer.add_scalar(f"Metrics/Valid/{key}", result[1], epoch)
                        self.writer.add_scalar(f"Metrics/Test/{key}", result[2], epoch)                    
                self.print_logger.info('---')
    

    def merge_result_rank(self):
        result_test = self.evaluate_func[self.model_name](self.test_loader)
        result_valid = self.evaluate_func[self.model_name](self.valid_loader)
        result_train = self.evaluate_func[self.model_name](self.train_loader)

        return {
            key: (result_train[key], result_valid[key], result_test[key])
            for key in result_test.keys()
        }
    
    
    @torch.no_grad()
    def _eval_tfidf(self, loader):
        all_preds = []
        all_labels = [] 
        self.model.eval()
        with torch.no_grad():
            for embeddings, labels in loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(embeddings)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                acc = self._acc(all_preds, all_labels)
                
                pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
                result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
                result_mrr.update({'ACC': round(acc, 5)})
            
                return result_mrr
        
        
    def _acc(self, pos_pred, neg_pred):
        if pos_pred.numel() == 0:
            hard_thres = (torch.max(neg_pred).item() + torch.min(neg_pred).item()) / 2
        else:
            hard_thres = (max(torch.max(pos_pred).item(), torch.max(neg_pred).item()) + min(torch.min(pos_pred).item(), torch.min(neg_pred).item())) / 2

        y_pred = torch.zeros_like(pos_pred)
        y_pred[pos_pred >= hard_thres] = 1

        neg_y_pred = torch.ones_like(neg_pred)
        neg_y_pred[neg_pred <= hard_thres] = 0

        y_pred = torch.cat([y_pred, neg_y_pred], dim=0)

        pos_y = torch.ones_like(pos_pred)
        neg_y = torch.zeros_like(neg_pred)
        y = torch.cat([pos_y, neg_y], dim=0)
        y_logits = torch.cat([pos_pred, neg_pred], dim=0)
        # Calculate accuracy    
        return (y == y_pred).float().mean().item()


    def _train_tfidf(self):
        total_loss = 0
        for embeddings, labels in self.train_loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(embeddings)
            
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)