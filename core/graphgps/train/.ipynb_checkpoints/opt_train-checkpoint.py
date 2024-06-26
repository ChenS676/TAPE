import os
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

import torch
import wandb
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from yacs.config import CfgNode as CN

from torch_geometric.data import Data
from embedding.tune_utils import param_tune_acc_mrr
from heuristic.eval import get_metric_score
from utils import config_device
from typing import Dict, Tuple
from utils import Logger

class Trainer():
    def __init__(self, 
                 FILE_PATH: str, 
                 cfg: CN, 
                 model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 splits: Dict[str, Data], 
                 run: int, 
                 repeat: int,
                 loggers: Logger):
        
        self.device = config_device(cfg)
        self.model = model.to(self.device)
        self.model_name = cfg.model.type 
        self.data_name = cfg.data.name
        self.FILE_PATH = FILE_PATH 
        self.epochs = cfg.train.epochs
        self.test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)
        self.optimizer = optimizer
        self.loggers = loggers
        
        model_types = ['VGAE', 'GAE', 'GAT', 'GraphSage', 'GNNStack']
        self.train_func = {model_type: self._train_gae if model_type in ['GAE', 'GAT', 'GraphSage', 'GNNStack'] else self._train_vgae for model_type in model_types}
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate for model_type in model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        
        self.run = run
        self.repeat = repeat
        self.results_rank = {}



    def _train_gae(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encoder(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def _train_vgae(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encoder(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    @torch.no_grad()
    def _test(self, data: Data):
        """test"""
        self.model.eval()

        pos_edge_index = self.test_data.pos_edge_label_index
        neg_edge_index = self.test_data.neg_edge_label_index

        z = self.model.encoder(self.test_data.x, self.test_data.edge_index)
        pos_y = z.new_ones(pos_edge_index.size(1)) 
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
        return roc_auc_score(y, pred), average_precision_score(y, pred), auc(fpr, tpr)

    @torch.no_grad()
    def _evaluate(self, data: Data):
       
        self.model.eval()
        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        z = self.model.encoder(data.x, data.edge_index)
        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        
        hard_thres = (y_pred.max() + y_pred.min())/2

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1)) 
        y = torch.cat([pos_y, neg_y], dim=0)
        
        y_pred[y_pred >= hard_thres] = 1
        y_pred[y_pred < hard_thres] = 0

        acc = torch.sum(y == y_pred)/len(y)
        
        pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'acc': round(acc.tolist(), 5)})
    
        return result_mrr
    
    
    def merge_result_rank(self):
        result_test = self.evaluate_func[self.model_name](self.test_data)
        result_valid = self.evaluate_func[self.model_name](self.valid_data)
        result_train = self.evaluate_func[self.model_name](self.train_data)

        return {
            key: (result_train[key], result_valid[key], result_test[key])
            for key in result_test.keys()
        }
    
    def train(self):
        best_auc, best_hits, best_hit100 = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            loss = self.train_func[self.model_name]()
            
            if epoch % 100 == 0:
                results_rank = self.merge_result_rank()
                print(results_rank)
                
                for key, result in results_rank.items():
                    print(key, result)
                    self.loggers[key].add_result(self.run, result)
                    print(self.run)
                    print(result)
                    # for key, result in results_rank.items():
                    #     print(key)
                    #     train_hits, valid_hits, test_hits = result

                    #     print(
                    #         f'Run: {self.run + 1:02d}, '
                    #           f'Epoch: {epoch:02d}, '
                    #           f'Loss: {loss:.4f}, '
                    #           f'Train: {100 * train_hits:.2f}%, '
                    #           f'Valid: {100 * valid_hits:.2f}%, '
                    #           f'Test: {100 * test_hits:.2f}%')
                    # print('---')
        
        # for key in self.loggers:
        #     print(key)
        #     self.loggers[key].print_statistics(self.run)
            
        return best_auc, best_hits



    def result_statistic(self):
        result_all_run = {}
        for key in self.loggers:
            print(key)
            best_metric,  best_valid_mean, mean_list, var_list = self.loggers[key].calc_all_stats()
            
            if key == 'AUC':
                best_auc_valid_str = best_metric
                best_auc_metric = best_valid_mean

            elif key == 'Hits@100':
                best_metric_valid_str = best_metric
                best_valid_mean_metric = best_valid_mean

            result_all_run[key] = [mean_list, var_list]

        print(f'{best_metric_valid_str} {best_auc_valid_str}')
        print(best_metric_valid_str)
        best_auc_metric = best_valid_mean_metric
        return best_valid_mean_metric, best_auc_metric, result_all_run
    

    def save_result(self, results_dict: Dict[str, float]):  # sourcery skip: avoid-builtin-shadow
        
        root = os.path.join(self.FILE_PATH, cfg.out_dir)
        acc_file = os.path.join(root, f'{self.data_name}_wb_acc_mrr.csv')
        print(results_dict)
        print(acc_file)
        os.makedirs(root, exist_ok=True)
        id = wandb.util.generate_id()
        param_tune_acc_mrr(id, results_dict, acc_file, self.data_name, self.model_name)


class Trainer_Saint(Trainer):
    def __init__(self, 
                 FILE_PATH,
                 cfg,
                 model, 
                 optimizer,
                 splits,
                 run, 
                 repeat, 
                 loggers,
                 gsaint=None,
                 batch_size=None, 
                 walk_length=None, 
                 num_steps=None, 
                 sample_coverage=None):
        super().__init__(FILE_PATH, cfg, model, optimizer, splits, run, repeat, loggers)

        
        self.device = config_device(cfg)
            
        self.model = model.to(self.device)
        
        self.model_name = cfg.model.type 
        self.data_name = cfg.data.name

        self.FILE_PATH = FILE_PATH 
        self.epochs = cfg.train.epochs
        
        # Added GSAINT normalization
        if gsaint is not None:
            if gsaint.__name__ == 'get_loader_RW':
                self.test_data = gsaint(splits['test'],   batch_size, walk_length, num_steps, sample_coverage)
                self.train_data = gsaint(splits['train'], batch_size, walk_length, num_steps, sample_coverage)
                self.valid_data = gsaint(splits['valid'], batch_size, walk_length, num_steps, sample_coverage)
            else:
                self.test_data = gsaint(splits['test'],   batch_size, num_steps, sample_coverage)
                self.train_data = gsaint(splits['train'], batch_size, num_steps, sample_coverage)
                self.valid_data = gsaint(splits['valid'], batch_size, num_steps, sample_coverage)
        else:
            self.test_data = splits['test']
            self.train_data = splits['train']
            self.valid_data = splits['valid']
        
        self.test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)
        self.optimizer = optimizer
        
