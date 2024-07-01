import os
import sys
import os
import sys
import time
from os.path import abspath, dirname, join

from torch.nn import BCEWithLogitsLoss
from torch_sparse import SparseTensor

sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))
from torch_geometric.utils import negative_sampling

import torch
import wandb
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from yacs.config import CfgNode as CN
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from embedding.tune_utils import param_tune_acc_mrr
from heuristic.eval import get_metric_score
from graphgps.utility.utils import config_device, Logger
from typing import Dict, Tuple
from graphgps.train.opt_train import (Trainer)
from graphgps.utility.ncn import PermIterator


class Trainer_NeoGNN(Trainer):
    def __init__(self,
                 FILE_PATH: str,
                 cfg: CN,
                 model: torch.nn.Module,
                 predictor: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data: Data,
                 splits: Dict[str, Data],
                 run: int,
                 repeat: int,
                 loggers: Dict[str, Logger],
                 print_logger: None,
                 batch_size=None,):
        self.device = config_device(cfg).device
        self.model = model.to(self.device)
        self.predictor = predictor.to(self.device)

        self.model_name = cfg.model.type
        self.data_name = cfg.data.name

        self.FILE_PATH = FILE_PATH
        self.epochs = cfg.train.epochs
        self.run = run
        self.repeat = repeat
        self.loggers = loggers
        self.print_logger = print_logger
        self.batch_size = batch_size
        self.gnn_batch_size = cfg.train.gnn_batch_size
        self.data = data

        self.test_data = splits['test']
        self.train_data = splits['train']
        self.valid_data = splits['valid']
        self.optimizer = optimizer
        self.train_func = self._train_neognn
        model_types = ['NeoGNN']
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate for model_type in model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

        self.run = run
        self.repeat = repeat
        self.results_rank = {}

        self.name_tag = cfg.wandb.name_tag
        self.run_result = {}

    def _train_neognn(self):
        self.model.train()
        total_loss = 0
        self.predictor.train()
        pos_train_edge = self.train_data['pos_edge_label_index'].to(self.device).T
        neg_train_edge = self.train_data['neg_edge_label_index'].to(self.device).T
        # permute the edges
        total_loss = total_examples = 0
        count = 0
        for perm, perm_large in zip(DataLoader(range(pos_train_edge.size(0)), self.batch_size,
                                               shuffle=True),
                                    DataLoader(range(pos_train_edge.size(0)), self.gnn_batch_size,
                                               shuffle=True)):
            self.optimizer.zero_grad()
            # compute scores of positive edges
            edge = pos_train_edge[perm].t()
            edge_large = pos_train_edge[perm_large].t()
            pos_out, pos_out_struct, _, _ = self.model(edge, self.data, self.data.A, self.predictor, emb=self.data.emb.weight)
            _, _, pos_out_feat_large = self.model(edge_large, self.data, self.data.A, self.predictor, emb=self.data.emb.weight, only_feature=True)

            # compute scores of negative edges
            # Just do some trivial random sampling.
            edge = neg_train_edge[perm].t()

            edge_large = neg_train_edge[perm_large].t()
            neg_out, neg_out_struct, _, _ = self.model(edge, self.data, self.data.A, self.predictor, emb=self.data.emb.weight)
            _, _, neg_out_feat_large = self.model(edge_large, self.data, self.data.A, self.predictor, emb=self.data.emb.weight, only_feature=True)

            pos_loss = -torch.log(pos_out_struct + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out_struct + 1e-15).mean()
            loss1 = pos_loss + neg_loss
            pos_loss = -torch.log(pos_out_feat_large + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out_feat_large + 1e-15).mean()
            loss2 = pos_loss + neg_loss
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            loss3 = pos_loss + neg_loss
            loss = loss1 + loss2 + loss3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.data.emb.weight, 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            count += 1

        return total_loss / total_examples

    def train(self):
        best_auc, best_hits, best_hit100 = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_neognn()
            if torch.isnan(torch.tensor(loss)):
                print('Loss is nan')
                break
            if epoch % 100 == 0:
                results_rank = self.merge_result_rank()
                print(results_rank)

                for key, result in results_rank.items():
                    print(key, result)
                    self.loggers[key].add_result(self.run, result)
                    print(self.run)
                    print(result)

        return best_auc, best_hits

    @torch.no_grad()
    def _test(self, data: Data):
        self.model.eval()
        self.predictor.eval()
        pos_edge = data['pos_edge_label_index'].to(self.device)
        neg_edge = data['neg_edge_label_index'].to(self.device)
        pos_pred,_,_,_ = self.model(pos_edge, self.data, self.data.A, self.predictor, emb=self.data.emb.weight)
        pos_pred = pos_pred.squeeze()
        neg_pred,_,_,_ = self.model(neg_edge, self.data, self.data.A, self.predictor, emb=self.data.emb.weight)
        neg_pred = neg_pred.squeeze()

        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        hard_thres = (y_pred.max() + y_pred.min()) / 2
        pos_y = torch.ones(pos_edge.size(1))
        neg_y = torch.zeros(neg_edge.size(1))
        y_true = torch.cat([pos_y, neg_y], dim=0)
        '''self.save_pred(y_pred, y_true, data)'''

        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach()
        y_pred = y_pred.clone().detach()

        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred), auc(fpr, tpr)

    @torch.no_grad()
    def _evaluate(self, eval_data: Data):

        self.model.eval()
        self.predictor.eval()
        pos_edge = eval_data['pos_edge_label_index'].to(self.device)
        neg_edge = eval_data['neg_edge_label_index'].to(self.device)
        pos_pred, _, _, _ = self.model(pos_edge, self.data, self.data.A, self.predictor, emb=self.data.emb.weight)
        pos_pred = pos_pred.squeeze()
        neg_pred, _, _, _ = self.model(neg_edge, self.data, self.data.A, self.predictor, emb=self.data.emb.weight)
        neg_pred = neg_pred.squeeze()
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        hard_thres = (y_pred.max() + y_pred.min()) / 2
        pos_y = torch.ones(pos_edge.size(1))
        neg_y = torch.zeros(neg_edge.size(1))
        y_true = torch.cat([pos_y, neg_y], dim=0)
        '''self.save_pred(y_pred, y_true, eval_data)'''

        pos_pred, neg_pred = y_pred[y_true == 1].cpu(), y_pred[y_true == 0].cpu()
        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach().cpu()
        y_pred = y_pred.clone().detach().cpu()



        acc = torch.sum(y_true == y_pred) / len(y_true)

        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'ACC': round(acc.tolist(), 5)})

        return result_mrr

    def save_pred(self, pred, true, data):
        root = os.path.join(self.FILE_PATH, cfg.out_dir, 'pred_record')
        os.makedirs(root, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        file_path = os.path.join(root, f'{cfg.dataset.name}_{timestamp}.txt')

        with open(file_path, 'w') as f:
            for idx, subgraph in enumerate(data):
                indices = torch.where(subgraph['z'] == 1)[0]
                if len(indices) < 2:
                    continue
                corresponding_node_ids = subgraph['node_id'][indices]
                pred_value = pred[idx]
                true_value = true[idx]
                f.write(f"{corresponding_node_ids[0].item()} {corresponding_node_ids[1].item()} {pred_value} {true_value}\n")


