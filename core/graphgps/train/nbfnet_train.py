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


class Trainer_NBFNet(Trainer):
    def __init__(self,
                 FILE_PATH: str,
                 cfg: CN,
                 model: torch.nn.Module,
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

        self.model_name = cfg.model.type
        self.data_name = cfg.data.name

        self.FILE_PATH = FILE_PATH
        self.epochs = cfg.train.epochs
        self.run = run
        self.repeat = repeat
        self.loggers = loggers
        self.print_logger = print_logger
        self.batch_size = batch_size
        self.data = data

        self.test_data = splits['test']
        self.test_data.edge_type = torch.zeros(self.test_data.edge_index.shape[1], dtype=torch.int).to(self.device)
        self.train_data = splits['train']
        self.train_data.edge_type = torch.zeros(self.train_data.edge_index.shape[1], dtype=torch.int).to(self.device)
        self.valid_data = splits['valid']
        self.valid_data.edge_type = torch.zeros(self.valid_data.edge_index.shape[1], dtype=torch.int).to(self.device)
        self.optimizer = optimizer
        self.train_func = self._train_nbfnet
        model_types = ['NBFNet']
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate for model_type in model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

        self.run = run
        self.repeat = repeat
        self.results_rank = {}

        self.name_tag = cfg.wandb.name_tag
        self.run_result = {}

    def _train_nbfnet(self):
        self.model.train()
        total_loss = 0
        # permute the edges
        total_loss = total_examples = 0
        count = 0
        train_triplets_pos = torch.cat([self.train_data['pos_edge_label_index'], torch.zeros_like(self.train_data['pos_edge_label'].unsqueeze(0), dtype=torch.int)]).t()
        train_triplets_neg = torch.cat([self.train_data['neg_edge_label_index'], torch.zeros_like(self.train_data['neg_edge_label'].unsqueeze(0), dtype=torch.int)]).t()
        for perm in PermIterator(self.device, train_triplets_pos.shape[0], self.batch_size):
            self.optimizer.zero_grad()
            batch_data = torch.stack([train_triplets_pos[perm], train_triplets_neg[perm]], dim=1)
            pred = self.model(self.train_data, batch_data)
            target = torch.zeros_like(pred)
            target[:, 0] = 1
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            neg_weight = torch.ones_like(pred)
            neg_weight[:, 1:] = 1 / cfg.task.num_negative
            loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        return total_loss

    def train(self):
        best_auc, best_hits, best_hit100 = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_nbfnet()
            if torch.isnan(torch.tensor(loss)):
                print('Loss is nan')
                break
            if epoch % 1 == 0:
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

        train_triplets_pos = torch.cat([eval_data['pos_edge_label_index'],
                                        torch.zeros_like(eval_data['pos_edge_label'].unsqueeze(0),
                                                         dtype=torch.int)]).t()
        train_triplets_neg = torch.cat([eval_data['neg_edge_label_index'],
                                        torch.zeros_like(eval_data['neg_edge_label'].unsqueeze(0),
                                                         dtype=torch.int)]).t()
        self.optimizer.zero_grad()
        pos_data = torch.stack([train_triplets_pos], dim=1)
        neg_data = torch.stack([train_triplets_neg], dim=1)
        pos_pred = self.model(self.train_data, pos_data)
        neg_pred = self.model(self.train_data, neg_data)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0)
        hard_thres = (y_pred.max() + y_pred.min()) / 2
        pos_pred, neg_pred = y_pred[y_true == 1].cpu(), y_pred[y_true == 0].cpu()

        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach()
        y_pred = y_pred.clone().detach()

        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred), auc(fpr, tpr)

    @torch.no_grad()
    def _evaluate(self, eval_data: Data):

        self.model.eval()

        train_triplets_pos = torch.cat([eval_data['pos_edge_label_index'],
                                        torch.zeros_like(eval_data['pos_edge_label'].unsqueeze(0),
                                                         dtype=torch.int)]).t()
        train_triplets_neg = torch.cat([eval_data['neg_edge_label_index'],
                                        torch.zeros_like(eval_data['neg_edge_label'].unsqueeze(0),
                                                         dtype=torch.int)]).t()
        self.optimizer.zero_grad()
        pos_data = torch.stack([train_triplets_pos], dim=1)
        neg_data = torch.stack([train_triplets_neg], dim=1)
        pos_pred = self.model(self.train_data, pos_data)
        neg_pred = self.model(self.train_data, neg_data)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0)
        hard_thres = (y_pred.max() + y_pred.min()) / 2
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


