import os
import sys
import os
import sys
import time
from os.path import abspath, dirname, join

from matplotlib import pyplot as plt
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


class Trainer_Subgraph_Sketching(Trainer):
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
        self.data = data.to(self.device)

        self.test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)
        self.optimizer = optimizer
        model_types = ['ELPH', 'BUDDY']
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate for model_type in model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

        self.run = run
        self.repeat = repeat
        self.results_rank = {}

        self.name_tag = cfg.wandb.name_tag
        self.run_result = {}

    def _train_elph(self):
        self.model.train()
        total_loss = 0
        pos_train_edge = self.train_data['pos_edge_label_index'].T.to(self.device)
        neg_train_edge = self.train_data['neg_edge_label_index'].T.to(self.device)
        links = torch.cat([pos_train_edge, neg_train_edge], dim=0)
        labels = torch.cat([torch.ones(pos_train_edge.size(0)), torch.zeros(neg_train_edge.size(0))]).to(self.device)

        for perm in PermIterator(self.device, links.shape[0], self.batch_size):
            self.optimizer.zero_grad()
            node_features, hashes, cards = self.model(self.data.x, self.data.edge_index)
            curr_links = links[perm]
            batch_node_features = node_features[curr_links]
            batch_emb = None
            subgraph_features = self.model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(self.device)
            logits = self.model.predictor(subgraph_features, batch_node_features, batch_emb)
            loss = BCEWithLogitsLoss()(logits.view(-1), labels[perm].to(self.device))

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * self.batch_size

        return total_loss / len(self.train_data)
    def _train_buddy(self):
        self.model.train()
        total_loss = 0
        pos_train_edge = self.train_data['pos_edge_label_index'].T.to(self.device)
        neg_train_edge = self.train_data['neg_edge_label_index'].T.to(self.device)
        links = torch.cat([pos_train_edge, neg_train_edge], dim=0)
        labels = torch.cat([torch.ones(pos_train_edge.size(0)), torch.zeros(neg_train_edge.size(0))]).to(self.device)
        sample_indices = torch.randperm(len(labels))[:len(labels)].to(self.device)


        for perm in PermIterator(self.device, links.shape[0], self.batch_size):
            self.optimizer.zero_grad()
            curr_links = links[perm]
            batch_node_features = self.train_data.x[curr_links].to(self.device)
            batch_emb = None
            RA = None
            degrees = self.train_data.degrees[curr_links].to(self.device)
            subgraph_features = self.train_data.subgraph_features[sample_indices[perm]].to(self.device)
            logits = self.model(subgraph_features, batch_node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)

            loss = BCEWithLogitsLoss()(logits.view(-1), labels[perm].to(self.device))

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * self.batch_size

        return total_loss / len(self.train_data)

    def train(self):
        best_auc, best_hits, best_hit100 = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            if self.model_name == 'ELPH':
                loss = self._train_elph()
            elif self.model_name == 'BUDDY':
                loss = self._train_buddy()
            if torch.isnan(torch.tensor(loss)):
                print('Loss is nan')
                break
            if epoch % 10 == 0:
                results_rank = self.merge_result_rank()
                print(results_rank)

                for key, result in results_rank.items():
                    print(key, result)
                    self.loggers[key].add_result(self.run, result)
                    print(self.run)
                    print(result)

        return best_auc, best_hits

    @torch.no_grad()
    def _evaluate(self, eval_data: Data):

        self.model.eval()
        pos_train_edge = eval_data['pos_edge_label_index'].T.to(self.device)
        neg_train_edge = eval_data['neg_edge_label_index'].T.to(self.device)
        links = torch.cat([pos_train_edge, neg_train_edge], dim=0)


        if self.model_name == 'ELPH':
            node_features, hashes, cards = self.model(self.data.x, self.data.edge_index)
            subgraph_features = self.model.elph_hashes.get_subgraph_features(links, hashes, cards).to(self.device)
            y_pred = self.model.predictor(subgraph_features, node_features[links], None)
        elif self.model_name == 'BUDDY':
            RA = None
            degrees = eval_data.degrees[links].to(self.device)
            subgraph_features = eval_data.subgraph_features.to(self.device)
            y_pred = self.model(subgraph_features, eval_data.x[links], degrees[:, 0], degrees[:, 1], RA, None)


        y_pred = y_pred.view(-1).cpu()
        hard_thres = (y_pred.max() + y_pred.min()) / 2
        pos_y = torch.ones(pos_train_edge.size(0))
        neg_y = torch.zeros(neg_train_edge.size(0))
        y_true = torch.cat([pos_y, neg_y], dim=0)

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

'''    def plotacc(self, y_true, y_pred, hard_thres):
        global counter
        counter += 1
        file_name = f'plot{counter}acc{torch.sum(y_true == torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))) / len(y_true)}.png'
        save_dir = './plots'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, color='blue', s=1, label='Predictions vs True Values')
        plt.axhline(y=hard_thres, color='red', linestyle='--', label=f'Hard Threshold: {hard_thres.item()}')
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.title('Scatter Plot of Predictions vs True Values with Hard Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
counter = 0'''