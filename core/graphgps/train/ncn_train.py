import os
import sys
import os
import sys
import time
from os.path import abspath, dirname, join

from torch.nn import BCEWithLogitsLoss
from torch_sparse import SparseTensor

sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

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
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class Trainer_NCN(Trainer):
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
        self.data = data

        self.test_data = splits['test']
        self.train_data = splits['train']
        self.valid_data = splits['valid']
        self.optimizer = optimizer
        self.train_func = self._train_ncn
        model_types = ['NCN', 'NCNC']
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate for model_type in model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

        self.run = run
        self.repeat = repeat
        self.results_rank = {}

        self.name_tag = cfg.wandb.name_tag
        self.run_result = {}

        self.tensorboard_writer = writer
        self.out_dir = cfg.out_dir
        self.run_dir = cfg.run_dir

        report_step = {
            'cora': 1,
            'pubmed': 1,
            'arxiv_2023': 1,
            'ogbn-arxiv': 1,
            'ogbn-products': 1,
        }

        self.report_step = report_step[cfg.data.name]

    def _train_ncn(self):
        self.model.train()
        total_loss = 0
        self.predictor.train()

        pos_train_edge = self.train_data['pos_edge_label_index'].to(self.device)
        adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool) # mask for adj
        negedge = self.train_data['neg_edge_label_index'].to(self.device)
        # permute the edges
        for perm in PermIterator(adjmask.device, adjmask.shape[0], self.batch_size):
            self.optimizer.zero_grad()
            # mask input edges (target link removal)
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask] # get the target edge index
            # get the adj matrix
            adj = SparseTensor.from_edge_index(tei, sparse_sizes=(self.data.num_nodes, self.data.num_nodes)).to_device(
                pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
            h = self.model(self.data.x, adj) # get the node embeddings
            edge = pos_train_edge[:, perm]
            pos_outs = self.predictor.multidomainforward(h, adj, edge) # get the prediction
            pos_losss = -F.logsigmoid(pos_outs).mean()
            edge = negedge[:, perm]
            neg_outs = self.predictor.multidomainforward(h, adj, edge)
            neg_losss = -F.logsigmoid(-neg_outs).mean()
            loss = neg_losss + pos_losss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss

    def train(self):
        best_auc, best_hits, best_hit100 = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_ncn()
            self.tensorboard_writer.add_scalar("Loss/train", loss, epoch)
            if torch.isnan(torch.tensor(loss)):
                print('Loss is nan')
                break
            if epoch % int(self.report_step) == 0:
                self.results_rank = self.merge_result_rank()

                for key, result in self.results_rank.items():
                    self.loggers[key].add_result(self.run, result)
                    self.print_logger.info(
                        f'Epoch: {epoch:03d}, Loss_train: {loss:.4f}, AUC: {self.results_rank["AUC"][0]:.4f}, AP: {self.results_rank["AP"][0]:.4f}, MRR: {self.results_rank["MRR"][0]:.4f}, Hit@10 {self.results_rank["Hits@10"][0]:.4f}')
                    self.print_logger.info(
                        f'Epoch: {epoch:03d}, Loss_valid: {loss:.4f}, AUC: {self.results_rank["AUC"][1]:.4f}, AP: {self.results_rank["AP"][1]:.4f}, MRR: {self.results_rank["MRR"][1]:.4f}, Hit@10 {self.results_rank["Hits@10"][1]:.4f}')
                    self.print_logger.info(
                        f'Epoch: {epoch:03d}, Loss_test: {loss:.4f}, AUC: {self.results_rank["AUC"][2]:.4f}, AP: {self.results_rank["AP"][2]:.4f}, MRR: {self.results_rank["MRR"][2]:.4f}, Hit@10 {self.results_rank["Hits@10"][2]:.4f}')

                    self.tensorboard_writer.add_scalar(f"Metrics/Train/{key}", result[0], epoch)
                    self.tensorboard_writer.add_scalar(f"Metrics/Valid/{key}", result[1], epoch)
                    self.tensorboard_writer.add_scalar(f"Metrics/Test/{key}", result[2], epoch)

                    train_hits, valid_hits, test_hits = result
                    self.print_logger.info(
                        f'Run: {self.run + 1:02d}, Key: {key}, '
                        f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_hits:.2f}, Valid: {100 * valid_hits:.2f}, Test: {100 * test_hits:.2f}%')

                self.print_logger.info('---')


        return best_auc, best_hits

    @torch.no_grad()
    def _test(self, data: Data):
        self.model.eval()
        self.predictor.eval()
        pos_edge = data['pos_edge_label_index'].to(self.device)
        neg_edge = data['neg_edge_label_index'].to(self.device)
        if data == self.test_data:
            adj = self.data.full_adj_t
            h = self.model(self.data.x, adj)
        else:
            adj = self.data.adj_t
            h = self.model(self.data.x, adj)
        pos_pred = torch.cat([self.predictor(h, adj, pos_edge[perm]).squeeze().cpu()
                              for perm in PermIterator(pos_edge.device, pos_edge.shape[0], self.batch_size, False)],
                             dim=0)

        neg_pred = torch.cat([self.predictor(h, adj, neg_edge[perm]).squeeze().cpu()
                              for perm in PermIterator(neg_edge.device, neg_edge.shape[0], self.batch_size, False)],
                             dim=0)

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
        if eval_data == self.test_data:
            adj = self.data.full_adj_t
            h = self.model(self.data.x, adj)
        else:
            adj = self.data.adj_t
            h = self.model(self.data.x, adj)
        pos_pred = torch.cat([self.predictor(h, adj, pos_edge[perm]).squeeze().cpu()
            for perm in PermIterator(pos_edge.device, pos_edge.shape[0], self.batch_size, False)],dim=0)

        neg_pred = torch.cat([self.predictor(h, adj, neg_edge[perm]).squeeze().cpu()
            for perm in PermIterator(neg_edge.device, neg_edge.shape[0], self.batch_size, False)],dim=0)

        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        hard_thres = (y_pred.max() + y_pred.min()) / 2
        pos_y = torch.ones(pos_edge.size(1))
        neg_y = torch.zeros(neg_edge.size(1))
        y_true = torch.cat([pos_y, neg_y], dim=0)
        '''self.save_pred(y_pred, y_true, eval_data)'''

        pos_pred, neg_pred = y_pred[y_true == 1].cpu(), y_pred[y_true == 0].cpu()
        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach()
        y_pred = y_pred.clone().detach()

        acc = torch.sum(y_true == y_pred) / len(y_true)

        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'ACC': round(acc.tolist(), 5)})

        return result_mrr

    def finalize(self):
        import time
        for _ in range(1):
            start_train = time.time()
            self._evaluate(self.test_data)
            self.run_result['eval_time'] = time.time() - start_train

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


