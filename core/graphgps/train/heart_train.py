
import torch
import torch.nn.functional as F

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from logger import Logger
import torch
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling
from torch_geometric.graphgym.config import cfg
# from sklearn.metrics import *
from torch_sparse import SparseTensor
from heuristic.eval import get_metric_score
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from graphgps.train.opt_train import Trainer

def train(model, 
          score_func,  
          train_pos, 
          data, 
          emb, 
          optimizer, 
          batch_size, 
          pos_train_weight,
          device):
    
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0

    # pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    
    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
        x = emb.weight
        emb_update = 1
    
    train_pos = train_pos.t()
    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)

        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
        train_edge_mask = train_pos[mask].transpose(1,0)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
        
        if pos_train_weight != None:
            pos_train_weight = pos_train_weight.to(mask.device)
            edge_weight_mask = pos_train_weight[mask]
            edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
        else:
            edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
            
        # masked adjacency matrix 
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
          
        ##################
        # print(adj)
        x = x.to(device)
        adj = adj.to(device)
        h = model(x, adj)

        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        
        row, col, _ = adj.coo()
        edge_index = torch.stack([col, row], dim=0)
        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                num_neg_samples=perm.size(0), method='dense')

        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = perm.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, 
         score_func, 
         data, 
         evaluation_edges, 
         emb, 
         evaluator_hit, 
         evaluator_mrr, 
         batch_size,
         data_name, 
         use_valedges_as_input, 
         device):
    
    model.eval()
    score_func.eval()

    # adj_t = adj_t.transpose(1,0)
    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges

    x = data.x if emb is None else emb.weight
    x = x.to(device)
    data = data.to(device)
    h = model(x, data.edge_index.to(x.device))
    # print(h[0][:10])
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device)
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device)
    neg_test_edge = neg_test_edge.to(x.device)

    pos_train_pred = test_edge(score_func, train_val_edge, h, batch_size)

    neg_valid_pred = test_edge(score_func, neg_valid_edge, h, batch_size)

    pos_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size)

    if use_valedges_as_input:
        print('use_val_in_edge')
        h = model(x, data.edge_index.to(x.device))

    pos_test_pred = test_edge(score_func, pos_test_edge, h, batch_size)

    neg_test_pred = test_edge(score_func, neg_test_edge, h, batch_size)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)


    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result_train = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, neg_valid_pred)
    result_valid = get_metric_score(evaluator_hit, evaluator_mrr, pos_valid_pred, pos_valid_pred)
    result_test = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]

    result = {
        k: (result_train[k], result_valid[k], result_test[k])
        for k, val in result_train.items()
    }
    return result, score_emb




@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size):

    # input_data  = input_data.transpose(1, 0)
    # with torch.no_grad():
    preds = []

    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()

        preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]

    return torch.cat(preds, dim=0)

class Trainer_Heart(Trainer):
    def __init__(self, 
                FILE_PATH,
                cfg,
                model, 
                emb,
                data,
                optimizer,
                splits,
                run, 
                repeat, 
                loggers,
                print_logger,
                device, 
                if_wandb):
        super().__init__(FILE_PATH,
                    cfg,
                    model, 
                    emb,
                    data,
                    optimizer,
                    splits,
                    run, 
                    repeat, 
                    loggers,
                    print_logger,
                    device)
        
        self.batch_size = cfg.train.batch_size
        model_types = ['VGAE', 'GAE', 'GAT', 'GraphSage']

        self.train_func = {model_type: self._train_heart for model_type in model_types}
        self.test_func = {model_type: self._eval_heart  for model_type in model_types}
        self.evaluate_func = {model_type: self._eval_heart  for model_type in model_types}

        self.if_wandb = if_wandb
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        
        self.train_loader = DataLoader(range(self.train_data.edge_index.size(1)),
                          batch_size=self.batch_size,
                          shuffle=True,  # Adjust the number of workers based on your system configuration
                          pin_memory=True,  # Enable pinning memory for faster data transfer
                          drop_last=True) 
        if if_wandb:
            iters = len(self.train_loader)
            step = self.epochs * iters
            best_loss = torch.inf
        
    def _train_heart(self):

        edge_index = self.train_data.edge_index
        pos_train_weight = None
        
        if self.emb is None: 
            x = self.train_data.x
            emb_update = 0
        else: 
            x = self.emb.weight
            emb_update = 1
      
        for perm in self.train_loader:  # Drop the last incomplete batch if dataset size is not divisible by batch size
            
            self.optimizer.zero_grad()
            num_nodes = x.size(0)

            ######################### remove loss edges from the aggregation
            mask = torch.ones(edge_index.size(1), dtype=torch.bool).to(edge_index.device)
            mask[perm] = 0
            train_edge_mask = edge_index[:, mask]
            train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)

            # visualize
            if pos_train_weight != None:
                pos_train_weight = pos_train_weight.to(mask.device)
                edge_weight_mask = pos_train_weight[mask]
                edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
            else:
                edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(edge_index.device)

            adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(edge_index.device)

            row, col, _ = adj.coo()
            batch_edge_index = torch.stack([col, row], dim=0)
            
            
            x = x.to(self.device)
            pos_edge =  edge_index[:, perm].to(self.device)
            if self.model_name == 'VGAE':
                h = self.model(x, batch_edge_index)
                loss = self.model.recon_loss(h, pos_edge)
                loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
            elif self.model_name in ['GAE', 'GAT', 'GraphSage']:
                h = self.model.encoder(x, batch_edge_index)
                loss = self.model.recon_loss(h, pos_edge)

            loss.backward()

            if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()           

        return loss.item() 


    @torch.no_grad()
    def test_edge(self, h, edge_index):
        preds = []
        edge_index = edge_index.t()

        for perm  in DataLoader(range(edge_index.size(0)), self.batch_size):
            edge = edge_index[perm].t()

            preds += [self.model.decoder(h, edge).cpu()]

        return torch.cat(preds, dim=0)


    @torch.no_grad()
    def _eval_heart(self, data: Data):
        self.model.eval()
        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        if self.model_name == 'VGAE':
            z = self.model(data.x, data.edge_index)
        elif self.model_name in ['GAE', 'GAT', 'GraphSage']:
            z = self.model.encoder(data.x, data.edge_index)
        
        pos_pred = self.test_edge(z, pos_edge_index)
        neg_pred = self.test_edge(z, neg_edge_index)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        
        hard_thres = (y_pred.max() + y_pred.min())/2

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1)) 
        y = torch.cat([pos_y, neg_y], dim=0)
        
        y_pred[y_pred >= hard_thres] = 1
        y_pred[y_pred < hard_thres] = 0

        y = y.to(self.device)
        y_pred = y_pred.to(self.device)
        acc = torch.sum(y == y_pred)/len(y)
        
        pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'acc': round(acc.tolist(), 5)})
    
        return result_mrr
    
    
