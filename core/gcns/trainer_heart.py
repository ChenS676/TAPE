import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
# from logger import Logger

import torch
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import *
from torch_sparse import SparseTensor
from heuristic.eval import get_metric_score
from torch.utils.data import DataLoader


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
        
        # visualize
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = perm.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def train_opt(model, score_func, train_pos, x, optimizer, batch_size):
    model.train()
    score_func.train()

    total_loss = total_examples = 0

    for perm in DataLoader(train_pos, batch_size, shuffle=True):
        optimizer.zero_grad()

        num_nodes = x.size(0)

        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0

        train_edge_mask = train_pos[mask].transpose(1, 0)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1, 0]]), dim=1)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)

        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)

        h = model(x, adj)

        edge_index = train_pos[perm].t()

        pos_out = score_func(h[edge_index[0]], h[edge_index[1]])
        pos_loss = torch.nn.BCEWithLogitsLoss()(pos_out, torch.ones_like(pos_out))

        neg_edge_index = torch_geometric.nn.negative_sampling(edge_index, num_nodes=num_nodes)
        neg_out = score_func(h[neg_edge_index[0]], h[neg_edge_index[1]])
        neg_loss = torch.nn.BCEWithLogitsLoss()(neg_out, torch.zeros_like(neg_out))

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
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

    if emb == None: x = data.x
    else: x = emb.weight
    
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
    
    result= {}
    for k, val in result_train.items():
        result.update({k: (result_train[k], result_valid[k], result_test[k])})
    return result, score_emb




@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size):
    input_data  = input_data.transpose(1, 0)
    # print(input_data.size())
    with torch.no_grad():
        preds = []
        for perm  in DataLoader(range(input_data.size(0)), batch_size):

            edge = input_data[perm]
            preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
            
        pred_all = torch.cat(preds, dim=0)

    return pred_all

