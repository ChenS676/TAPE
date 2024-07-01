import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from os.path import abspath, dirname, join
from pprint import pprint
import logging
import torch
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.graphgym.config import cfg
from torch_geometric import seed_everything
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from ogb.linkproppred import Evaluator
from graphgps.network.heart_gnn import (GCN_Variant, GAT_Variant, SAGE_Variant, mlp_model, GIN_Variant, DGCNN, GAE_forall)
from data_utils.load import load_data_lp
from graphgps.utility.utils import Logger, save_emb, get_root_dir, get_logger, config_device, set_cfg, get_git_repo_root_path
from graphgps.train.heart_train import train, test, test_edge


def get_config_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "config")

# dir_path    = get_root_dir()
# log_prin    = get_logger('testrun', 'log', get_config_dir())

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/heart_gnn_models.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gat_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        default='cora',
                        help='data name')
    parser.add_argument('--batch_size', dest='bs', type=int, required=False,
                        default=2**15,
                        help='data name')
    parser.add_argument('--device', dest='device', required=False, default='cpu', 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=False,
                        default=300,
                        help='data name')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()



def data_preprocess(cfg):

    splits, text, data = load_data_lp[cfg.data.name](cfg.data)

    edge_index = data.edge_index
    emb = None # here is your embedding
    node_num = data.num_nodes

    if hasattr(data, 'x') and data.x != None:
        x = data.x
        cfg.model.input_channels = x.size(1)
    else:
        emb = torch.nn.Embedding(node_num, args.hidden_channels)
        cfg.model.input_channels = args.hidden_channels

    if not hasattr(data, 'edge_weight'): 
        train_edge_weight = torch.ones(splits['train'].edge_index.shape[1])
        train_edge_weight = train_edge_weight.to(torch.float)

    data = T.ToSparseTensor()(data)

    if cfg.train.use_valedges_as_input:
        # in the previous setting we share the same train and valid 
        val_edge_index = splits['valid'].edge_index.t()
        val_edge_index = to_undirected(val_edge_index)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)

        edge_weight = torch.ones(full_edge_index.shape[1])
        train_edge_weight = torch.ones(splits['train'].edge_index.shape[1])
        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])

        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        print(data.full_adj_t)
        print(data.adj_t)
    else:
        data.full_adj_t = data.adj_t

    if emb != None:
        torch.nn.init.xavier_uniform_(emb.weight)
    return data, splits, emb, cfg, train_edge_weight


if __name__ == "__main__":

    FILE_PATH = get_git_repo_root_path() + '/'

    args = parse_args()
    # Load args file

    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    cfg = config_device(cfg)

    print(f"device {cfg.device}")

    data, splits, emb, cfg, train_edge_weight = data_preprocess(cfg)

    pprint(cfg)
    cfg_model = eval(f'cfg.model.{cfg.model.type}')
    cfg_model.input_channels = data.x.size(1)
    cfg_score = eval(f'cfg.score.{cfg.score.type}')
    model = eval(cfg.model.type)(cfg_model.input_channels, cfg_model.hidden_channels,
                                 cfg_model.hidden_channels, cfg_model.num_layers, 
                                 cfg_model.dropout).to(cfg.device)

    score_func = eval(cfg.score.type)(cfg_score.hidden_channels, 
                                            cfg_score.hidden_channels,
                                            1, 
                                            cfg_score.num_layers_predictor, 
                                            cfg_score.dropout,
                                            'inner' ).to(cfg.device)

    # train_pos = data['train_pos'].to(x.device)

    # eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    # config reset parameters 
    model.reset_parameters()
    score_func.reset_parameters()

    if cfg_model.emb is True:
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(score_func.parameters()) + list(emb.parameters() ),lr=args.lr, weight_decay=args.l2)
    else:
        optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()),lr=cfg.train.lr, weight_decay=cfg.train.l2)

    if cfg.data.name =='ogbl-collab':
        eval_metric = 'Hits@50'
    elif cfg.data.name =='ogbl-ddi':
        eval_metric = 'Hits@20'

    elif cfg.data.name =='ogbl-ppa':
        eval_metric = 'Hits@100'

    elif cfg.data.name =='ogbl-citation2':
        eval_metric = 'MRR'

    elif cfg.data.name in ['cora', 'pubmed', 'arxiv_2023']:
        eval_metric = 'Hits@100'

    if cfg.data.name != 'ogbl-citation2':
        pos_train_edge = splits['train'].edge_index

        pos_valid_edge = splits['valid'].pos_edge_label_index
        neg_valid_edge = splits['valid'].neg_edge_label_index
        pos_test_edge = splits['test'].pos_edge_label_index
        neg_test_edge = splits['test'].neg_edge_label_index

    else:
        source_edge, target_edge = splits['train']['source_node'], splits['train']['target_node']
        pos_train_edge = torch.cat([source_edge.unsqueeze(0), target_edge.unsqueeze(0)], dim=0)

        # idx = torch.randperm(split_edge['train']['source_node'].numel())[:split_edge['valid']['source_node'].size(0)]
        # source, target = split_edge['train']['source_node'][idx], split_edge['train']['target_node'][idx]
        # train_val_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)

        source, target = splits['valid']['source_node'],  splits['valid']['target_node']
        pos_valid_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        neg_valid_edge = splits['valid']['target_node_neg'] 

        source, target = splits['test']['source_node'],  splits['test']['target_node']
        pos_test_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        neg_test_edge = splits['test']['target_node_neg']

    loggers = {
        'Hits@20': Logger(cfg.train.runs),
        'Hits@50': Logger(cfg.train.runs),
        'Hits@100': Logger(cfg.train.runs),
        'MRR': Logger(cfg.train.runs),
        'AUC':Logger(cfg.train.runs),
        'AP':Logger(cfg.train.runs),
        'mrr_hit20':  Logger(cfg.train.runs),
        'mrr_hit50':  Logger(cfg.train.runs),
        'mrr_hit100':  Logger(cfg.train.runs),
    }

    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]

    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]

    for run in range(cfg.train.runs):

        print('#################################          ', run, '          #################################')
        seed = args.seed if cfg.train.runs == 1 else run
        print('seed: ', seed)

        seed_everything(seed)

        save_path = cfg.save.output_dir+'/lr'+str(cfg.train.lr) \
            + '_drop' + str(cfg_model.dropout) + '_l2'+ \
                str(cfg.train.l2) + '_numlayer' + str(cfg_model.num_layers)+ \
                    '_numPredlay' + str(cfg_score.num_layers_predictor) +\
                        '_numGinMlplayer' + str(cfg_score.gin_mlp_layer)+ \
                            '_dim'+str(cfg_model.hidden_channels) + '_'+ 'best_run_'+str(seed)

        if emb != None:
            torch.nn.init.xavier_uniform_(emb.weight)

        model.reset_parameters()
        score_func.reset_parameters()

        if emb != None:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()) + list(emb.parameters() ),lr=cfg.train.lr, weight_decay=cfg.train.l2)
        else:
            optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(score_func.parameters()),lr=cfg.train.lr, weight_decay=cfg.train.l2)
        best_valid = 0
        kill_cnt = 0

        for epoch in range(1, 1 + cfg.train.epochs):
            loss = train(model, 
                         score_func, 
                         pos_train_edge, 
                         data, 
                         emb, 
                         optimizer, 
                         cfg.train.batch_size, 
                         train_edge_weight, 
                         cfg.device)

            # for attention score   
            # print(model.convs[0].att_src[0][0][:10])

            if epoch % 100 == 0:
                results_rank, score_emb = test(model, 
                                               score_func,
                                               splits['test'],
                                               evaluation_edges, 
                                               emb, 
                                               evaluator_hit, 
                                               evaluator_mrr, 
                                               cfg.train.batch_size, 
                                               cfg.data.name, 
                                               cfg.train.use_valedges_as_input, 
                                               cfg.device)


                for key, _ in loggers.items():
                    loggers[key].add_result(run, results_rank[key])

                for key, result in results_rank.items():
                    train_hits, valid_hits, test_hits = result

                logging.info(
                    f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_hits:.2f}%, '
                        f'Valid: {100 * valid_hits:.2f}%, '
                        f'Test: {100 * test_hits:.2f}%')

                r = torch.tensor(loggers[eval_metric].results[run])
                best_valid_current = round(r[:, 1].max().item(),4)
                best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                print(eval_metric)

                logging.info(f'best valid: {100*best_valid_current:.2f}%, '
                                f'best test: {100*best_test:.2f}%')

                if len(loggers['AUC'].results[run]) > 0:
                    r = torch.tensor(loggers['AUC'].results[run])
                    best_valid_auc = round(r[:, 1].max().item(), 4)
                    best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)

                    print('AUC')
                    logging.info(f'best valid: {100*best_valid_auc:.2f}%, '
                                f'best test: {100*best_test_auc:.2f}%')

                print('---')

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if cfg.save: save_emb(score_emb, save_path)
                else:
                    kill_cnt += 1

                    if kill_cnt > cfg.train.kill_cnt: 
                        print("Early Stopping!!")
                        break

        for key in loggers:
            if len(loggers[key].results[0]) > 0:
                print(key)
                loggers[key].print_statistics( run)
                print('\n')



    result_all_run = {}
    for key in loggers.keys():
        if len(loggers[key].results[0]) > 0:
            print(key)
            best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()
            if key == eval_metric:
                best_metric_valid_str = best_metric
                # best_valid_mean_metric = best_valid_mean
            if key == 'AUC':
                best_auc_valid_str = best_metric
                # best_auc_metric = best_valid_mean
            result_all_run[key] = [mean_list, var_list]



    if cfg.train.runs == 1:
        print(str(best_valid_current) + ' ' + str(best_test) + ' ' + str(best_valid_auc) + ' ' + str(best_test_auc))

    else:
        print(str(best_metric_valid_str) +' ' +str(best_auc_valid_str))
