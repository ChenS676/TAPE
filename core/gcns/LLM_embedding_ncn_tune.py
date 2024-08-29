import copy
import itertools
import os, sys

from torch_sparse import SparseTensor
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import logging
import wandb
import torch
from functools import partial
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
          create_optimizer, config_device,  create_logger, custom_set_out_dir

from graphgps.config import (dump_cfg, dump_run_cfg)

from graphgps.utility.ncn import PermIterator
from graphgps.network.ncn import predictor_dict, GCN
from data_utils.load import load_data_lp
from graphgps.train.ncn_train import Trainer_NCN


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/seal.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        default='cora',
                        help='data name')
    parser.add_argument('--device', dest='device', required=True,
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=False,
                        default=100,
                        help='data name')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    parser.add_argument('--embedder', type=str, required=False, default='llama')

    return parser.parse_args()

def ncn_dataset(data, splits):
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    # Use training + validation edges for inference on test set.
    if cfg.data.use_valedges_as_input:
        val_edge_index = splits['valid']['pos_edge_label_index']
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data



if __name__ == "__main__":
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)

    torch.set_num_threads(cfg.num_threads)
    batch_sizes = [cfg.train.batch_size]

    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)


    for run_id, seed, split_index in zip(
                *run_loop_settings(cfg, args)):
        id = wandb.util.generate_id()
        custom_set_run_dir(cfg, run_id)
        set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)
        splits, text, data = load_data_lp[cfg.data.name](cfg.data)
        data.edge_index = splits['train']['pos_edge_label_index']
        data = ncn_dataset(data, splits).to(cfg.device)
        saved_features_path = './' + args.embedder + cfg.data.name + 'saved_node_features.pt'
        if os.path.exists(saved_features_path):
            node_features = torch.load(saved_features_path, map_location=torch.device('cpu'))
            data.x = node_features
            data.x = torch.tensor(data.x)
            print('Replaced node_features with saved features')
        else:
            print('Please regenerate node features')
        path = f'{os.path.dirname(__file__)}/ncn_{cfg.data.name}'
        print_logger = set_printing(cfg)
        print_logger.info(
            f"The {cfg['data']['name']} graph {splits['train']['x'].shape} is loaded on {splits['train']['x'].device},"
            f"\n Train: {2 * splits['train']['pos_edge_label'].shape[0]} samples,"
            f"\n Valid: {2 * splits['train']['pos_edge_label'].shape[0]} samples,"
            f"\n Test: {2 * splits['test']['pos_edge_label'].shape[0]} samples")
        dump_cfg(cfg)
        if cfg.model.type == 'NCN':
            hyperparameter_search = {'hiddim': [64, 256], "gnndp": [0.0, 0.2, 0.5],
                                 "xdp": [0.0, 0.3, 0.7], "tdp": [0.0, 0.2],
                                 "gnnedp": [0.0], "predp": [0.0, 0.05], "preedp": [0.0, 0.4],
                                 "batch_size": [256, 512, 1024, 2048], "gnnlr": [0.001, 0.0001], "prelr": [0.001, 0.0001]}
            print_logger.info(f"hypersearch space: {hyperparameter_search}")
            for hiddim, gnndp, xdp, tdp, gnnedp, predp, preedp, batch_size, gnnlr, prelr in tqdm(
                    itertools.product(*hyperparameter_search.values())):
                cfg.model.hiddim = hiddim
                cfg.train.batch_size = batch_size
                cfg.optimizer.gnnlr = gnnlr
                cfg.optimizer.prelr = prelr
                cfg.model.gnndp = gnndp
                cfg.model.xdp = xdp
                cfg.model.tdp = tdp
                cfg.model.gnnedp = gnnedp
                cfg.model.predp = predp
                cfg.model.preedp = preedp
                print_logger.info(f"hidden: {hiddim}")
                print_logger.info(f"bs : {cfg.train.batch_size}")
                print_logger.info(
                    f"gnndp: {gnndp}, xdp: {xdp}, tdp: {tdp}, gnnedp: {gnnedp}, predp: {predp}, preedp: {preedp}, gnnlr: {gnnlr}, prelr: {prelr}")
                start_time = time.time()
                model = GCN(data.num_features, cfg.model.hiddim, cfg.model.hiddim, cfg.model.mplayers,
                            cfg.model.gnndp, cfg.model.ln, cfg.model.res, cfg.data.max_x,
                            cfg.model.model, cfg.model.jk, cfg.model.gnnedp, xdropout=cfg.model.xdp,
                            taildropout=cfg.model.tdp,
                            noinputlin=False)
                predfn = predictor_dict[cfg.model.type]
                if cfg.model.type == 'NCN':
                    predfn = partial(predfn)
                if cfg.model.type == 'NCNC':
                    predfn = partial(predfn, scale=cfg.model.probscale, offset=cfg.model.proboffset, pt=cfg.model.pt)
                predictor = predfn(cfg.model.hiddim, cfg.model.hiddim, 1, cfg.model.nnlayers,
                                   cfg.model.predp, cfg.model.preedp, cfg.model.lnnn)

                optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": cfg.optimizer.gnnlr},
                                              {'params': predictor.parameters(), 'lr': cfg.optimizer.prelr}])
                logging.info(f"{model} on {next(model.parameters()).device}")
                logging.info(cfg)
                cfg.params = params_count(model)
                logging.info(f'Num parameters: {cfg.params}')

                hyper_id = wandb.util.generate_id()
                cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}'
                custom_set_run_dir(cfg, cfg.wandb.name_tag)

                dump_run_cfg(cfg)
                print_logger.info(f"config saved into {cfg.run_dir}")
                print_logger.info(f'Run {run_id} with seed {seed} and split {split_index} on device {cfg.device}')

                # Execute experiment
                trainer = Trainer_NCN(FILE_PATH,
                                      cfg,
                                      model,
                                      predictor,
                                      optimizer,
                                      data,
                                      splits,
                                      run_id,
                                      args.repeat,
                                      loggers,
                                      print_logger=print_logger,
                                      batch_size=batch_size)

                trainer.train()

                run_result = {}
                for key in trainer.loggers.keys():
                    # refer to calc_run_stats in Logger class
                    _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
                    run_result[key] = test_bvalid

                run_result.update(
                    {"gnndp": gnndp, "xdp": xdp, "tdp": tdp, "gnnedp": gnnedp, "predp": predp, "preedp": preedp,
                     "gnnlr": gnnlr, "prelr": prelr})
                run_result.update({"hiddim": hiddim, "batch_size": batch_size})
                print_logger.info(run_result)

                to_file = f'{cfg.data.name}_{cfg.model.type}_tune_result.csv'
                trainer.save_tune(run_result, to_file)

                print_logger.info(f"runing time {time.time() - start_time}")
        elif cfg.model.type == 'NCNC':
            hyperparameter_search = {'probscale': [1.0, 3.0, 5,0], 'proboffset': [0.0, 1.0, 3.0, 5,0], 'pt':[0.05, 0.1]}
            print_logger.info(f"hypersearch space: {hyperparameter_search}")
            for probscale, proboffset, pt in tqdm(itertools.product(*hyperparameter_search.values())):
                cfg.model.probscale = probscale
                cfg.model.proboffset = proboffset
                cfg.model.pt = pt
                print_logger.info(f"bs : {cfg.train.batch_size}")
                print_logger.info(f"probscale: {probscale}, pt: {pt}, proboffset: {proboffset}")
                start_time = time.time()
                model = GCN(data.num_features, cfg.model.hiddim, cfg.model.hiddim, cfg.model.mplayers,
                            cfg.model.gnndp, cfg.model.ln, cfg.model.res, cfg.data.max_x,
                            cfg.model.model, cfg.model.jk, cfg.model.gnnedp, xdropout=cfg.model.xdp, taildropout=cfg.model.tdp,
                            noinputlin=False)
                predfn = predictor_dict[cfg.model.type]
                if cfg.model.type == 'NCN':
                    predfn = partial(predfn)
                if cfg.model.type == 'NCNC':
                    predfn = partial(predfn, scale=cfg.model.probscale, offset=cfg.model.proboffset, pt=cfg.model.pt)
                predictor = predfn(cfg.model.hiddim, cfg.model.hiddim, 1, cfg.model.nnlayers,
                                   cfg.model.predp, cfg.model.preedp, cfg.model.lnnn)

                optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": cfg.optimizer.gnnlr},
                                              {'params': predictor.parameters(), 'lr': cfg.optimizer.prelr}])
                logging.info(f"{model} on {next(model.parameters()).device}")
                logging.info(cfg)
                cfg.params = params_count(model)
                logging.info(f'Num parameters: {cfg.params}')

                hyper_id = wandb.util.generate_id()
                cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{cfg.model.type}_hyper{hyper_id}'
                custom_set_run_dir(cfg, cfg.wandb.name_tag)

                dump_run_cfg(cfg)
                print_logger.info(f"config saved into {cfg.run_dir}")
                print_logger.info(f'Run {run_id} with seed {seed} and split {split_index} on device {cfg.device}')

                # Execute experiment
                trainer = Trainer_NCN(FILE_PATH,
                                       cfg,
                                       model,
                                       predictor,
                                       optimizer,
                                       data,
                                       splits,
                                       run_id,
                                       args.repeat,
                                       loggers,
                                       print_logger=print_logger,
                                       batch_size=cfg.train.batch_size)

                trainer.train()

                run_result = {}
                for key in trainer.loggers.keys():
                    if trainer.loggers[key].results == [[], []]:
                        run_result[key] = None
                    else:
                        # refer to calc_run_stats in Logger class
                        _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id)
                        run_result[key] = test_bvalid

                run_result.update({"probscale": probscale, "pt": pt, "proboffset": proboffset})
                run_result.update({"batch_size": cfg.train.batch_size})
                print_logger.info(run_result)

                to_file = f'{cfg.data.name}_{cfg.model.type}_tune_result.csv'
                trainer.save_tune(run_result, to_file)

                print_logger.info(f"runing time {time.time() - start_time}")
