# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
torch.cuda.empty_cache()
import itertools
from tqdm import tqdm
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.cmd_args import parse_args
import argparse
import wandb

from data_utils.load import load_data_lp
from graphgps.train.heart_train import Trainer_Heart
from graphgps.config import (dump_cfg, dump_run_cfg)
from graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger, LinearDecayLR
from graphgps.score.custom_score import mlp_score, InnerProduct
from graphgps.network.heart_gnn import GAT_Variant, GAE_forall, GCN_Variant, \
                                SAGE_Variant, GIN_Variant, DGCNN
from yacs.config import CfgNode as CN
import os
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def create_GAE_model(cfg_model: CN, 
                       cfg_score: CN,
                       model_name: str):
    if model_name in {'GAT', 'VGAE', 'GAE', 'GraphSage'}:
        raise NotImplementedError('Current model does not exist')
        # model = create_model(cfg_model)

    elif model_name == 'GAT_Variant':
        encoder = GAT_Variant(cfg_model.in_channels, 
                                        cfg_model.hidden_channels, 
                                        cfg_model.out_channels, 
                                        cfg_model.num_layers,
                                        cfg_model.dropout,
                                        cfg_model.heads,
                                        )
    elif model_name == 'GCN_Variant':
        encoder = GCN_Variant(cfg_model.in_channels, 
                                        cfg_model.hidden_channels, 
                                        cfg_model.out_channels, 
                                        cfg_model.num_layers,
                                        cfg_model.dropout,
                                        )
    elif model_name == 'SAGE_Variant':
        encoder = SAGE_Variant(cfg_model.in_channels, 
                                        cfg_model.hidden_channels, 
                                        cfg_model.out_channels, 
                                        cfg_model.num_layers,
                                        cfg_model.dropout,
                                        )
    elif model_name == 'GIN_Variant':
        encoder = GIN_Variant(cfg_model.in_channels, 
                                        cfg_model.hidden_channels, 
                                        cfg_model.out_channels, 
                                        cfg_model.num_layers,
                                        cfg_model.dropout,
                                        cfg_model.mlp_layer
                                        )
                
    if cfg_score.product == 'dot':
        decoder = mlp_score(cfg_model.out_channels,
                            cfg_score.score_hidden_channels, 
                            cfg_score.score_out_channels,
                            cfg_score.score_num_layers,
                            cfg_score.score_dropout,
                            cfg_score.product)
    elif cfg_score.product == 'inner':
        decoder = InnerProduct()

    else:
        # Without this else I got: UnboundLocalError: local variable 'model' referenced before assignment
        raise ValueError('Current model does not exist')

    return GAE_forall(encoder=encoder, decoder=decoder) 

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


FILE_PATH = f'{get_git_repo_root_path()}/'

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/gae_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='pubmed',
                        help='data name')
    parser.add_argument('--batch_size', dest='bs', type=int, required=False,
                        default=2**15,
                        help='data name')
    parser.add_argument('--device', dest='device', required=True, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=True,
                        default=400,
                        help='data name')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        default='GCN_Variant',
                        help='model name')
    parser.add_argument('--score', dest='score', type=str, required=False, default='mlp_score',
                        help='decoder name')
    parser.add_argument('--wandb', dest='wandb', required=False, 
                        help='data name')
    parser.add_argument('--repeat', type=int, default=3,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()

product_space = {
    '0': 'inner', 
    '1': 'dot'    
}

hyperparameter_space = {
    'GAT_Variant': {'out_channels': [2**5, 2**6], 'hidden_channels':  [2**5, 2*4],
                                'heads': [2**2, 2**3], 'negative_slope': [0.1], 'dropout': [0, 0.1], 
                                'num_layers': [1, 2, 3], 
                                'base_lr': [0.015],
                    'score_num_layers_predictor': [3],
                    'score_gin_mlp_layer': [2],
                    'score_hidden_channels': [2**6], 
                    'score_out_channels': [1], 
                    'score_num_layers': [3], 
                    'score_dropout': [0.1], 
                    'product': [0, 1]},
    
    'GCN_Variant': {'out_channels': [2**5, 2**6], 
                    'hidden_channels': [2**5, 2**6], 
                    'batch_size': [2**10],
                    'dropout': [0.1],
                    'num_layers': [1, 2, 3],
                    'negative_slope': [0.1],
                    'base_lr': [0.0089],
                    'score_num_layers_predictor': [3],
                    'score_gin_mlp_layer': [2],
                    'score_hidden_channels': [2**6], 
                    'score_out_channels': [1], 
                    'score_num_layers': [3], 
                    'score_dropout': [0.1], 
                    'product': [0, 1]},
                    

    'SAGE_Variant': {'out_channels': [2**7, 2**8], 
                     'hidden_channels': [2**6, 2**7, 2**8], 
                     'base_lr': [0.0089],
                    'score_num_layers_predictor': [3],
                    'score_gin_mlp_layer': [2],
                    'score_hidden_channels': [2**6], 
                    'score_out_channels': [1], 
                    'score_num_layers': [3], 
                    'score_dropout': [0.1], 
                    'product': [0, 1]},
    
    'GIN_Variant': {'out_channels': [2**6, 2**7, 2**8], 
                    'hidden_channels': [2**6, 2**7, 2**8],
                    'num_layers': [1, 2, 3], 
                    'base_lr': [0.0089],
                    'mlp_layer': [1, 2, 3],
                    'score_num_layers_predictor': [3],
                    'score_gin_mlp_layer': [2],
                    'score_hidden_channels': [2**6], 
                    'score_out_channels': [1], 
                    'score_num_layers': [3], 
                    'score_dropout': [0.1], 
                    'product': [0, 1]},
    
    'VGAE_Variant': {'out_channels': [32], 'hidden_channels': [32], 'batch_size': [2**10]},
}


yaml_file = {   
             'GAT_Variant': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
             'GAE_Variant': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
             'GIN_Variant': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
             'GCN_Variant': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
             'SAGE_Variant': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
             'DGCNN': 'core/yamls/cora/gcns/heart_gnn_models.yaml'
            }


def project_main(): # sourcery skip: avoid-builtin-shadow, low-code-quality
    
    # process params
    args = parse_args()
    args.cfg_file = yaml_file[args.model]
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.name = args.data
    
    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    
    cfg_model = eval(f'cfg.model.{args.model}')
    cfg_score = eval(f'cfg.score.{args.score}')
    cfg.model.type = args.model
    # save params
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)

    torch.set_num_threads(20)
    
    loggers = create_logger(args.repeat)
    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        print(f'run id : {run_id}')
        # Set configurations for each run TODO clean code here 
        # if args.wandb:
        id = wandb.util.generate_id()
        cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{args.model}'

        custom_set_run_dir(cfg, cfg.wandb.name_tag)

        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)

        cfg = config_device(cfg)

        splits, _, data = load_data_lp[cfg.data.name](cfg.data)
        cfg_model.in_channels = splits['train'].x.shape[1]

        print_logger = set_printing(cfg)
        print_logger.info(
            f"The {cfg['data']['name']} graph with shape {splits['train']['x'].shape} is loaded on {splits['train']['x'].device},\n"
            f"Split index: {cfg['data']['split_index']} based on {data.edge_index.size(1)} samples.\n"
            f"Train: {cfg['data']['split_index'][0]}% ({2 * splits['train']['pos_edge_label'].shape[0]} samples),\n"
            f"Valid: {cfg['data']['split_index'][1]}% ({2 * splits['valid']['pos_edge_label'].shape[0]} samples),\n"
            f"Test:  {cfg['data']['split_index'][2]}% ({2 * splits['test']['pos_edge_label'].shape[0]} samples)"
        )

        dump_cfg(cfg)
        hyperparameter_gnn = hyperparameter_space[args.model]
        print_logger.info(f"hypersearch space: {hyperparameter_gnn}")

        keys = hyperparameter_gnn.keys()

        product = itertools.product(*hyperparameter_gnn.values())

        for combination in tqdm(product):
            for key, value in zip(keys, combination):
                if key == 'product':
                    value = product_space[str(value)]
                if hasattr(cfg_model, key):
                    print_logger.info(f"Object cfg_model has attribute '{key}' with value: {getattr(cfg_model, key)}")
                    setattr(cfg_model, key, value)
                    print_logger.info(f"Object cfg_model.{key} updated to {getattr(cfg_model, key)}")
                elif hasattr(cfg_score, key):
                    print_logger.info(f"Object cfg_score has attribute '{key}' with value: {getattr(cfg_score, key)}")
                    setattr(cfg_score, key, value)
                    print_logger.info(f"Object cfg_score.{key} updated to {getattr(cfg_score, key)}")
                    
                elif hasattr(cfg.train, key):
                    print_logger.info(f"Object cfg.train has attribute '{key}' with values {getattr(cfg.train, key)}")
                    setattr(cfg.train, key, value)
                    print_logger.info(f"Object cfg.train.{key} updated to {getattr(cfg.train, key)}")
                elif hasattr(cfg.optimizer, key):    
                    print_logger.info(f"Object cfg.train has attribute '{key}' with values {getattr(cfg.optimizer, key)}")
                    setattr(cfg.train, key, value)
                    print_logger.info(f"Object cfg.train.{key} updated to {getattr(cfg.optimizer, key)}")
                
                
            if cfg_score.product == 'dot':
                cfg_score.in_channels = cfg_model.out_channels
            
            print_logger.info(f"out : {eval(f'cfg.model.{args.model}.out_channels')}, hidden: {eval(f'cfg.model.{args.model}.hidden_channels')}")
            print_logger.info(f"bs : {cfg.train.batch_size}, lr: {cfg.optimizer.base_lr}")
            print_logger.info(f"The model {args.model} is initialized.")


            model = create_GAE_model(cfg_model, cfg_score, args.model)
            model.to(cfg.device)
            print_logger.info(f"{model} on {next(model.parameters()).device}" )

            cfg.model.params = params_count(model)
            print_logger.info(f'Num parameters: {cfg.model.params}')

            optimizer = create_optimizer(model, cfg)
            scheduler = LinearDecayLR(optimizer, start_lr=cfg.optimizer.base_lr, end_lr=cfg.optimizer.base_lr/10, num_epochs=cfg.train.epochs)

            if cfg.train.finetune: 
                model = init_model_from_pretrained(model, cfg.train.finetune,
                                                cfg.train.freeze_pretrained)
                
            if args.wandb:
                hyper_id = wandb.util.generate_id()
                cfg.wandb.name_tag = f'{cfg.data.name}_run{id}_{args.model}_hyper{hyper_id}'
                wandb.init(project=f'GAE-sweep-{args.data}', id=cfg.wandb.name_tag, config=cfg, settings=wandb.Settings(_service_wait=300), save_code=True)
                wandb.watch(model, log="all",log_freq=10)

            dump_run_cfg(cfg)
            print_logger.info(f"config saved into {cfg.run_dir}")
            print_logger.info(f'Run {run_id} with seed {seed} and split {split_index} on device {cfg.device}')

            trainer = Trainer_Heart(
                FILE_PATH,
                cfg,
                model,
                None,
                data,
                optimizer,
                scheduler, 
                splits,
                run_id,
                args.repeat,
                loggers,
                print_logger,
                cfg.device,
                bool(args.wandb),
                tensorboard_writer=writer
            )

            assert not args.epoch < trainer.report_step or args.epoch % trainer.report_step, "Epochs should be divisible by report_step"
            
            trainer.train()
            trainer.finalize()
            
            run_result = {}
            for key in trainer.loggers.keys():
                print(key)
                _, _, _, test_bvalid = trainer.loggers[key].calc_run_stats(run_id, True)
                run_result[key] = test_bvalid

            # save params TODO if the updated params is saved.
            for k in hyperparameter_gnn.keys():
                if hasattr(cfg_model, k):
                    run_result[k] = getattr(cfg_model, k)
                if hasattr(cfg_score, k):
                    run_result[k] = getattr(cfg_score, k)
                elif hasattr(cfg.train, k):
                    run_result[k] = getattr(cfg.train, k)
                elif hasattr(cfg.optimizer, k):
                    run_result[k] = getattr(cfg.optimizer, k)

            run_result['epochs'] = cfg.train.epochs
            run_result['train_time'] = trainer.run_result['train_time']
            run_result['test_time'] = trainer.run_result['eval_time']
            run_result['params'] = cfg.model.params

            print_logger.info(run_result)
            to_file = f'{args.data}_{cfg.model.type}heart_tune_time_.csv'
            trainer.save_tune(run_result, to_file)

            print_logger.info(f"train time per epoch {run_result['train_time']}")
            print_logger.info(f"test time per epoch {run_result['test_time']}")
            print_logger.info(f"train time per epoch {run_result['train_time']}")
            print_logger.info(f"test time per epoch {run_result['test_time']}")
            
            if args.wandb:
                wandb.finish()
    
        
if __name__ == "__main__":
    project_main()
