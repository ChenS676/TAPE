import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np

# Assuming other necessary imports from your script
from graphgps.utility.utils import (
    set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir,
    custom_set_run_dir, set_printing, run_loop_settings, create_optimizer,
    config_device, init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
)
from torch_geometric.graphgym.config import dump_cfg, makedirs_rm_exist
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.cmd_args import parse_args
from yacs.config import CfgNode as CN
import os
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from data_utils.load import load_data_nc, load_data_lp
import optuna
from graphgps.lm_trainer.tfidf_trainer import Trainer_TFIDF
from graphgps.network.heart_gnn import mlp_model
import argparse
import wandb

FILE_PATH = f'{get_git_repo_root_path()}/'


yaml_file = {   
             'tfidf': 'core/yamls/cora/lms/tfidf.yaml',
             'word2vec': 'core/yamls/cora/lms/word2vec.yaml',
             'bert': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
            }


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='pubmed',
                        help='data name')
    parser.add_argument('--device', dest='device', required=False, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=False,
                        default=400,
                        help='data name')
    parser.add_argument('--embedder', dest='embedder_type', type=str, required=False,
                        default='tfidf',
                        help='word embedding method')
    parser.add_argument('--score', dest='score', type=str, required=False, default='mlp_score',
                        help='decoder name')
    parser.add_argument('--repeat', type=int, default=3,
                        help='The number of repeated jobs.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
   
   
def project_main():  # sourcery skip: avoid-builtin-shadow, hoist-statement-from-loop

    # process params
    args = parse_args()
    args.cfg_file = yaml_file[args.embedder_type]
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.name = args.data
    
    cfg.data.device = args.device
    cfg.decoder.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    cfg.embedder.type = args.embedder_type
   
    # save params
    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)

    torch.set_num_threads(20)
    
    train_dataset = torch.load(f'./generated_dataset/{cfg.embedder.type}_train_dataset.pt')
    train_labels = torch.load(f'./generated_dataset/{cfg.embedder.type}_train_labels.pt')
    val_dataset = torch.load(f'./generated_dataset/{cfg.embedder.type}_val_dataset.pt')
    val_labels = torch.load(f'./generated_dataset/{cfg.embedder.type}_val_labels.pt')
    test_dataset = torch.load(f'./generated_dataset/{cfg.embedder.type}_test_dataset.pt')
    test_labels = torch.load(f'./generated_dataset/{cfg.embedder.type}_test_labels.pt')

    train_dataloader = DataLoader(EmbeddingDataset(train_dataset, train_labels), batch_size=cfg.train.batch_size, shuffle=True)
    val_dataloader = DataLoader(EmbeddingDataset(val_dataset, val_labels), batch_size=cfg.train.batch_size, shuffle=False)
    test_dataloader = DataLoader(EmbeddingDataset(test_dataset, test_labels), batch_size=cfg.train.batch_size, shuffle=False)

    loggers = create_logger(args.repeat)
    
    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        print(f'run id : {run_id}')
        # Set configurations for each run TODO clean code here 

        wandb_id = wandb.util.generate_id()
        cfg.wandb.name_tag = f'{cfg.data.name}_run{wandb_id}_{args.embedder_type}{args.score}'

        custom_set_run_dir(cfg, cfg.wandb.name_tag)

        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)

        cfg = config_device(cfg)

        print_logger = set_printing(cfg)
        print_logger.info(
            f"The {cfg['data']['name']} graph with shape {train_dataloader} is loaded on {cfg.device},\n"
        )

        dump_cfg(cfg)
        in_channels = train_dataset.shape[1]

        model = mlp_model(in_channels, 
                            cfg.decoder.hidden_channels, 
                            cfg.decoder.out_channels, 
                            cfg.decoder.num_layers,
                            cfg.decoder.dropout).to(cfg.device)
        model = model.to(cfg.device)

        print_logger.info(f"{model} on {next(model.parameters()).device}" )

        cfg.decoder.params = params_count(model)
        print_logger.info(f'Num parameters: {cfg.decoder.params}')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.base_lr)

        best_val_loss = float('inf')
        best_model_path = 'fine_tuned_models/best_mlp_model.pth'

        trainer = Trainer_TFIDF(
                FILE_PATH,
                cfg,
                model,
                train_dataloader,
                val_dataloader, 
                test_dataloader,
                optimizer,
                run_id,
                args.repeat,
                loggers,
                print_logger,
                cfg.device,
                writer
            )

        assert not args.epoch < trainer.report_step or args.epoch % trainer.report_step, "Epochs should be divisible by report_step"
        
        trainer.train()
    
    
if __name__ == '__main__':
    project_main()