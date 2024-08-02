import os
import sys
import time
from os.path import abspath, dirname, join
from torch.nn import BCEWithLogitsLoss
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

import torch
import wandb
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from heuristic.eval import get_metric_score
from graphgps.utility.utils import (
    set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir,
    custom_set_run_dir, set_printing, run_loop_settings, create_optimizer,
    config_device, init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
)
from torch_geometric.graphgym.config import dump_cfg, makedirs_rm_exist
from data_utils.load import load_data_nc, load_data_lp
from typing import Dict, Tuple
from scipy.sparse._csr import csr_matrix 
from graphgps.train.opt_train import (Trainer)
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
writer = SummaryWriter()

def process_texts(pos_edge_index, neg_edge_index, text):
    dataset = []
    labels = []
    # Process positive edges
    for i in tqdm(range(pos_edge_index.shape[1])):
        node1 = pos_edge_index[0, i].item()
        node2 = pos_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        combined_text = text1 + " " + text2
        dataset.append(combined_text)
        labels.append(1)

    # Process negative edges
    for i in tqdm(range(neg_edge_index.shape[1])):
        node1 = neg_edge_index[0, i].item()
        node2 = neg_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        combined_text = text1 + " " + text2
        dataset.append(combined_text)
        labels.append(0)
    
    return dataset, labels

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model_id = "microsoft/deberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

evaluator_hit = Evaluator(name='ogbl-collab')
evaluator_mrr = Evaluator(name='ogbl-citation2')

FILE_PATH = f'{get_git_repo_root_path()}/'

args = parse_args()
# Load args file

cfg = set_cfg(FILE_PATH, args.cfg_file)
cfg.merge_from_list(args.opts)
custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
dump_cfg(cfg)

# Set Pytorch environment
torch.set_num_threads(cfg.run.num_threads)

loggers = create_logger(args.repeat)

splits, text = load_data_lp[cfg.data.name](cfg.data)

train_dataset, train_labels = process_texts(
    splits['train'].pos_edge_label_index, 
    splits['train'].neg_edge_label_index, 
    text
)
val_dataset, val_labels = process_texts(
    splits['valid'].pos_edge_label_index, 
    splits['valid'].neg_edge_label_index, 
    text
)
test_dataset, test_labels = process_texts(
    splits['test'].pos_edge_label_index, 
    splits['test'].neg_edge_label_index, 
    text
)

