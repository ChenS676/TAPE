import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
from torch_geometric.graphgym.config import (dump_cfg, 
                                             makedirs_rm_exist)
from sklearn.metrics import *
import torch
from data_utils.load import load_data_nc, load_data_lp
import numpy as np
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

def process_edges(pos_edge_index, neg_edge_index, text):
    texts = []
    label = []
    # Process positive edges
    for i in tqdm(range(pos_edge_index.shape[1])):
        node1 = pos_edge_index[0, i].item()
        node2 = pos_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        texts.append(text1 + ". " + text2)
        label.append(1)


    # Process negative edges
    for i in tqdm(range(neg_edge_index.shape[1])):
        node1 = neg_edge_index[0, i].item()
        node2 = neg_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        texts.append(text1 + ". " + text2)
        label.append(1)
        
    data_dict = {
        "text": texts,
        "label": label
    }
    
    return data_dict

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = process_edges(
    splits['train'].pos_edge_label_index, 
    splits['train'].neg_edge_label_index, 
    text
)

val_dataset = process_edges(
    splits['valid'].pos_edge_label_index, 
    splits['valid'].neg_edge_label_index, 
    text
)

test_dataset = process_edges(
    splits['test'].pos_edge_label_index, 
    splits['test'].neg_edge_label_index, 
    text
)

train_data = Dataset.from_dict(train_dataset)
val_data = Dataset.from_dict(val_dataset)
test_data = Dataset.from_dict(test_dataset)

dataset = DatasetDict(
    {'train': train_data,
     'val': val_data,
     'test': test_data
    })

model_name = "microsoft/deberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_data = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()