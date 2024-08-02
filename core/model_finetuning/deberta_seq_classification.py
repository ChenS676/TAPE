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
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
writer = SummaryWriter()

class TextPairDataset(Dataset):
    def __init__(self, texts1, texts2, labels, tokenizer, max_length=512):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text1, text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

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
        dataset.append([text1, text2])
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

train_text_one = [row[0] for row in train_dataset]
train_text_two = [row[1] for row in train_dataset]
train_dataset = TextPairDataset(train_text_one, train_text_two, train_labels, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

train_text_one = [row[0] for row in val_dataset]
train_text_two = [row[1] for row in val_dataset]
val_dataset = TextPairDataset(val_text_one, val_text_two, val_labels, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)


# Training loop (simplified for demonstration purposes)
accumulation_steps = 200
best_val_accuracy = 0.0
patience = 10
early_stop_counter = 0
num_epochs = 1

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()

for epoch in range(num_epochs):  # Adjust the number of epochs as needed
    epoch_loss = 0.0
    model.train()
    
    for i, batch in enumerate(tqdm(train_dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accumulation_steps  # Normalize loss

        loss.backward()
        epoch_loss += loss.item()

        # Perform optimizer step and zero gradients after accumulating enough gradients
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            print(f"TRAIN: Epoch {epoch}, Batch {i+1}, Loss: {loss.item() * accumulation_steps}")  # Print unnormalized loss

            # Validation step
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    val_labels = val_batch['labels'].to(device)

                    val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
                    val_loss += val_outputs.loss.item()
                    
                    # Calculate accuracy
                    predictions = torch.argmax(val_outputs.logits, dim=-1)
                    val_correct += (predictions == val_labels).sum().item()
                    val_total += val_labels.size(0)

            val_loss /= len(val_dataloader)
            val_accuracy = val_correct / val_total

            print(f"VAL: Epoch {epoch}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

            # Check for model improvement and save the best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                #torch.save(model.state_dict(), 'best_model.pth')
                print("Best model saved.")
                early_stop_counter = 0  # Reset counter if validation accuracy improves
            else:
                early_stop_counter += 1

            # Early stopping
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    # If the number of batches is not perfectly divisible by accumulation_steps
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}, Batch {i+1}, Loss: {loss.item() * accumulation_steps}")
