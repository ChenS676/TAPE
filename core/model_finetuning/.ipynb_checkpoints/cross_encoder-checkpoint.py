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
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from torch.utils.data import DataLoader

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

dataset = []
pos_train_edge_index = splits['train'].pos_edge_label_index
neg_train_edge_index = splits['train'].neg_edge_label_index

pos_val_edge_index = splits['valid'].pos_edge_label_index
neg_val_edge_index = splits['valid'].neg_edge_label_index

pos_test_edge_index = splits['test'].pos_edge_label_index
neg_test_edge_index = splits['test'].neg_edge_label_index

# Process positive edges
for i in range(pos_train_edge_index.shape[1]):
    node1 = pos_train_edge_index[0, i].item()
    node2 = pos_train_edge_index[1, i].item()
    text1 = text[node1]
    text2 = text[node2]
    dataset.append(InputExample(texts=[text1, text2], label=1))

# Process negative edges
for i in range(neg_train_edge_index.shape[1]):
    node1 = neg_train_edge_index[0, i].item()
    node2 = neg_train_edge_index[1, i].item()
    text1 = text[node1]
    text2 = text[node2]
    dataset.append(InputExample(texts=[text1, text2], label=0))

val_dataset = []
val_labels = []
# Process positive edges
for i in range(pos_val_edge_index.shape[1]):
    node1 = pos_val_edge_index[0, i].item()
    node2 = pos_val_edge_index[1, i].item()
    text1 = text[node1]
    text2 = text[node2]
    val_dataset.append([text1, text2])
    val_labels.append(1)

# Process negative edges
for i in range(neg_val_edge_index.shape[1]):
    node1 = neg_val_edge_index[0, i].item()
    node2 = neg_val_edge_index[1, i].item()
    text1 = text[node1]
    text2 = text[node2]
    val_dataset.append([text1, text2])
    val_labels.append(0)
    
test_dataset = []
test_labels = []
# Process positive edges
for i in range(pos_test_edge_index.shape[1]):
    node1 = pos_test_edge_index[0, i].item()
    node2 = pos_test_edge_index[1, i].item()
    text1 = text[node1]
    text2 = text[node2]
    test_dataset.append([text1, text2])
    test_labels.append(1)

# Process negative edges
for i in range(neg_test_edge_index.shape[1]):
    node1 = neg_test_edge_index[0, i].item()
    node2 = neg_test_edge_index[1, i].item()
    text1 = text[node1]
    text2 = text[node2]
    test_dataset.append([text1, text2])
    test_labels.append(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrossEncoder('sentence-transformers/all-mpnet-base-v2', num_labels=2, device=device)

train_dataloader = DataLoader(dataset, shuffle=True, batch_size=16)

print(len(dataset))

model.fit(
    train_dataloader=train_dataloader,
    epochs=5,
    warmup_steps=500
)

scores = model.predict(test_dataset)
test_pred = []
for score in scores:
    pred_label = np.argmax(score)
    test_pred.append(pred_label)

print(accuracy_score(test_labels, test_pred))
print(roc_auc_score(test_labels, test_pred))
print(confusion_matrix(test_labels, test_pred))
