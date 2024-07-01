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
from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir, custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
from torch_geometric.graphgym.config import dump_cfg, makedirs_rm_exist
from data_utils.load import load_data_nc, load_data_lp
from sklearn.feature_extraction.text import TfidfVectorizer

def get_embeddings(text):
    if embeddig_model_type == "tfidf":
        test_dataset
    return embedding_model.encode(text)

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def process_edges(pos_edge_index, neg_edge_index, text):
    dataset = []
    labels = []
    # Process positive edges
    for i in tqdm(range(pos_edge_index.shape[1])):
        node1 = pos_edge_index[0, i].item()
        node2 = pos_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        embedding_text1 = get_embeddings(text1)
        embedding_text2 = get_embeddings(text2)
        combined_embedding = np.concatenate((embedding_text1, embedding_text2))
        dataset.append(combined_embedding)
        labels.append(1)

    # Process negative edges
    for i in tqdm(range(neg_edge_index.shape[1])):
        node1 = neg_edge_index[0, i].item()
        node2 = neg_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        embedding_text1 = get_embeddings(text1)
        embedding_text2 = get_embeddings(text2)
        combined_embedding = np.concatenate((embedding_text1, embedding_text2))
        dataset.append(combined_embedding)
        labels.append(0)
    
    return dataset, labels

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

embedding_model_name = "tfidf"


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if embedding_model_name == "tfidf":
    # Prepare train, validation, and test datasets
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
    vectorizer = TfidfVectorizer()
    train_dataset = vectorizer.fit_transform(train_dataset).toarray()
    val_dataset = vectorizer.transform(val_dataset).toarray()
    test_dataset  = vectorizer.transform(test_dataset).toarray()
else:
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # Prepare train, validation, and test datasets
    train_dataset, train_labels = process_edges(
        splits['train'].pos_edge_label_index, 
        splits['train'].neg_edge_label_index, 
        text
    )
    val_dataset, val_labels = process_edges(
        splits['valid'].pos_edge_label_index, 
        splits['valid'].neg_edge_label_index, 
        text
    )
    test_dataset, test_labels = process_edges(
        splits['test'].pos_edge_label_index, 
        splits['test'].neg_edge_label_index, 
        text
    )


# Convert to tensors
train_dataset = torch.tensor(train_dataset, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_dataset = torch.tensor(val_dataset, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.long)
test_dataset = torch.tensor(test_dataset, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Save datasets
torch.save(train_dataset, f'./data/{embedding_model_name}_train_dataset.pt')
torch.save(train_labels, f'./data/{embedding_model_name}_train_labels.pt')
torch.save(val_dataset, f'./data/{embedding_model_name}_val_dataset.pt')
torch.save(val_labels, f'./data/{embedding_model_name}_val_labels.pt')
torch.save(test_dataset, f'./data/{embedding_model_name}_test_dataset.pt')
torch.save(test_labels, f'./data/{embedding_model_name}_test_labels.pt')