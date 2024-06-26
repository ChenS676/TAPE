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
import optuna

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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.softmax(out)
        return out

    
embedding_model_name = "tfidf"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Convert to tensors
train_dataset = torch.load(f'./data/{embedding_model_name}_train_dataset.pt')
train_labels = torch.load(f'./data/{embedding_model_name}_train_labels.pt')
val_dataset = torch.load(f'./data/{embedding_model_name}_val_dataset.pt')
val_labels = torch.load(f'./data/{embedding_model_name}_val_labels.pt')
test_dataset = torch.load(f'./data/{embedding_model_name}_test_dataset.pt')
test_labels = torch.load(f'./data/{embedding_model_name}_test_labels.pt')

def train_model(trial):
    hidden_size = trial.suggest_int('hidden_size', 32, 512)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

    # Create DataLoader for embeddings and labels
    train_dataloader = DataLoader(EmbeddingDataset(train_dataset, train_labels), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(EmbeddingDataset(val_dataset, val_labels), batch_size=batch_size, shuffle=False)

    input_size = train_dataset.shape[1]
    num_classes = 2

    # Initialize MLP model
    model = MLP(input_size, hidden_size, num_classes)
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for embeddings, labels in train_dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for embeddings, labels in val_dataloader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

def objective(trial):
    return train_model(trial)

# Hyperparameter tuning using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2)

# Get the best hyperparameters
best_params = study.best_params
print(f'Best hyperparameters: {best_params}')

# Train final model with the best hyperparameters
hidden_size = best_params['hidden_size']
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']

train_dataloader = DataLoader(EmbeddingDataset(train_dataset, train_labels), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(EmbeddingDataset(val_dataset, val_labels), batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(EmbeddingDataset(test_dataset, test_labels), batch_size=batch_size, shuffle=False)

input_size = train_dataset.shape[1]
num_classes = 2

model = MLP(input_size, hidden_size, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 15

best_val_loss = float('inf')
best_model_path = 'fine_tuned_models/best_mlp_model.pth'

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for embeddings, labels in train_dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for embeddings, labels in val_dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss / len(train_dataloader):.4f}, Val Loss: {val_loss:.4f}')

# Load the best model for evaluation
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for embeddings, labels in test_dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        outputs = model(embeddings)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f'Accuracy: {accuracy_score(all_labels, all_preds):.4f}')
print(f'ROC AUC: {roc_auc_score(all_labels, all_preds):.4f}')
print('Confusion Matrix:')
print(confusion_matrix(all_labels, all_preds))