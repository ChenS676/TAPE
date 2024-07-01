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
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
# Assuming other necessary imports from your script
from graphgps.utility.utils import (
    set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir,
    custom_set_run_dir, set_printing, run_loop_settings, create_optimizer,
    config_device, init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
)
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from heuristic.eval import get_metric_score


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
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Sample dataset
np.random.seed(0)
torch.manual_seed(0)

    
embedding_model_name = "tfidf"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Convert to tensors
train_dataset = torch.load(f'./generated_dataset/{embedding_model_name}_train_dataset.pt')
train_labels = torch.load(f'./generated_dataset/{embedding_model_name}_train_labels.pt')
val_dataset = torch.load(f'./generated_dataset/{embedding_model_name}_val_dataset.pt')
val_labels = torch.load(f'./generated_dataset/{embedding_model_name}_val_labels.pt')
test_dataset = torch.load(f'./generated_dataset/{embedding_model_name}_test_dataset.pt')
test_labels = torch.load(f'./generated_dataset/{embedding_model_name}_test_labels.pt')

from sklearn.linear_model import RidgeClassifier

clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
clf.fit(train_dataset, train_labels)
pred = clf.predict(test_dataset)
acc = sum(np.asarray(test_labels) == pred) / len(test_labels)    

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(random_state=1, max_iter=300).fit(train_dataset, train_labels)
test_proba = clf.predict_proba(test_dataset)
test_pred = clf.predict(test_dataset)
acc = clf.score(test_dataset, test_labels)

y_pos_pred, y_neg_pred = test_pred[test_labels == 1], test_pred[test_labels == 0]
evaluator_hit = Evaluator(name='ogbl-collab')
evaluator_mrr = Evaluator(name='ogbl-citation2')
metrics = get_metric_score(evaluator_hit, evaluator_mrr, y_pos_pred, y_neg_pred)

print(f'Accuracy: {acc:.4f}')
print(f'metrics : {metrics}')

exit(-1)
hidden_size = 1024
learning_rate = 1e-5
batch_size = 64
patience = 5
weight_decay = 1e-3
num_epochs = 200

train_dataloader = DataLoader(EmbeddingDataset(train_dataset, train_labels), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(EmbeddingDataset(val_dataset, val_labels), batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(EmbeddingDataset(test_dataset, test_labels), batch_size=batch_size, shuffle=False)

input_size = train_dataset.shape[1]
num_classes = 2

model = MLP(input_size, hidden_size, num_classes)
model.apply(init_weights)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

best_f1_score = 0
best_model_path = 'best_mlp_model.pth'
early_stopping_counter = 0

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
    val_correct = 0
    val_total = 0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_loss /= len(val_dataloader)
    val_accuracy = val_correct / val_total
    val_f1 = f1_score(val_targets, val_preds)

    if val_f1 > best_f1_score:
        best_f1_score = val_f1
        torch.save(model.state_dict(), best_model_path)
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print("Early stopping triggered")
        break

    scheduler.step(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss / len(train_dataloader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1 Score: {val_f1:.4f}')

# Load the best model for evaluation
model.load_state_dict(torch.load(best_model_path))
model.eval()

test_preds = []
test_targets = []

with torch.no_grad():
    for embeddings, labels in test_dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        outputs = model(embeddings)

        _, predicted = torch.max(outputs.data, 1)

        test_preds.extend(predicted.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

print(f'Accuracy: {accuracy_score(test_targets, test_preds):.4f}')
print(f'ROC AUC: {roc_auc_score(test_targets, test_preds):.4f}')
print('Confusion Matrix:')
print(confusion_matrix(test_targets, test_preds))
print('F1_score:', f1_score(test_targets, test_preds))