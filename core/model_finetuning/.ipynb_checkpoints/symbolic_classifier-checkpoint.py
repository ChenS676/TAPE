import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import sys
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

# Assuming other necessary imports from your script
from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir, custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
from torch_geometric.graphgym.config import dump_cfg, makedirs_rm_exist
from data_utils.load import load_data_nc, load_data_lp
import optuna
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import Evaluator


embedding_model_name = "tfidf"

# Load datasets
train_dataset = torch.load(f'./data/{embedding_model_name}_train_dataset.pt').numpy()
train_labels = torch.load(f'./data/{embedding_model_name}_train_labels.pt').numpy()
val_dataset = torch.load(f'./data/{embedding_model_name}_val_dataset.pt').numpy()
val_labels = torch.load(f'./data/{embedding_model_name}_val_labels.pt').numpy()
test_dataset = torch.load(f'./data/{embedding_model_name}_test_dataset.pt').numpy()
test_labels = torch.load(f'./data/{embedding_model_name}_test_labels.pt').numpy()

# Combine train and validation datasets for cross-validation
X_train = np.concatenate((train_dataset, val_dataset), axis=0)
y_train = np.concatenate((train_labels, val_labels), axis=0)

"""# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best hyperparameters: {best_params}')

# Train the final model with the best hyperparameters
best_rf = RandomForestClassifier(**best_params)"""
best_rf = RandomForestClassifier()
best_rf.fit(X_train, y_train)

# Evaluate on the test set
y_pred = best_rf.predict(test_dataset)
y_pred_proba = best_rf.predict_proba(test_dataset)[:, 1]

# Print evaluation metrics
print(f'Accuracy: {accuracy_score(test_labels, y_pred):.4f}')
print(f'ROC AUC: {roc_auc_score(test_labels, y_pred_proba):.4f}')
print('Confusion Matrix:')
print(confusion_matrix(test_labels, y_pred))
