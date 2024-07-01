import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils.load import load_data_lp
from graphgps.utility.utils  import set_cfg, parse_args, get_git_repo_root_path

# from utils import set_cfg, parse_args, get_git_repo_root_path
from sentence_transformers import SentenceTransformer
from typing import List
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from openai import OpenAI
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses
import torch.nn as nn
from graphgps.utility.utils import Logger, save_emb, get_root_dir, get_logger, config_device
from time import time
from sklearn.linear_model import RidgeClassifier

from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from typing import Dict
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import pandas as pd

FILE_PATH = f'{get_git_repo_root_path()}/'

args = parse_args()

cfg = set_cfg(FILE_PATH, args.cfg_file)
cfg.merge_from_list(args.opts)
cfg = config_device(cfg)
splits, text, _ = load_data_lp[cfg.data.name](cfg.data)

categories = [
    "pos",
    "neg",
]

def create_data(splits: Dict, text: List[str], key: str):
    dataset, target = [], []
    pos_edge_index = splits[key].pos_edge_label_index
    neg_edge_index = splits[key].neg_edge_label_index

    # Process positive edges
    for i in range(pos_edge_index.shape[1]):
        node1 = pos_edge_index[0, i].item()
        node2 = pos_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        dataset.append(text1+'CLS'+text2)
        target.append(1)
    # Process negative edges
    for i in range(neg_edge_index.shape[1]):
        node1 = neg_edge_index[0, i].item()
        node2 = neg_edge_index[1, i].item()
        text1 = text[node1]
        text2 = text[node2]
        dataset.append(text1+'CLS'+text2)
        target.append(0)
    return dataset, target


def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


def load_dataset(verbose=False, remove=()):
    """Load and vectorize the 20 newsgroups dataset."""

    data_train, target_train = create_data(splits, text, 'train')

    data_test, target_test =  create_data(splits, text, 'test')

    # Extracting features from the training data using a sparse vectorizer
    t0 = time()
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
    X_train = vectorizer.fit_transform(data_train) 
    duration_train = time() - t0

    # Extracting features from the test data using the same vectorizer
    t0 = time()
    X_test = vectorizer.transform(data_test)
    duration_test = time() - t0

    feature_names = vectorizer.get_feature_names_out()

    if verbose:
        # compute size of loaded data
        data_train_size_mb = size_mb(data_train)
        data_test_size_mb = size_mb(data_test)

        print(
            f"{len(data_train)} documents - "
            f"{data_train_size_mb:.2f}MB (training set)"
        )
        print(f"{len(data_test)} documents - {data_test_size_mb:.2f}MB (test set)")
        print(
            f"vectorize training done in {duration_train:.3f}s "
            f"at {data_train_size_mb / duration_train:.3f}MB/s"
        )
        print(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
        print(
            f"vectorize testing done in {duration_test:.3f}s "
            f"at {data_test_size_mb / duration_test:.3f}MB/s"
        )
        print(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

    return X_train, X_test, target_train, target_test, feature_names, categories

X_train, X_test, y_train, y_test, feature_names, categories = load_dataset(
    verbose=True
)

import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import numpy as np
from gensim.models import KeyedVectors

def get_tf_idf_by_texts(texts, known_mask, test_mask, max_features=1433, use_tokenizer=False):
    if known_mask is None and test_mask is None:
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)
        X = tf_idf_vec.fit_transform(texts)
        torch_feat = torch.FloatTensor(X.todense())
        norm_torch_feat = F.normalize(torch_feat, dim=-1)
        return torch_feat, norm_torch_feat
    
    if use_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="/tmp")
        tf_idf_vec = TfidfVectorizer(analyzer="word", max_features=500, tokenizer=lambda x: tokenizer.tokenize(x, max_length=512, truncation=True))
    else:
        tf_idf_vec = TfidfVectorizer(stop_words='english', max_features=max_features)

    text_known = [texts[i] for i in range(len(texts)) if known_mask[i]]
    text_test = [texts[i] for i in range(len(texts)) if test_mask[i]]

    x_known = tf_idf_vec.fit_transform(text_known)
    x_test = tf_idf_vec.transform(text_test)

    x_known = torch.FloatTensor(x_known.todense())
    x_test = torch.FloatTensor(x_test.todense())

    dim = x_known.shape[1]
    torch_feat = torch.zeros(len(texts), dim)
    torch_feat[known_mask] = x_known 
    torch_feat[test_mask] = x_test

    norm_torch_feat = F.normalize(torch_feat, dim=-1)
    return torch_feat, norm_torch_feat

# Example usage
texts = [
    "This is a sample document.",
    "This document is another example.",
    "Yet another example of a document.",
    "This is the last sample text."
]

known_mask = [True, True, False, False]
test_mask = [False, False, True, True]

torch_feat, norm_torch_feat = get_tf_idf_by_texts(texts, known_mask, test_mask, max_features=10, use_tokenizer=False)

print("TF-IDF Features:\n", torch_feat)
print("Normalized TF-IDF Features:\n", norm_torch_feat)
