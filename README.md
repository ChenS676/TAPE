# TAG4LP
TAG4LP is a project to share the public text-attributed graph (TAG) datasets and benchmark the performance of the different baseline methods for link prediction
We welcome more to share datasets that are valuable for TAGs research.


## Datasets 🔔
We collect and construct 8 TAG datasets from Cora, Pubmed, Arxiv\_2023, ogbn-paper100M, citationv8, paperwithcode API, History, Photo.
Now you can go to the 'Files and version' in [TAG4LP](https://drive.google.com/file/d/15ZWzRESVpNFowt3zfm3v8-5DGdnMjFzk/view?usp=drive_link) to find the datasets we upload! 
In each dataset folder, you can find the **csv** file (which save the text attribute of the dataset), **pt** file (which represent the pyg graph file).
You can use the node initial feature we created, and you also can extract the node feature from our code under core/data_utils. 


## Environments
You can quickly install the corresponding dependencies
```shell
conda env create -f environment.yml
```

## Pipeline 🎮
We describe below how to use our repository to perform the experiments reported in the paper. We are also adjusting the style of the repository to make it easier to use.
(Please complete the ['Datasets'](get-tapedataset.sh) above first)
### 1. LMGJoint for Link Prediction
You can use Pwc_small Cora PubMed Arxiv_2023 Pwc_medium Ogbn_arxiv Citationv8 ogbn-papers100M.

```python
WANDB_DISABLED=True CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node 4 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/cora/lms/ft-minilm.yaml
```

### 2. Reconstruct Baseline for PLM-related work 
```
WANDB_DISABLED=True CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node 4 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/cora/lms/ft-llama.yaml
```

## Config
all configuration file can be found be in core/yamls

# Experiment Setup and Execution

This repository provides commands and setup details for running various machine learning models and methods on datasets such as **History**, **Photo**, and others. Each method is executed 5 times using seeds `[0, 1, 2, 3, 4]`.

---

## **TF-IDF and Word2Vec (W2V)**

### **Execution Commands**

#### History
```bash
python3 lp_edge_embed.py --data history --device cuda:0 --embedder w2v --epochs 1000
python3 lp_edge_embed.py --data history --device cuda:0 --embedder tfidf --epochs 100
```

#### Photo
```bash
python3 lp_edge_embed.py --data photo --device cpu --embedder w2v --epochs 1000
python3 lp_edge_embed.py --data photo --device cpu --embedder tfidf --epochs 100
```

- **Note**: 
  - TF-IDF is trained for 100 epochs to avoid overfitting.
  - W2V is trained for 1000 epochs (can be extended for better results).

### **Results**
```python
Photo:
W2V: 64.71 ± 0.12
TF-IDF: 62.80 ± 0.62

History:
W2V: 65.54 ± 0.21
TF-IDF: 60.94 ± 0.48
```

---

## **BERT/e5-Large/MiniLM/LLama**
#### History
```bash
python3 core/embedding_mlp/embedding_LLM_main.py --data history --cfg core/yamls/history/lms/minilm.yaml --downsampling 0.1 --device cuda:2
python3 core/embedding_mlp/embedding_LLM_main.py --data history --cfg core/yamls/history/lms/e5-large.yaml --downsampling 0.1 --device cuda:2
python3 core/embedding_mlp/embedding_LLM_main.py --data history --cfg core/yamls/history/lms/llama.yaml --downsampling 0.1 --device cuda:2
```

#### Photo
```bash
python3 core/embedding_mlp/embedding_LLM_main.py --data photo --cfg core/yamls/photo/lms/minilm.yaml --downsampling 0.1 --device cuda:2
python3 core/embedding_mlp/embedding_LLM_main.py --data photo --cfg core/yamls/photo/lms/e5-large.yaml --downsampling 0.1 --device cuda:2
python3 core/embedding_mlp/embedding_LLM_main.py --data photo --cfg core/yamls/photo/lms/llama.yaml --downsampling 0.1 --device cuda:2
```

- **LLaMA Usage**: 
  - Export HuggingFace token before running:
    ```bash
    export HUGGINGFACE_HUB_TOKEN=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
    ```

---

## **NCN/NCNC**
#### History
```bash
python3 core/gcns/ncn_main.py --cfg core/yamls/history/gcns/ncn.yaml --data history --device cuda:0 --epochs 300 --downsampling 0.1
python3 core/gcns/ncn_main.py --cfg core/yamls/history/gcns/ncnc.yaml --data history --device cuda:0 --epochs 300 --downsampling 0.1
```

#### Photo
```bash
python3 core/gcns/ncn_main.py --cfg core/yamls/photo/gcns/ncn.yaml --data photo --device cuda:0 --epochs 300 --downsampling 0.1
python3 core/gcns/ncn_main.py --cfg core/yamls/photo/gcns/ncnc.yaml --data photo --device cuda:0 --epochs 300 --downsampling 0.1
```

---

## **LM-MLP-FT**
#### History
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=True python3 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/history/lms/ft-minilm.yaml --downsampling 0.1
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=True python3 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/history/lms/ft-mpnet.yaml --downsampling 0.1
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=True python3 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/history/lms/e5-large.yaml --downsampling 0.1
```

#### Photo
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=True python3 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/photo/lms/ft-minilm.yaml --downsampling 0.1
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=True python3 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/photo/lms/ft-mpnet.yaml --downsampling 0.1
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=True python3 core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/photo/lms/e5-large.yaml --downsampling 0.1
```

---

## **HL-GNN**
```bash
python3 core/gcns/hlgnn_main.py --cfg core/yamls/cora/gcns/hlgnn.yaml --data cora --device cuda:0 --epochs 100
python3 core/gcns/hlgnn_main.py --cfg core/yamls/pubmed/gcns/hlgnn.yaml --data pubmed --device cuda:0 --epochs 300
python3 core/gcns/hlgnn_main.py --cfg core/yamls/arxiv_2023/gcns/hlgnn.yaml --data arxiv_2023 --device cuda:0 --epochs 500
```

---

## **NCN + LLM**
#### History
```bash
python3 core/gcns/LLM_embedding_ncn_main.py --data history --cfg core/yamls/history/gcns/ncn.yaml --embedder minilm --device cuda:0 --downsampling 0.1 --epochs 100
python3 core/gcns/LLM_embedding_ncn_main.py --data history --cfg core/yamls/history/gcns/ncn.yaml --embedder e5-large --device cuda:0 --downsampling 0.1 --epochs 100
```

#### Photo
```bash
python3 core/gcns/LLM_embedding_ncn_main.py --data photo --cfg core/yamls/photo/gcns/ncn.yaml --embedder minilm --device cuda:0 --downsampling 0.1 --epochs 100
python3 core/gcns/LLM_embedding_ncn_main.py --data photo --cfg core/yamls/photo/gcns/ncn.yaml --embedder e5-large --device cuda:0 --downsampling 0.1 --epochs 100
```

---

## **LMGJoint**
#### History
```bash
CUDA_LAUNCH_BLOCKING=1 python core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/history/lms/ft-minilm.yaml --decoder core/yamls/history/gcns/ncn.yaml --repeat 1
CUDA_LAUNCH_BLOCKING=1 python core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/history/lms/ft-e5-large.yaml --decoder core/yamls/history/gcns/ncn.yaml --repeat 1
```

#### Photo
```bash
CUDA_LAUNCH_BLOCKING=1 python core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/photo/lms/ft-minilm.yaml --decoder core/yamls/photo/gcns/ncn.yaml --repeat 1
CUDA_LAUNCH_BLOCKING=1 python core/finetune_embedding_mlp/lm_trainer.py --cfg core/yamls/photo/lms/ft-e5-large.yaml --decoder core/yamls/photo/gcns/ncn.yaml --repeat 1
```
