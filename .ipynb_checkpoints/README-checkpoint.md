# Benchmark TAG 

<img src="./overview.svg">


## 0.0 Python environment setup with Conda
```
conda create --name TAPE python=3.8
conda activate TAPE

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c dglteam/label/cu113 dgl
pip install yacs
pip install transformers
pip install --upgrade accelerate
pip install gitpython
pip install ipython
pip install wandb
```

## 0.1 Here is my install examples in horeka server
```

Currently Loaded Modules:
  1) devel/cmake/3.18   2) devel/cuda/10.2   3) devel/cudnn/10.2 (E)   4) compiler/gnu/11.1

  Where:
   E:  Experimental

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu114
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
pip install ogb
conda install -c dglteam/label/cu113 dgl
pip install yacs
pip install transformers
pip install --upgrade accelerate

```

## 0.2 Horka Server installation 2.0
```
module purge
module load compiler/intel/2023.1.0
module load devel/cuda/11.8

conda create --name TAPE python=3.9
conda activate TAPE

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
conda install scikit-learn
conda install -c pyg pyg
pip install ogb
pip install transformers
pip install gitpython
pip install ipython
pip install yacs
pip install sentence-transformers
pip install wandb
pip install python-dotenv
pip install sentencepiece
```

## 1. Download/Test TAG datasets 

```
bash core/scripts/get-tapedataset.sh 
python load_arxiv_2023.py 
python load_cora.py
python load_ogbn_arxiv.py
python load_products.py
python load_pubmed.py
#TODO add paperwithcode dataset
#TODO use SemOpenAlex
```

In case, you have issue [#43](https://github.com/wkentaro/gdown/issues/43), please try solution [1](https://github.com/wkentaro/gdown/issues/43#issuecomment-1892954390), [2](https://stackoverflow.com/questions/65312867/how-to-download-large-file-from-google-drive-from-terminal-gdown-doesnt-work).

### A. Original Text Attributes
All graph encoder modules including node encoder and edge encoder are implemented in GraphGym transferred from [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/graphgym.html#).

#### FeatNodeEncoder

### B. LLM responses

## 2. Fine-tuning the LMs
### To use the orginal text attributes
### To use the GPT responses



## 3. Training the GNNs
### To use different GNN models

## 4. Reproducibility
