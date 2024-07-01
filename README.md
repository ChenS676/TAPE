# Benchmark TAG 

<img src="./overview.svg">


## 0.0 Python environment setup with Conda
```
conda create --name EAsF python=3.10
conda activate EAsF

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

## 0.1 Here is my install examples in horeka server with cuda113
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
## 0.2 Here is my install examples in haicore server
```
module load devel/cmake/3.26
module load compiler/intel/2023.1.0_llvm
module load devel/cuda/11.8

Currently Loaded Modules:
  1) devel/cmake/3.26   2) compiler/intel/2023.1.0_llvm   3) devel/cuda/11.8 (E)

  Where:
   E:  Experimental

# install pytorch 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

python -c "import torch; print(torch.__version__)" 
2.3.1
python -c "import torch; print(torch.version.cuda)"
11.8

nvcc --version 
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
pip install tqdm wandb pandas ogb yacs
```

## 0.1 Here is my install examples in horeka server cuda 118
```
Currently Loaded Modules:
  1) devel/cmake/3.18   2) compiler/gnu/13   3) devel/cuda/11.8

conda create --name TAPE3 python=3.10
conda activate TAPE3
# install pytorch 
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"


pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia


pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
pip install torch_geometric
pip install ogb yacs pandas wandb
pip install fast-pagerank datasketch ogb
pip install --upgrade accelerate

# test your installed pytorch geometric 
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.graphgym.config import cfg
from torch_geometric import seed_everything
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
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

## 2. Fine-tuning the LMs
To use the orginal text attributes in custom_main.py
```
....

splits, _, data = load_data_lp[cfg.data.name](cfg.data)

# LLM: finetuning
if cfg.train.finetune: 
    # load custom embedding 
    #  basically data.x = $your embedding in tensor
    data = init_model_from_pretrained(model, cfg.train.finetune,
                                        cfg.train.freeze_pretrained)
...
```

### To load pretrained embedding
```

```


## 3. Training the GNNs
### To use different GNN models

## 4. Reproducibility
