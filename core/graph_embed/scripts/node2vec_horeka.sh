#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --partition=cpuonly
#SBATCH --job-name=tag_node2vec
#SBATCH --mem-per-cpu=1600mb

#SBATCH --output=log/TAG_Benchmar_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error


#SBATCH --chdir=/hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE_emb/core/graph_embed/res_outputs_emb

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhelson.g@gmail.com
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate TAG-LP
cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_gerrman/TAPE_emb/core/graph_embed
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


ls -ltr

# python3 node2vec_tagplus.py --sweep core/yamls/cora/embedding/n2v_sweep3.yaml --cfg core/yamls/cora/embedding/node2vec.yaml 
# python3 node2vec_tagplus.py --sweep core/yamls/pubmed/embedding/n2v_sweep.yaml --cfg core/yamls/pubmed/embedding/node2vec.yaml 
python3 node2vec_tagplus.py --sweep core/yamls/arxiv_2023/embedding/n2v_sweep3.yaml --cfg core/yamls/arxiv_2023/embedding/node2vec.yaml 

# python3 node2vec_tag.py --sweep core/yamls/cora/embedding/n2v_sweep3.yaml --cfg core/yamls/cora/embedding/node2vec.yaml 
# python3 node2vec_tag.py --sweep core/yamls/pubmed/embedding/n2v_sweep.yaml --cfg core/yamls/pubmed/embedding/node2vec.yaml 
# python3 node2vec_tag.py --sweep core/yamls/arxiv_2023/embedding/n2v_sweep3.yaml --cfg core/yamls/arxiv_2023/embedding/node2vec.yaml 

# python wb_tune_struc2vec.py --sweep core/yamls/cora/struc2vec_sp1.yaml --cfg core/yamls/cora/struc2vec.yaml 
# python wb_tune_struc2vec.py --sweep core/yamls/cora/struc2vec_sp2.yaml --cfg core/yamls/cora/struc2vec.yaml 
# python wb_tune_struc2vec.py --sweep core/yamls/cora/struc2vec_sp3.yaml --cfg core/yamls/cora/struc2vec.yaml 


