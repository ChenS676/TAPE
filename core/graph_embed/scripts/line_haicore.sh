#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --gres=gpu:full:4
#SBATCH --partition=normal
#SBATCH --job-name=tag_struc2vec
#SBATCH --mem-per-cpu=1600mb

#SBATCH --output=log/TAG_Benchmar_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error


#SBATCH --chdir=/hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/graph_embed/res_outputs_emb

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhelson.g@gmail.com

source /hkfs/home/haicore/aifb/cc7738/anaconda3/etc/profile.d/conda.sh
conda activate ss-1

cd /hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/graph_embed
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


ls -ltr

# python3 line_tag.py --cfg core/yamls/cora/embedding/line.yaml 
# python3 line_tag.py --cfg core/yamls/pubmed/embedding/line.yaml 
python3 line_tag.py --cfg core/yamls/arxiv_2023/embedding/line.yaml 
