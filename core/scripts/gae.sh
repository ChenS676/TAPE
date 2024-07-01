#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --partition=accelerated
#SBATCH --job-name=tag_struc2vec
#SBATCH --mem-per-cpu=1600mb

#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error


#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE/scripts

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate nui
cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE/core/gcns
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


# python custom_main.py --cfg core/yamls/cora/gcns/gae.yaml --sweep core/yamls/cora/gcns/gae_sp1.yaml
# python core/gcns/wb_tune.py --cfg core/yamls/cora/gcns/gae.yaml --sweep core/yamls/cora/gcns/gae_sp1.yaml 
python universal_tune_heart.py --device cuda:0 --data pubmed --model GraphSage --epochs 1000 > graphsage-pubmed-0-universal_tune_heart_output.txt
python universal_tune_heart.py --device cuda:1 --data arxiv_2023 --model GraphSage --epochs 1000 > graphsage-arxiv_2023-1-universal_tune_heart_output.txt
python universal_tune_heart.py --device cuda:2 --data arxiv_2023 --model GAT --epochs 1000 > gat-arxiv_2023-1-universal_tune_heart_output.txt
python universal_tune_heart.py --device cuda:3 --data arxiv_2023 --model GAE --epochs 1000 > gae-arxiv_2023-1-universal_tune_heart_output.txt
python universal_tune_heart.py --device cuda:3 --data arxiv_2023 --model VGAE --epochs 1000 > vgae-arxiv_2023-1-universal_tune_heart_output.txt


echo "Running on data $data with device cuda:$device and model $model"
python universal_tune_heart.py --device cuda:0 --data cora --model GAT --epochs 100 > gat-cora-0-universal_tune_heart_output.txt
python universal_tune.py --device cuda:0 --data cora --model GAT --epochs 100 > gat-cora-0-universal_tune_heart_output.txt