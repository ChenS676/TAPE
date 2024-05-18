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
python core/gcns/wb_tune.py --cfg core/yamls/cora/gcns/gae.yaml --sweep core/yamls/cora/gcns/gae_sp1.yaml 