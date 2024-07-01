#!/bin/sh
#SBATCH --time=4:00:00
#SBATCH --nodes=12
#SBATCH --ntasks=152
#SBATCH --partition=dev_cpuonly
#SBATCH --job-name=citation2_data_dist
#SBATCH --mem=243200mb
#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/batch

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu

source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate EAsF
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12



cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/ipynbs


#'ogbl-ppa' 'ogbl-collab' 'ogbl-ddi' 'ogbl-vessel'; do
python data_dist.py --data ogbl-collab
