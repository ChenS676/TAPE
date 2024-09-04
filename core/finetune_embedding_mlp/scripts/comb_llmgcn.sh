#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --partition=accelerated
#SBATCH --job-name=gnn_wb
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=50160mb
#BATCH  --cpu-per-gpu=38
#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --gres=gpu:1  # Ensure you are allowed to use these many GPUs, otherwise reduce the number here
#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-iclr25/cc7738-benchmark_tag/TAPE_gerrman/TAPE_emb/core/finetune_embedding_mlp/res_comb_res
module purge/res_comb_res

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhelson.g@gmail.com

# Request GPU resources
source /hkfs/home/project/hk-project-test-p0022257/cc7738/anaconda3/etc/profile.d/conda.sh

# cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_gerrman/TAPE_emb/core/finetune_embedding_mlp

cd /hkfs/work/workspace/scratch/cc7738-iclr25/cc7738-benchmark_tag/TAPE_gerrman/TAPE_emb/core/finetune_embedding_mlp/
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12

# export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

# WANDB_DISABLED=True gdb --args python3 comb_lm_trainer.py --cfg core/yamls/cora/comb/gcn_encoder.yaml
WANDB_DISABLED=True python3 comb_lm_trainer.py --cfg core/yamls/cora/comb/gcn_encoder.yaml --repeat 3
