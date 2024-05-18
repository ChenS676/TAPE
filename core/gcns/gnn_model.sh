# #!/bin/bash
# #SBATCH --time=24:00:00
# #SBATCH --nodes=24
# #SBATCH --ntasks=152
# #SBATCH --partition=accelerated
# #SBATCH --job-name=gnn_wb
# #SBATCH --mem=501600mb
# #SBATCH --output=log/TAG_Benchmark_%j.output
# #SBATCH --error=error/TAG_Benchmark_%j.error
# #SBATCH --gres=gpu:4

# #SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/batch

# # Notification settings:
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=cc7738@kit.edu
# source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

# conda activate base
# conda activate EAsF
# # <<< conda initialize <<<
# module purge
# module load devel/cmake/3.18
# module load devel/cuda/11.8
# module load compiler/gnu/12


cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/gcns

# Define the list of items for the loop
items=(ogbn-arxiv ogbn-product pubmed ogbn-products)

# Run the loop in parallel
for item in ogbn-arxiv ogbn-products pubmed ogbn-products; do
    echo "Processing $item"
    python custom_tune.py --data "$item" > "$item-VAE-custom_tune_output.txt" 
done 


python custom_tune.py --data pubmed --device cuda:0  > pubmed-VAE-custom_tune_output.txt

python custom_tune.py --data cora --device cuda:0 > cora-VAE-tune2_output.txt