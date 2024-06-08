#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=152
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --job-name=NS_Cora
#SBATCH --mem=501600mb

#SBATCH --output=log/NS_Cora_Benchmark_%j.output
#SBATCH --error=error/NS_Cora_Benchmark_%j.error


#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_gerrman/core/gcns_gerrman

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhelson.g@gmail.com

# Request GPU resources
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate TAPE
cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_gerrman/core/gcns_gerrman
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12

device_list=(0 1 2 3)
data=cora  #pubmed arxiv_2023
yaml=core/yamls/cora/gcns/ns_gnn_models.yaml
model_list=(GAT GAE VGAE GraphSage)

# for index in "${!model_list[@]}"; do
#     for device in {0..3}; do
#         model=${model_list[$device]}
#         echo "python3 final_ns_tune.py --cfg $yaml --device cuda:$device --data $data --model $model --epochs 200 "
#         python3 final_ns_tune.py --cfg $yaml --device cuda:$device --data $data --model $model --epochs 200 --wandb True &

#         while [ "$(jobs -r | wc -l)" -ge 4 ]; do
#             sleep 1
#         done
#     done
#     echo "round $index"
# done

# echo "Press Ctrl+C to stop all jobs."
# read -r -d '' _ < /dev/tty

# # If Ctrl+C is pressed, you might want to kill all background jobs:
# echo "Stopping all background jobs..."
# kill $(jobs -p)

python3 final_ns_tune.py --cfg $yaml --data $data --device cuda:1 --epochs 100 --model GAE --repeat 1 --mark_done