#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --job-name=gnn_wb

#SBATCH --nodes=1

#SBATCH --mem=501600mb

#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --gres=gpu:full:4

#SBATCH --chdir=/hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/res_outputs

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhelson.g@gmail.com

# Request GPU resources
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate ss
cd /hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/data_utils
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) rwcpp.cpp -o rwcpp$(python3-config --extension-suffix)
cd /hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/gcns
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12

device_list=(4 5 6 7)
data=cora  #cora synthetic pubmed arxiv_2023
yaml=core/yamls/cora/gcns/ns_gnn_models.yaml
model_list=(GAT GAE VGAE GraphSage)

# for index in "${!model_list[@]}"; do
#     for device in {0..3}; do
#         model=${model_list[$device]}
#         # cuda:$device
#         echo "python3 final_ns_tune.py --cfg $yaml --device cpu:$device --data $data --model $model --epochs 20 --wandb True"
#         python3 final_ns_tune.py --cfg $yaml --device cpu:$device --data $data --model $model --epochs 20 --wandb True &

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

python3 gsaint_tune.py --cfg $yaml --data $data --epochs 6 --model GAE --device cpu:0 --repeat 1
# python3 final_ns_tune.py --cfg $yaml --data $data --device cuda:0 --epochs 20 --model GAE --repeat 1 --mark_done
# python3 final_ns_tune.py --cfg $yaml --data $data --epochs 20 --model GAE --device cuda:0 
# python3 test_ns.py --cfg core/yamls/pubmed/gcns/ns_gnn_models.yaml --data pubmed --epochs 100 --model GAE --device cuda:0