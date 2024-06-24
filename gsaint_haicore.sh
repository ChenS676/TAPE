#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --job-name=gnn_wb

#SBATCH --nodes=1
#SBATCH --mem=501600mb
#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --gres=gpu:full:4  # Ensure you are allowed to use these many GPUs, otherwise reduce the number here
#SBATCH --chdir=/hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/res_outputs

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhelson.g@gmail.com

# Request GPU resources
source /hkfs/home/haicore/aifb/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate ss
cd /hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/data_utils
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) rwcpp.cpp -o rwcpp$(python3-config --extension-suffix)

cd /hkfs/work/workspace_haic/scratch/cc7738-TAGBench/TAPE_gerrman/TAPE/core/gcns
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12

device_list=(0 1)  # Adjust according to your SLURM --gres=gpu configuration
data=cora  # Can be 'cora', 'synthetic', 'pubmed', or 'arxiv_2023'
yaml=core/yamls/cora/gcns/ns_gnn_models.yaml
model_list=(GAT GraphSage)

for index in "${!model_list[@]}"; do
    device=${device_list[$index % ${#device_list[@]}]}  # This ensures each model gets a unique device from the list

    model=${model_list[$index]}
    echo "python3 gsaint_main.py --cfg $yaml --device cuda:$device --data $data --model $model --epochs 1000 --wandb True"
    python3 gsaint_main.py --cfg $yaml --device cuda:$device --data $data --model $model --epochs 1000 --wandb True &

    # Wait if the maximum number of jobs (equal to number of GPUs) is reached
    while [ "$(jobs -r | wc -l)" -ge ${#device_list[@]} ]; do
        sleep 1
    done
    echo "round $index"
done

echo "Press Ctrl+C to stop all jobs."
read -r -d '' _ < /dev/tty

# If Ctrl+C is pressed, you might want to kill all background jobs:
echo "Stopping all background jobs..."
kill $(jobs -p)


# For ploting result in tensorboard: tensorboard --logdir runs/Jun23_16-13-_haicn1992.localdomain/