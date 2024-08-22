#!/bin/sh


#SBATCH --time=2-00:00:00
#SBATCH --partition=cpuonly
#SBATCH --job-name=w2v-mlp-arxiv2023
#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/batch

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu

source /hkfs/home/project/hk-project-test-p0022257/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate TAG-LP
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12



cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/embedding_mlp

data="arxiv_2023"
max_iter=2000
embedders=("w2v")
decoders=("dot" "concat" "euclidean")
device="cpu"

for embedder in "${embedders[@]}"; do
    for i in ${!decoders[@]}; do
        decoder=${decoders[$i]}
        # device=${devices[$i]}
        echo "python lp_node_embed.py --data $data --embedder $embedder --device $device --decoder $decoder --epoch $max_iter"
        python lp_node_embed.py --data $data --embedder $embedder --device $device --decoder $decoder --epoch $max_iter --report_step 200 &
    done
    wait 
done

# for i in ${!decoders[@]}; do
#     decoder=${decoders[$i]}
#     device=${devices[$i]}
#     echo "python lp_edge_embed.py --data $data --embedder $embedder --device $device "
#     python lp_edge_embed.py --data $data --embedder $embedder --device $device --epochs $max_iter &
#     wait 
# done
