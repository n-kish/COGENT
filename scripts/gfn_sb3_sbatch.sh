#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=ppo_train
#SBATCH --partition=cpu
#SBATCH --mem=10G
#SBATCH --error="/home/knagiredla/robonet/output/ppo.err"
#SBATCH --time=10:00:00
#SBATCH --output=__OUTPUT_PATH__


source ~/.bashrc
conda activate gfn_sb3_gpu
$@
