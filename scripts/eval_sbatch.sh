#!/bin/bash
#SBATCH --job-name=PPO_eval
#SBATCH --partition=cpu
#SBATCH --mem=20G
#SBATCH --error=/home/knagiredla/robonet/output/error.err
#SBATCH --time=10:00:00
#SBATCH --output=/home/knagiredla/robonet/output/%j.out
#SBATCH --qos=batch-short
$@