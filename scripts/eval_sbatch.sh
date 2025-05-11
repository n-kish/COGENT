#!/bin/bash
#SBATCH --job-name=PPO_eval
#SBATCH --partition=cpu
#SBATCH --mem=15G
#SBATCH --error=/home/knagiredla/robonet/sensitivity_results/error.err
#SBATCH --time=04:00:00
#SBATCH --output=/home/knagiredla/robonet/sensitivity_results/tmp/%j.out
#SBATCH --qos=batch-short
$@