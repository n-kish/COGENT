#!/bin/bash
#SBATCH --qos=batch-long
#SBATCH --job-name=gfn_full
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --error="/home/knagiredla/robonet/sensitivity_output/gfn_simtest.err"
#SBATCH --time=240:00:00
#SBATCH --output=/home/knagiredla/robonet/sensitivity_output/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=knagiredla@deakin.edu.au


source ~/.bashrc
conda activate gfn_sb3_gpu
$@
