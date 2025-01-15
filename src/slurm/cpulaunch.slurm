#!/bin/bash
#SBATCH --job-name=bash
#SBATCH -o output=angio_%j.out
#SBATCH --partition=xhicpu
#SBATCH --nodelist=wnode04
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:0
#SBATCH --mem=44000
#SBATCH --job-name="pinns"

DEVICE=${1:-cuda}

cd ..
module load cuda
python main.py --device "$DEVICE"
