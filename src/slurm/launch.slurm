#!/bin/bash
#SBATCH --job-name=bash
#SBATCH -o output=angio_%j.out
#SBATCH --partition=xgpu
#SBATCH --nodelist=gnode01
#SBATCH --ntasks-per-node=4 
#SBATCH --gres=gpu:1       
#SBATCH --mem=44000          
#SBATCH --job-name="pinns"

DEVICE=${1:-cuda}

cd ..
module load cuda
python main.py --device "$DEVICE"
