#!/bin/bash
#SBATCH --job-name=bash
#SBATCH -o output=angio_%j.out
#SBATCH --partition=xgpu
#SBATCH --nodelist=gnode04
#SBATCH --ntasks-per-node=4 
#SBATCH --gres=gpu:1       
#SBATCH --mem=44000          
#SBATCH --job-name="pinns"

module load cuda/12.1 
source activate ~/pinn/bin/activate
python main.py