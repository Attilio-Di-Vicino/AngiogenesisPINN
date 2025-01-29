#!/bin/bash
#SBATCH -A IscrC_LASAN24_0
#SBATCH --output=angio_%j.out
#SBATCH --error=angio_%j.err
#SBATCH -p boost_usr_prod
#SBATCH --time 00:20:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=pinns_angio_gpu

module load cuda/12.1 
source ~/pinn/bin/activate
python3 ../main.py --device cuda
