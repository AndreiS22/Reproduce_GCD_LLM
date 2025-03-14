#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as1g21@soton.ac.uk
#SBATCH --time=24:0:00

module load conda/py3-latest
conda activate GCD-env

python train.py --dump_path /home/as1g21/GCD/results --exp_name DEMO --base 10 --maxint 1000000 --env_base_seed 42
