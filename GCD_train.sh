#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xw3g19@soton.ac.uk
#SBATCH --time=48:0:00

module load conda/py3-latest
conda activate GCD-py37

python train.py --dump_path /ECShome/xw3g19/Reproduce_GCD_LLM/GCD/results --exp_name LCM --n_enc_layers 2 --n_dec_layers 2 --batch_size 512 --base 10 --maxint 1000000 --env_base_seed 42
