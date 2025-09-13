#!/bin/bash
#SBATCH --job-name=GAN
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=5-23:59
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"
#SBATCH --nodes=1

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu9/9.3

source activate gan

python train.py \
        --outdir=./runs \
        --data=./datasets/o_bc_left_4x_768_360_median_Ultra_Co11_rgb_256x256.zip \
        --gpus=4 \
        --batch=256 \
        --preset=FFHQ-256 \
        --tick=1 \
        --snap=500 \
	--workers=8 \
	--kimg=10000000
