#!/bin/bash
#SBATCH --job-name=GAN
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --time=10-23:59
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
        --gpus=8 \
        --batch=512 \
        --preset=FFHQ-256 \
        --tick=1 \
        --snap=2000 \
	--workers=16 \
	--kimg=10000000
