#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_inference
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=152
#SBATCH --time=20:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/hgf_pdv3669-H2/logs/inference.txt

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=1

group_workspace=/hkfs/work/workspace/scratch/hgf_pdv3669-H2

source ${group_workspace}/miniconda3/bin/activate
conda activate aihero
python ${group_workspace}/-AI-HERO-2-Health/inference.py