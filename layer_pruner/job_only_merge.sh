#!/bin/bash

# shellcheck disable=SC2206
## SBATCH -p compute
#SBATCH -p cs
#SBATCH -A condo_cs_ross
##SBATCH -p nvidia
#SBATCH --gres=gpu:1
## SBATCH --constraint=80g
#SBATCH --job-name=out_model
#SBATCH --output=out_model.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --tasks-per-node=1
#SBATCH --time=1-00:00:00

python merge_model.py