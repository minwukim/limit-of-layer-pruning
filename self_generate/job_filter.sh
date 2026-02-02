#!/bin/bash
#SBATCH -p compute
#SBATCH --job-name=process_openthoughts
#SBATCH --output=/scratch/ss13750/nnsight/openthoughts/process_filter.out
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4

source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate merger

python -u /scratch/ss13750/nnsight/openthoughts/download_and_filter.py --num_samples 100000 --max_length 8000
