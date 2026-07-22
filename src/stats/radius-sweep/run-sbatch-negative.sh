#!/bin/bash
#SBATCH --partition=gpu-bio           # retrains the smoothing classifier (small MLP, but on the
                                       # full CryptoBench train_val/val embeddings) 2x
#SBATCH --time=03:00:00               # walltime; 2 retrains (10 A, 15 A), each with early stopping (patience=10)
#SBATCH --nodes=1
#SBATCH --mem=64000
#SBATCH --job-name="negative-radius-sweep"
#SBATCH --output=/home/skrhakv/Projects/seq2pocket/src/stats/radius-sweep/logs/negative-%j.log
#SBATCH --mail-user=vit.skrhak@matfyz.cuni.cz
#SBATCH --mail-type=END,FAIL

cd /home/skrhakv/Projects/seq2pocket/src/stats/radius-sweep
source activate base
conda activate cryptic-nn

python3 negative_radius_sweep.py
