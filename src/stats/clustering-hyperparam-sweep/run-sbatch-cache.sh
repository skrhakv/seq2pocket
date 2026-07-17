#!/bin/bash
#SBATCH --partition=gpu-bio           # ESM-2 3B inference, same as the original evaluation runs
#SBATCH --time=04:00:00           # walltime; only 194 proteins (CryptoBench val split), well under
                                   # the 48h budgeted for the full ~2900-protein Table 3 repro runs
#SBATCH --nodes=1                 # number of nodes (can be only 1)
#SBATCH --mem=512000               # memory resource per node
#SBATCH --job-name="clustering-sweep-cache"
#SBATCH --output=/home/skrhakv/Projects/seq2pocket/src/stats/clustering-hyperparam-sweep/logs/cache-%j.log
#SBATCH --mail-user=vit.skrhak@matfyz.cuni.cz
#SBATCH --mail-type=END,FAIL

cd /home/skrhakv/Projects/seq2pocket/src/stats/clustering-hyperparam-sweep
source activate base
conda activate cryptic-nn

python3 cache_predictions.py
