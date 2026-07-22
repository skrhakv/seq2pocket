#!/bin/bash
#SBATCH --partition=gpu-bio           # CPU-only job (tiny MLP forward passes only), reusing this
                                       # partition per repo convention
#SBATCH --time=02:00:00               # walltime; cheap CPU sweep over 5 radii x 195 proteins
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --job-name="candidate-radius-sweep"
#SBATCH --output=/home/skrhakv/Projects/seq2pocket/src/stats/radius-sweep/logs/candidate-%j.log
#SBATCH --mail-user=vit.skrhak@matfyz.cuni.cz
#SBATCH --mail-type=END,FAIL

# Requires val_predictions_cache.pkl, produced by
# ../clustering-hyperparam-sweep/run-sbatch-cache.sh (already run).

cd /home/skrhakv/Projects/seq2pocket/src/stats/radius-sweep
source activate base
conda activate cryptic-nn

python3 candidate_radius_sweep.py
