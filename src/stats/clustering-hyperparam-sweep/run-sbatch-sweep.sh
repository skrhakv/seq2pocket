#!/bin/bash
#SBATCH --partition=gpu-bio           # CPU-only job -- no GPU needed, but reusing this partition:
                                       # clustering_utils.execute_atom_clustering hardcodes
                                       # MeanShift(n_jobs=-1), which hangs/spams ChildProcessError
                                       # on the login node's sandboxed shell; runs fine on a
                                       # compute node (see rog-compactness/inference/run-sbatch.sh).
#SBATCH --time=06:00:00               # walltime; 12 grid points x 194 proteins, generous margin
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4             # CPU-only work (numpy/scipy/biopython SASA + clustering)
#SBATCH --mem=16000
#SBATCH --job-name="clustering-sweep"
#SBATCH --output=/home/skrhakv/Projects/seq2pocket/src/stats/clustering-hyperparam-sweep/logs/sweep-%j.log
#SBATCH --mail-user=vit.skrhak@matfyz.cuni.cz
#SBATCH --mail-type=END,FAIL

# Requires val_predictions_cache.pkl, produced by run-sbatch-cache.sh -- submit
# this only after that job has finished (or chain it with
# `sbatch --dependency=afterok:<cache_job_id> run-sbatch-sweep.sh`).

cd /home/skrhakv/Projects/seq2pocket/src/stats/clustering-hyperparam-sweep
source activate base
conda activate cryptic-nn

python3 sweep.py
