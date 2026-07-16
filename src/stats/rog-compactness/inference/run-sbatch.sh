#!/bin/bash
#SBATCH --partition=gpu-bio           # partition you want to run job in
#SBATCH --time=24:00:00           # walltime for the job in format (days-)hours:minutes:seconds
#SBATCH --nodes=1                 # number of nodes (can be only 1)
#SBATCH --mem=128000               # memory resource per node
#SBATCH --job-name="rog-compactness"     # change to your job name
#SBATCH --output=/home/skrhakv/Projects/seq2pocket/src/stats/rog-compactness/inference/logs/inference-%j.log       # stdout and stderr output file
#SBATCH --mail-user=vit.skrhak@matfyz.cuni.cz
#SBATCH --mail-type=END,FAIL

source activate base
conda activate cryptic-nn

cd /home/skrhakv/Projects/seq2pocket/src/stats/rog-compactness/inference

echo "=== GBS (LIGYSIS) inference ==="
python3 run_inference_gbs.py

echo "=== CBS (CryptoBench) inference ==="
python3 run_inference_cbs.py

cd /home/skrhakv/Projects/seq2pocket/src/stats/rog-compactness

# CPU-only from here, but clustering_utils.execute_atom_clustering hardcodes
# MeanShift(n_jobs=-1), which hangs/spams ChildProcessError on the login
# node's sandboxed shell -- runs fine on an actual compute node, hence
# bundled into this same job rather than run interactively.

echo "=== RoG compactness: WITH vs WITHOUT smoothing (GBS) ==="
python3 rog_compactness.py --task gbs

echo "=== RoG compactness: WITH vs WITHOUT smoothing (CBS) ==="
python3 rog_compactness.py --task cbs
