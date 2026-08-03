#!/bin/bash
#SBATCH --partition=gpu-bio           # partition you want to run job in
#SBATCH --time=24:00:00           # walltime for the job in format (days-)hours:minutes:seconds
#SBATCH --nodes=1                 # number of nodes (can be only 1)
#SBATCH --mem=128000               # memory resource per node
#SBATCH --job-name="raw-predictions-inference"     # change to your job name
#SBATCH --output=/home/skrhakv/Projects/seq2pocket/src/stats/hull-compactness/inference/logs/inference-%j.log       # stdout and stderr output file
#SBATCH --mail-user=vit.skrhak@matfyz.cuni.cz
#SBATCH --mail-type=END,FAIL

source activate base
conda activate cryptic-nn

cd /home/skrhakv/Projects/seq2pocket/src/stats/hull-compactness/inference

echo "=== GBS (LIGYSIS) inference ==="
python3 run_inference_gbs.py

echo "=== CBS (CryptoBench) inference ==="
python3 run_inference_cbs.py

# Produces raw-{gbs,cbs}.pkl (data/stats/hole-metrics/), consumed by
# ../hull_compactness.py for the fill-rate analysis.
