#!/bin/bash
#SBATCH --partition=gpu-bio           # partition you want to run job in
#SBATCH --time=48:00:00           # walltime for the job in format (days-)hours:minutes:seconds
#SBATCH --nodes=1                 # number of nodes (can be only 1)
#SBATCH --mem=512000               # memory resource per node
#SBATCH --job-name="table3-repro"     # change to your job name
#SBATCH --output=/home/skrhakv/Projects/seq2pocket/src/stats/table3-repro/logs/run-%j.log
#SBATCH --mail-user=vit.skrhak@matfyz.cuni.cz
#SBATCH --mail-type=END,FAIL
#SBATCH --exclusive               # Use whole node -- ESM-2 3B inference, same as the original evaluation runs

cd /home/skrhakv/Projects/seq2pocket/src/stats/table3-repro
source activate base
conda activate cryptic-nn

echo "=== GBS (LIGYSIS) Table 3 reproduction ==="
python3 run_gbs.py

echo "=== CBS (CryptoBench) Table 3 reproduction ==="
python3 run_cbs.py
