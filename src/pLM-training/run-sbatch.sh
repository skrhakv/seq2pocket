#!/bin/bash
#SBATCH --partition=gpu-bio           # partition you want to run job in
#SBATCH --time=48:00:00           # walltime for the job in format (days-)hours:minutes:seconds
#SBATCH --nodes=1                 # number of nodes (can be only 1)
#SBATCH --mem=512000               # memory resource per node
#SBATCH --job-name="esm-3b"     # change to your job name
#SBATCH --output=/home/skrhakv/Projects/seq2pocket/src/pLM-training/output-650M.txt       # stdout and stderr output file
#SBATCH --mail-user=vit.skrhak@matfyz.cuni.cz # send email when job changes state to email address user@example.com
#SBATCH --exclusive               # Use whole node

cd /home/skrhakv/Projects/seq2pocket/src/pLM-training
source activate base
conda activate cryptic-nn
python3 train.py