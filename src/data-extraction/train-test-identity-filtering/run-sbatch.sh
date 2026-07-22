#!/bin/bash
#SBATCH --partition=gpu-bio           # CPU-only job, reusing this partition per repo convention
#SBATCH --time=04:00:00               # walltime; ~27M sequence pairs, checkpointed so safe to requeue if it runs out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32000
#SBATCH --job-name="train-test-identity-filtering"
#SBATCH --output=/home/skrhakv/Projects/seq2pocket/src/data-extraction/train-test-identity-filtering/logs/filter-%j.log
#SBATCH --mail-user=vit.skrhak@matfyz.cuni.cz
#SBATCH --mail-type=END,FAIL

cd /home/skrhakv/Projects/seq2pocket/src/data-extraction/train-test-identity-filtering
source activate base
conda activate seq2pocket

python3 filter_train_by_identity.py --max_workers 32
