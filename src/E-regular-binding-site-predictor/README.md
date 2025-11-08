# Regular binding sites
Let's replicate the whole process, but on regular binding sites. Specifically, let's finetune the ESM2-3B model on the scPDB dataset and evaluate it on LIGYSIS. Let's see if P2Rank and finetuned models are, again, complementary, or not.

## Structure
1. `extraction`: extract the scPDB dataset, LIGYSIS dataset and P2Rank predictions on LIGYSIS.
2. `train`: finetune the ESM2 model
3. `pocket-level-evaluation.ipynb`: evaluate the finetuned ESM2 model.