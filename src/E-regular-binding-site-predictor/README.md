# Regular binding sites
Let's replicate the whole process, but on regular binding sites. Specifically, let's finetune the ESM2-3B model on the scPDB dataset and evaluate it on LIGYSIS. Let's see if P2Rank and finetuned models are, again, complementary, or not.

## Structure
1. `extraction`: extract the scPDB dataset, LIGYSIS dataset and P2Rank predictions on LIGYSIS.
2. `train`: finetune the ESM2 model
3. `evaluation`: evaluate the scPDB-trained ESM2 model on LIGYSIS. Also repeat the evaluation of P2Rank on LIGYSIS.
4. `ligysis-similarity`: some proteins have the same binding site - e.g. zinc finger (for example: `2EM2`). Let's recalculate the similarity.