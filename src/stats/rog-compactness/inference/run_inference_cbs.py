#!/usr/bin/env python3
"""
GPU inference step: run the seq2pocket CBS model on the CryptoBench test set
and save per-residue predictions (with and without smoothing) for downstream
CPU-only analysis (see ../rog_compactness.py and ../rog_random_control.py).

Mirrors src/evaluation/cryptobench/residue-level-evaluation.ipynb exactly
(same paths, same DECISION_THRESHOLD / SMOOTHING_DECISION_THRESHOLD /
POSITIVE_DISTANCE_THRESHOLD, same smoothing loop, same 3-tuple model output
unpacking) -- the only addition is that per-protein ground truth /
predictions (with and without smoothing) are saved to disk instead of only
being reduced to aggregate MCC/F1/ACC. The raw per-residue probabilities
(pre-threshold) are also saved -- needed by rog_compactness.py's clustering
step (clustering_utils.execute_atom_clustering requires a probabilities
argument for its cluster scoring), even though the notebook itself never
persists them.

One deliberate deviation, discussed with the user: this script calls
loaded_model.eval() on the main pLM, which the notebook never does. The
checked-in checkpoint loads with .training=True (dropout active in the
internal ESM encoder), so the notebook as literally written runs inference
non-deterministically. The user chose to keep .eval() here (deterministic,
reproducible results) rather than replicate that latent bug -- unlike the
smoothing classifier (smoother.pt), where the notebook's dropout-active
behavior *is* replicated deliberately (see table3-repro/table3_core.py).

The compactness analysis itself is *not* done here: it is cheap and
CPU-only, so it lives alongside this file in rog-compactness/ and can be
re-run without re-running ESM-2 3B inference.

Output: {PROJECT_DIRECTORY}/data/stats/hole-metrics/raw-cbs.pkl
    {protein_id: {'y_test': (L,) int8, 'pred_without': (L,) int8, 'pred_with': (L,) int8,
                  'probabilities': (L,) float32}}
"""
import sys
import pickle
import numpy as np
import torch

PATH = '/home/skrhakv/Projects/seq2pocket'
sys.path.append(f'{PATH}/src/utils')
sys.path.append(f'{PATH}/../cryptic-nn/src')

import finetuning_utils
from eval_utils import CryptoBenchClassifier
from transformers import AutoTokenizer

torch.manual_seed(420)

MODEL_PATH = f'{PATH}/data/models/cbs-model.pt'
ESM_MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DECISION_THRESHOLD = 0.7
DATASET_PATH = '/work/skrhakv/cryptic-nn/test.txt'

ESM_EMBEDDINGS_PATH = '/work/skrhakv/cryptic-nn/embeddings'
DISTANCE_MATRIX_PATH = '/work/skrhakv/cryptic-nn/distance-matrices'
POSITIVE_DISTANCE_THRESHOLD = 15

SMOOTHING_MODEL_PATH = f'{PATH}/data/models/smoother.pt'
SMOOTHING_DECISION_THRESHOLD = 0.4

OUTPUT_PATH = f'{PATH}/data/stats/hole-metrics/raw-cbs.pkl'


def main():
    import __main__
    setattr(__main__, "CryptoBenchClassifier", CryptoBenchClassifier)

    print(f'Device: {DEVICE}')

    loaded_model = torch.load(MODEL_PATH, weights_only=False).to(DEVICE)
    loaded_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)

    val_dataset = finetuning_utils.process_sequence_dataset(
        DATASET_PATH,
        tokenizer,
        load_ids=True,
    )

    smoothing_model = torch.load(SMOOTHING_MODEL_PATH, weights_only=False).to(DEVICE)
    smoothing_model.eval()

    results = {}
    skipped = []

    with torch.no_grad():
        for i, batch in enumerate(val_dataset):
            protein_id = batch['ids'][0]
            del batch['ids']
            batch = finetuning_utils.collate_fn([batch], tokenizer=tokenizer)
            output1, _, _ = loaded_model(batch)

            labels = batch['labels'].to(DEVICE)
            flattened_labels = labels.flatten()
            mask = flattened_labels != -100
            y_test = flattened_labels[mask].cpu().numpy()
            logits = output1.flatten()[mask]

            probabilities = torch.sigmoid(logits).cpu().numpy()
            pred_without = (probabilities > DECISION_THRESHOLD).astype(np.float32)
            pred_with = pred_without.copy()

            distance_matrix_path = f'{DISTANCE_MATRIX_PATH}/{protein_id}.npy'
            embedding_path = f'{ESM_EMBEDDINGS_PATH}/{protein_id}.npy'
            try:
                distance_matrix = np.load(distance_matrix_path)
            except FileNotFoundError:
                print(f'[{i+1}/{len(val_dataset)}] {protein_id}: SKIP (no distance matrix)')
                skipped.append(f'{protein_id}: no distance matrix')
                continue

            if distance_matrix.shape[0] != len(y_test):
                print(f'[{i+1}/{len(val_dataset)}] {protein_id}: SKIP (distance matrix / label length mismatch)')
                skipped.append(f'{protein_id}: length mismatch')
                continue

            try:
                X_test = np.load(embedding_path)
            except FileNotFoundError:
                print(f'[{i+1}/{len(val_dataset)}] {protein_id}: SKIP (no embeddings)')
                skipped.append(f'{protein_id}: no embeddings')
                continue

            if X_test.shape[0] != distance_matrix.shape[0]:
                print(f'[{i+1}/{len(val_dataset)}] {protein_id}: SKIP (embedding length mismatch)')
                skipped.append(f'{protein_id}: embedding length mismatch')
                continue

            # smoothing loop -- identical logic to residue-level-evaluation.ipynb
            for residue_idx in np.where(pred_without == 0.0)[0]:
                current_residue_embedding = X_test[residue_idx]
                close_residues_indices = np.where(distance_matrix[residue_idx] < POSITIVE_DISTANCE_THRESHOLD)[0]
                close_binding_residues_indices = np.intersect1d(
                    close_residues_indices, np.where(pred_without == 1.0)[0]
                )
                if len(close_binding_residues_indices) == 0:
                    continue
                elif len(close_binding_residues_indices) == 1:
                    surrounding_embedding = X_test[close_binding_residues_indices].reshape(-1)
                else:
                    surrounding_embedding = np.mean(X_test[close_binding_residues_indices], axis=0).reshape(-1)

                concatenated_embedding = torch.tensor(
                    np.concatenate((current_residue_embedding, surrounding_embedding), axis=0),
                    dtype=torch.float32,
                ).to(DEVICE)
                smoothing_logit = smoothing_model(concatenated_embedding).squeeze()
                if (torch.sigmoid(smoothing_logit) > SMOOTHING_DECISION_THRESHOLD).float() == 1:
                    pred_with[residue_idx] = 1.0

            results[protein_id] = {
                'y_test': y_test.astype(np.int8),
                'pred_without': pred_without.astype(np.int8),
                'pred_with': pred_with.astype(np.int8),
                'probabilities': probabilities.astype(np.float32),
            }
            print(f'[{i+1}/{len(val_dataset)}] {protein_id}: OK ({int(y_test.sum())} true binding residues, '
                  f'{int(pred_with.sum() - pred_without.sum())} added by smoothing)')

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(results, f)

    print(f'\nProcessed {len(results)} proteins, skipped {len(skipped)}.')
    if skipped:
        for s in skipped:
            print(f'  - {s}')
    print(f'Saved raw predictions to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
