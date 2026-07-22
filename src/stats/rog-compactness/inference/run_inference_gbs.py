#!/usr/bin/env python3
"""
Output: {PROJECT_DIRECTORY}/data/stats/hole-metrics/raw-gbs.pkl
    {protein_id: {'y_test': (L,) int8, 'pred_without': (L,) int8, 'pred_with': (L,) int8,
                  'probabilities': (L,) float32}}
"""
import sys
import csv
import pickle
import numpy as np
import torch

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'
sys.path.append(f'{PROJECT_DIRECTORY}/src/utils')
sys.path.append(f'{PROJECT_DIRECTORY}/../cryptic-nn/src')

import eval_utils
from eval_utils import CryptoBenchClassifier
import finetuning_utils
from transformers import AutoTokenizer

torch.manual_seed(420)

DATA_DIRECTORY = f'{PROJECT_DIRECTORY}/data'
MODEL_PATH = f'{DATA_DIRECTORY}/models/gbs-model-enhanced-scPDB-filtered.pt'
ESM_MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DECISION_THRESHOLD = 0.7

ESM_EMBEDDINGS_PATH = f'{DATA_DIRECTORY}/embeddings/ligysis'
COORDINATES_DIR = f'{DATA_DIRECTORY}/coordinates/ligysis'
POSITIVE_DISTANCE_THRESHOLD = 15  # smoothing candidate radius, same as manuscript / notebook
RESIDUE_LEVEL_ANNOTATIONS = f'{DATA_DIRECTORY}/data-extraction/ligysis_for_residue_level_evaluation.csv'

SMOOTHING_MODEL_PATH = f'{DATA_DIRECTORY}/models/smoother.pt'
SMOOTHING_DECISION_THRESHOLD = 0.4

OUTPUT_PATH = f'{DATA_DIRECTORY}/stats/hole-metrics/raw-gbs.pkl'


def main():
    import __main__
    setattr(__main__, "CryptoBenchClassifier", CryptoBenchClassifier)
    print(f'Device: {DEVICE}')

    loaded_model = torch.load(MODEL_PATH, weights_only=False).to(DEVICE)
    loaded_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)

    smoothing_model = torch.load(SMOOTHING_MODEL_PATH, weights_only=False).to(DEVICE)
    smoothing_model.eval()

    # regenerate the residue-level annotation file exactly as the notebook does
    # (pools all pockets of a protein into a single row)
    sequences = {}
    annotations = {}
    with open(f'{DATA_DIRECTORY}/data-extraction/ligysis_for_pocket_level_evaluation.csv', 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            sequences[row[0]] = row[4]
            if row[0] not in annotations:
                annotations[row[0]] = row[3]
            else:
                annotations[row[0]] += ' ' + row[3]
    with open(RESIDUE_LEVEL_ANNOTATIONS, 'w') as f:
        for protein_id in sequences:
            f.write(f"{protein_id[:4]};{protein_id[4:]};UNKNOWN;{annotations[protein_id]};{sequences[protein_id]}\n")

    val_dataset = finetuning_utils.process_sequence_dataset(
        RESIDUE_LEVEL_ANNOTATIONS,
        tokenizer,
        load_ids=True,
    )

    results = {}
    skipped = []

    with torch.no_grad():
        for i, batch in enumerate(val_dataset):
            protein_id = batch['ids'][0]
            del batch['ids']
            batch = finetuning_utils.collate_fn([batch], tokenizer=tokenizer)
            output1 = loaded_model(batch)

            labels = batch['labels'].to(DEVICE)
            flattened_labels = labels.flatten()
            mask = flattened_labels != -100
            y_test = flattened_labels[mask].cpu().numpy()
            logits = output1.flatten()[mask]

            probabilities = torch.sigmoid(logits).cpu().numpy()
            pred_without = (probabilities > DECISION_THRESHOLD).astype(np.float32)
            pred_with = pred_without.copy()

            coordinates_path = f'{COORDINATES_DIR}/{protein_id}.npy'
            embedding_path = f'{ESM_EMBEDDINGS_PATH}/{protein_id}.npy'
            try:
                coordinates = np.load(coordinates_path)
                distance_matrix = eval_utils.compute_distance_matrix(coordinates)
            except FileNotFoundError:
                print(f'[{i+1}/{len(val_dataset)}] {protein_id}: SKIP (no coordinates)')
                skipped.append(f'{protein_id}: no coordinates')
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
