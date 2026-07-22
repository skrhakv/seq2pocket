import pickle
import sys

import numpy as np
import torch
from sklearn.metrics import auc, matthews_corrcoef, precision_recall_curve

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'
CRYPTIC_NN_DATA = '/work/skrhakv/cryptic-nn'
sys.path.append(f'{PROJECT_DIRECTORY}/src/utils')
sys.path.append(f'{PROJECT_DIRECTORY}/src/stats/table3-repro')

from eval_utils import CryptoBenchClassifier  # noqa: E402
import __main__
setattr(__main__, "CryptoBenchClassifier", CryptoBenchClassifier)

import table3_core  # noqa: E402

DEVICE = 'cpu'
DECISION_THRESHOLD = 0.7

CACHE_PATH = f'{PROJECT_DIRECTORY}/src/stats/clustering-hyperparam-sweep/val_predictions_cache.pkl'
EMBEDDINGS_DIR = f'{CRYPTIC_NN_DATA}/embeddings'
DISTANCE_MATRICES_DIR = f'{CRYPTIC_NN_DATA}/distance-matrices'
SMOOTHING_MODEL_PATH = f'{PROJECT_DIRECTORY}/data/models/smoother.pt'

RADIUS_GRID = [10, 15]


def smoothing_scores(predictions, probabilities, X_test, distance_matrix, smoothing_model):
    scores = probabilities.copy()
    for residue_idx in np.where(predictions == 0.0)[0]:
        current_residue_embedding = X_test[residue_idx]
        close_residues_indices = np.where(
            distance_matrix[residue_idx] < table3_core.POSITIVE_DISTANCE_THRESHOLD
        )[0]
        close_binding_residues_indices = np.intersect1d(close_residues_indices, np.where(predictions == 1.0)[0])
        if len(close_binding_residues_indices) == 0:
            continue
        elif len(close_binding_residues_indices) == 1:
            surrounding_embedding = X_test[close_binding_residues_indices].reshape(-1)
        else:
            surrounding_embedding = np.mean(X_test[close_binding_residues_indices], axis=0).reshape(-1)

        concatenated_embedding = torch.tensor(
            np.concatenate((current_residue_embedding, surrounding_embedding), axis=0), dtype=torch.float32
        ).to(DEVICE)
        smoothing_logit = smoothing_model(concatenated_embedding).squeeze()
        scores[residue_idx] = torch.sigmoid(smoothing_logit).item()
    return scores


def evaluate_radius(cache, smoothing_model, radius):
    table3_core.POSITIVE_DISTANCE_THRESHOLD = radius

    y_true_parts, y_pred_parts, y_score_parts = [], [], []
    for entry in cache.values():
        pdb_id, chain_id = entry['pdb_id'], entry['chain_id']
        probabilities = entry['probabilities']

        try:
            X_test = np.load(f'{EMBEDDINGS_DIR}/{pdb_id}{chain_id}.npy')
            distance_matrix = np.load(f'{DISTANCE_MATRICES_DIR}/{pdb_id}{chain_id}.npy')
        except FileNotFoundError:
            continue
        if X_test.shape[0] != probabilities.shape[0] or distance_matrix.shape[0] != probabilities.shape[0]:
            continue

        true_binding = np.zeros(probabilities.shape[0])
        for pocket in entry['pockets']:
            true_binding[pocket] = 1

        predictions_without = (probabilities > DECISION_THRESHOLD).astype(float)
        predictions_with = table3_core.apply_smoothing(predictions_without, X_test, distance_matrix, smoothing_model)
        scores_with = smoothing_scores(predictions_without, probabilities, X_test, distance_matrix, smoothing_model)

        y_true_parts.append(true_binding)
        y_pred_parts.append(predictions_with)
        y_score_parts.append(scores_with)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    y_score = np.concatenate(y_score_parts)

    mcc = matthews_corrcoef(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)
    return mcc, auprc


def main():
    torch.manual_seed(420)
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    print(f'{len(cache)} cached proteins')

    smoothing_model = torch.load(SMOOTHING_MODEL_PATH, weights_only=False, map_location=DEVICE).to(DEVICE)
    smoothing_model.eval()

    print(f'\n{"candidate_radius":<18}{"AUPRC":<10}{"MCC":<10}')
    for radius in RADIUS_GRID:
        mcc, auprc = evaluate_radius(cache, smoothing_model, radius)
        print(f'{radius:<18}{mcc:<10.4f}{auprc:<10.4f}')


if __name__ == '__main__':
    main()
