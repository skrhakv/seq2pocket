"""
Runs the CBS finetuned pLM (cbs-model.pt) + the main smoothing classifier
(smoother.pt, 3B embeddings) once per protein in the CryptoBench validation
split (val.txt, 194 proteins).
"""
import csv
import pickle
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'
CRYPTIC_NN_DATA = '/work/skrhakv/cryptic-nn'
sys.path.append(f'{PROJECT_DIRECTORY}/src/utils')
sys.path.append(f'{PROJECT_DIRECTORY}/../cryptic-nn/src')
sys.path.append(f'{PROJECT_DIRECTORY}/src/stats/table3-repro')

import eval_utils  # noqa: E402
from eval_utils import CryptoBenchClassifier
import __main__
setattr(__main__, "CryptoBenchClassifier", CryptoBenchClassifier)

import table3_core  # noqa: E402  (reuse apply_smoothing exactly as used for Table 3)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'
DECISION_THRESHOLD = 0.7

VAL_PATH = f'{CRYPTIC_NN_DATA}/val.txt'
EMBEDDINGS_DIR = f'{CRYPTIC_NN_DATA}/embeddings'          # 3B embeddings, 2560-dim
DISTANCE_MATRICES_DIR = f'{CRYPTIC_NN_DATA}/distance-matrices'

MODEL_PATH = f'{PROJECT_DIRECTORY}/data/models/cbs-model.pt'
SMOOTHING_MODEL_PATH = f'{PROJECT_DIRECTORY}/data/models/smoother.pt'
OUTPUT_PATH = f'{PROJECT_DIRECTORY}/src/stats/clustering-hyperparam-sweep/val_predictions_cache.pkl'


def read_val_split(path):
    """val.txt format: pdbid;chain;pocket_type;binding_residues;sequence"""
    proteins = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            pdb_id, chain_id, _pocket_type, residues, sequence = row[0], row[1], row[2], row[3], row[4]
            protein_id = f'{pdb_id.lower()}_{chain_id}'
            if residues == '':
                continue
            pocket = [int(r[1:]) for r in residues.split(' ')]
            proteins.setdefault(protein_id, {'sequence': sequence, 'pockets': []})
            proteins[protein_id]['pockets'].append(pocket)
    return proteins


def main():
    model = torch.load(MODEL_PATH, weights_only=False).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    smoothing_model = torch.load(SMOOTHING_MODEL_PATH, weights_only=False).to(DEVICE)

    proteins = read_val_split(VAL_PATH)
    print(f'{len(proteins)} proteins in the validation split')

    cache = {}
    for i, (protein_id, info) in enumerate(proteins.items()):
        pdb_id, chain_id = protein_id.split('_')
        sequence = info['sequence']

        probabilities = eval_utils.compute_prediction(sequence, model, tokenizer)

        try:
            X_test = np.load(f'{EMBEDDINGS_DIR}/{pdb_id}{chain_id}.npy')
            distance_matrix = np.load(f'{DISTANCE_MATRICES_DIR}/{pdb_id}{chain_id}.npy')
        except FileNotFoundError:
            print(f'[{i+1}/{len(proteins)}] {protein_id}: SKIP (missing embeddings/distance matrix)')
            continue

        if X_test.shape[0] != probabilities.shape[0] or distance_matrix.shape[0] != probabilities.shape[0]:
            print(f'[{i+1}/{len(proteins)}] {protein_id}: SKIP (length mismatch)')
            continue

        predictions_without = (probabilities > DECISION_THRESHOLD).astype(float)
        # deliberately NOT calling smoothing_model.eval() -- matches the main
        # pipeline's behavior (see table3_core.py docstring, point 2)
        predictions_with = table3_core.apply_smoothing(predictions_without, X_test, distance_matrix, smoothing_model)

        cache[protein_id] = {
            'pdb_id': pdb_id,
            'chain_id': chain_id,
            'probabilities': probabilities,
            'predictions_with_smoothing': predictions_with,
            'pockets': info['pockets'],
        }

        if (i + 1) % 25 == 0:
            print(f'[{i+1}/{len(proteins)}] processed')

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(cache, f)
    print(f'Cached predictions for {len(cache)} proteins -> {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
