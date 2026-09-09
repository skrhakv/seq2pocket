"""
Recomputes DCC_MAX for each LIGYSIS test pocket, mirroring
src/clustering/atom-pocket-level-evaluation.ipynb's own with-smoothing cell
as closely as possible, using:
- data/data-extraction/ligysis_for_pocket_level_evaluation.csv for the actual
  binding sites (mmCIF, zero-based, per-chain positional indices - confirmed
  by tracing the notebook's own code, which indexes `coordinates` with these
  numbers directly)
- data/visualizations/predictions/ligysis/coordinates/*.npy - the same
  per-residue coordinate arrays the notebook itself loads (`coordinates =
  np.load(...)`), same zero-based indexing as above
- data/visualizations/predictions/ligysis/*.pkl - the with-smoothing model's
  predicted clusters, saved by that same cell, also zero-based-indexed

Usage:
    python3 05_dcc_evaluation.py
"""
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA_DIR = HERE.parent.parent.parent / 'data'
WORK_DIR = HERE / 'work'
PRED_DIR = DATA_DIR / 'visualizations' / 'predictions' / 'ligysis'
COORD_DIR = PRED_DIR / 'coordinates'
TEST_CSV = DATA_DIR / 'data-extraction' / 'ligysis_for_pocket_level_evaluation.csv'
DCC_HIT_THRESHOLD = 12.0
Z_THRESHOLDS = (1.0, 1.5, 2.0, 2.5, 3.0)


def load_actual_binding_sites():
    """protein_id ("pdbid_chain") -> list of mmCIF-position arrays, one per
    annotated pocket - same parsing as eval_utils.read_test_binding_residues."""
    pockets = defaultdict(list)
    with open(TEST_CSV) as f:
        for protein_id, _label, _type, residues_raw, *_ in csv.reader(f, delimiter=';'):
            if residues_raw.strip():
                positions = np.array(sorted(set(int(tok[1:]) for tok in residues_raw.split())))
                pockets[f'{protein_id[:4]}_{protein_id[4:]}'].append(positions)
    return pockets


def main():
    actual_sites = load_actual_binding_sites()
    total_pockets = sum(len(v) for v in actual_sites.values())

    dcc_max = {}
    n_no_prediction = n_missing_coords = 0
    for i, (protein_id, pockets) in enumerate(actual_sites.items()):
        pdb_id, chain_id = protein_id.split('_')
        if (i + 1) % 500 == 0:
            print(f'{i + 1}/{len(actual_sites)} proteins processed')

        pkl_path = PRED_DIR / f'{pdb_id}{chain_id}.pkl'
        coord_path = COORD_DIR / f'{pdb_id}{chain_id}.npy'
        if not pkl_path.exists():
            n_no_prediction += 1  # no predicted residues at all - notebook `continue`s too
            continue
        if not coord_path.exists():
            n_missing_coords += 1
            continue

        cluster_residues, _additional = pickle.load(open(pkl_path, 'rb'))
        coordinates = np.load(coord_path)

        predicted_centers = [coordinates[c].mean(axis=0) for c in cluster_residues if c]
        if not predicted_centers:
            continue

        for pocket_idx, positions in enumerate(pockets):
            valid = positions[positions < len(coordinates)]
            if len(valid) == 0:
                continue
            pocket_center = coordinates[valid].mean(axis=0)
            dcc_max[(pdb_id, chain_id, pocket_idx)] = min(
                np.linalg.norm(pocket_center - c) for c in predicted_centers)

    print(f'{n_no_prediction} proteins with no predicted residues, {n_missing_coords} missing coordinates '
          f'(both still counted as misses in the denominator)')

    all_keys = [(pid.split('_')[0], pid.split('_')[1], idx)
                for pid, pockets in actual_sites.items() for idx in range(len(pockets))]
    assert len(all_keys) == total_pockets

    def hit_rate(keys):
        return sum(1 for k in keys if dcc_max.get(k, float('inf')) < DCC_HIT_THRESHOLD) / len(keys) if keys else float('nan')

    print(f'\n=== full LIGYSIS: DCC_MAX hit-rate = {hit_rate(all_keys):.4f} (n={len(all_keys)}) ===')

    with open(WORK_DIR / 'dcc_max_by_pocket.json', 'w') as f:
        json.dump({f'{k[0]}_{k[1]}_{k[2]}': float(v) for k, v in dcc_max.items()}, f)

    best_z = {}
    with open(WORK_DIR / 'best_match_per_test_pocket.jsonl') as f:
        for row in map(json.loads, f):
            best_z[(row['test_pdb'], row['test_chain'], row['test_pocket_idx'])] = row['best_z_score']

    print('\n=== low pocket-similarity subsets (best ProBiS z-score < threshold, incl. no-match) ===')
    for z in Z_THRESHOLDS:
        low = [k for k in all_keys if best_z.get(k) is None or best_z[k] < z]
        high = [k for k in all_keys if k not in set(low)]
        print(f'z <  {z}  (low similarity)   n={len(low):<5} DCC_MAX={hit_rate(low):.4f}')
        print(f'z >= {z}  (high similarity)  n={len(high):<5} DCC_MAX={hit_rate(high):.4f}\n')

    no_match = [k for k in all_keys if best_z.get(k) is None]
    print(f'no ProBiS match at all         n={len(no_match):<5} DCC_MAX={hit_rate(no_match):.4f}')


if __name__ == '__main__':
    main()
