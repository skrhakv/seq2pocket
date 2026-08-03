#!/usr/bin/env python3
import argparse
import json
import sys
import time
import pickle
import numpy as np

sys.path.append('/home/skrhakv/Projects/seq2pocket/src/stats/hull-compactness')
import hull_common  # noqa: E402


def process_protein(protein_id: str, d: dict):
    results = []
    for fp in hull_common.iter_pocket_gap_fill(protein_id, d):
        n_gap = len(fp['gap_residues'])
        n_filled = len(fp['filled_residues'])
        results.append({
            'protein_id': fp['protein_id'],
            'pocket_index': fp['pocket_index'],
            'n_without': fp['n_without'],
            'n_gap': n_gap,
            'n_filled': n_filled,
            'fill_rate': n_filled / n_gap,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['gbs', 'cbs'], required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--raw-path', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    raw_path = args.raw_path or (hull_common.GBS_RAW_PATH if args.task == 'gbs' else hull_common.CBS_RAW_PATH)
    with open(raw_path, 'rb') as f:
        raw = pickle.load(f)
    print(f'Loaded {len(raw)} proteins from {raw_path}')

    protein_ids = list(raw.keys())
    if args.limit is not None:
        protein_ids = protein_ids[:args.limit]

    all_results = []
    t_start = time.time()
    for i, protein_id in enumerate(protein_ids):
        all_results.extend(process_protein(protein_id, raw[protein_id]))
        if (i + 1) % 200 == 0 or (i + 1) == len(protein_ids):
            elapsed = time.time() - t_start
            print(f'[{i+1}/{len(protein_ids)}] proteins processed, {len(all_results)} pockets so far, '
                  f'{elapsed:.1f}s elapsed')

    print(f'\nTotal: {len(all_results)} pockets from {len(protein_ids)} proteins '
          f'(with a detected "without" footprint of >= {hull_common.MIN_RESIDUES_FOR_HULL} residues, '
          f'a non-degenerate convex hull, and >= 1 gap residue).')

    n_gap = np.array([r['n_gap'] for r in all_results])
    n_filled = np.array([r['n_filled'] for r in all_results])
    fill_rate = np.array([r['fill_rate'] for r in all_results])

    pooled_fill_rate = n_filled.sum() / n_gap.sum()
    print(f'\nPer-pocket fill_rate: mean={fill_rate.mean():.4f}, median={np.median(fill_rate):.4f}')
    print(f'Pooled fill_rate (total filled / total gap, fixed denominator): '
          f'{pooled_fill_rate:.4f}  ({int(n_filled.sum())}/{int(n_gap.sum())} gap residues)')
    print(f'Mean gap residues per pocket: {n_gap.mean():.2f}')

    output_path = args.output or f'/home/skrhakv/Projects/seq2pocket/src/stats/hull-compactness/results-{args.task}.json'
    with open(output_path, 'w') as f:
        json.dump({
            'params': {
                'cluster_bandwidth': hull_common.CLUSTER_BANDWIDTH,
                'cluster_scoring_method': hull_common.CLUSTER_SCORING_METHOD,
                'min_residues_for_hull': hull_common.MIN_RESIDUES_FOR_HULL,
            },
            'pockets': all_results,
            'pooled_fill_rate': pooled_fill_rate,
            'mean_fill_rate': float(fill_rate.mean()),
            'median_fill_rate': float(np.median(fill_rate)),
        }, f, indent=2)
    print(f'\nSaved full results to {output_path}')


if __name__ == '__main__':
    main()
