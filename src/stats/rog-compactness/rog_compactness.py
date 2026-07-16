#!/usr/bin/env python3
"""
Radius-of-gyration (RoG) compactness of the predicted pocket footprint,
with vs. without smoothing.

reuses raw-{gbs,cbs}.pkl (pred_without/pred_with/y_test, produced
by inference/run_inference_{gbs,cbs}.py) and the existing distance-matrix
sources.
"""
import argparse
import json
import sys
import time
import pickle
import numpy as np

sys.path.append('/home/skrhakv/Projects/seq2pocket/src/stats/rog-compactness')
import rog_common  # noqa: E402


def process_protein(task: str, protein_id: str, d: dict):
    results = []
    for fp in rog_common.iter_pocket_footprints(task, protein_id, d):
        dm = fp['distance_matrix']
        n_without, n_with = len(fp['pos_without']), len(fp['pos_with'])
        results.append({
            'protein_id': fp['protein_id'],
            'pocket_index': fp['pocket_index'],
            'n_without': int(n_without),
            'n_with': int(n_with),
            'rg_without': rog_common.rog(dm, fp['pos_without']),
            'rg_with': rog_common.rog(dm, fp['pos_with']),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['gbs', 'cbs'], required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--raw-path', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    raw_path = args.raw_path or (rog_common.GBS_RAW_PATH if args.task == 'gbs' else rog_common.CBS_RAW_PATH)
    with open(raw_path, 'rb') as f:
        raw = pickle.load(f)
    print(f'Loaded {len(raw)} proteins from {raw_path}')

    protein_ids = list(raw.keys())
    if args.limit is not None:
        protein_ids = protein_ids[:args.limit]

    all_results = []
    t_start = time.time()
    for i, protein_id in enumerate(protein_ids):
        all_results.extend(process_protein(args.task, protein_id, raw[protein_id]))
        if (i + 1) % 200 == 0 or (i + 1) == len(protein_ids):
            elapsed = time.time() - t_start
            print(f'[{i+1}/{len(protein_ids)}] proteins processed, {len(all_results)} pockets so far, '
                  f'{elapsed:.1f}s elapsed')

    print(f'\nTotal: {len(all_results)} pockets from {len(protein_ids)} proteins '
          f'(with a detected "without" footprint of >= {rog_common.MIN_WITHOUT_RESIDUES} residues).')

    rg_without = np.array([r['rg_without'] for r in all_results])
    rg_with = np.array([r['rg_with'] for r in all_results])
    n_without = np.array([r['n_without'] for r in all_results])
    n_with = np.array([r['n_with'] for r in all_results])

    print(f'\nMean N   -- without: {n_without.mean():.2f}, with: {n_with.mean():.2f}')
    print(f'Mean raw RoG -- without: {rg_without.mean():.4f}, with: {rg_with.mean():.4f}')

    print('\n=== WITH vs WITHOUT smoothing (paired, per normalization) ===')
    print('(H1 for each: normalized RoG decreases with smoothing, i.e. more compact)')
    normalizations = {
        'raw (no normalization)': lambda rg, n: rg,
        'RoG / N': lambda rg, n: rg / n,
    }
    test_results = {}
    for name, f in normalizations.items():
        norm_without = f(rg_without, n_without)
        norm_with = f(rg_with, n_with)
        stat, p, n, med = rog_common.paired_test(norm_without, norm_with, alternative='greater')
        print(f'  {name:28s}: mean(without)={norm_without.mean():.5f}, mean(with)={norm_with.mean():.5f}, '
              f'median(without-with)={med:.5f}, n={n}, statistic={stat}, p={p}')
        test_results[name] = {
            'mean_without': float(norm_without.mean()), 'mean_with': float(norm_with.mean()),
            'statistic': stat, 'p_value': p, 'n': n, 'median_diff': med,
        }

    output_path = args.output or f'/home/skrhakv/Projects/seq2pocket/src/stats/rog-compactness/results-{args.task}.json'
    with open(output_path, 'w') as f:
        json.dump({
            'params': {
                'cluster_bandwidth': rog_common.CLUSTER_BANDWIDTH,
                'cluster_scoring_method': rog_common.CLUSTER_SCORING_METHOD,
                'min_without_residues': rog_common.MIN_WITHOUT_RESIDUES,
            },
            'pockets': all_results,
            'with_vs_without': test_results,
        }, f, indent=2)
    print(f'\nSaved full results to {output_path}')


if __name__ == '__main__':
    main()
