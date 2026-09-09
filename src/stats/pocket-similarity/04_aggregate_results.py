"""
Collapse probis_pair_results.jsonl to one row per LIGYSIS test pocket: its
best (highest z-score) match anywhere in the Foldseek-shortlisted training
candidates. This is the table to join against per-pocket DCC_MAX results to
define/stratify the "low pocket similarity" subset.

Usage:
    python3 04_aggregate_results.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
WORK_DIR = HERE / 'work'
FIELDS = ('z_score', 'e_value', 'train_pdb', 'train_chain', 'train_ligand')


def main():
    best = {}
    with open(WORK_DIR / 'probis_pair_results.jsonl') as f:
        for row in map(json.loads, f):
            if row['z_score'] is None:
                continue
            key = (row['test_pdb'], row['test_chain'], row['test_pocket_idx'])
            if key not in best or row['z_score'] > best[key]['z_score']:
                best[key] = row

    with open(WORK_DIR / 'test_pockets.jsonl') as f:
        all_test_pockets = sorted((p['pdb_id'], p['chain'], p['pocket_idx']) for p in map(json.loads, f))

    out_path = WORK_DIR / 'best_match_per_test_pocket.jsonl'
    with open(out_path, 'w') as out:
        for pdb_id, chain, pocket_idx in all_test_pockets:
            row = best.get((pdb_id, chain, pocket_idx))
            out.write(json.dumps({
                'test_pdb': pdb_id, 'test_chain': chain, 'test_pocket_idx': pocket_idx,
                **{f'best_{f}': (row[f] if row else None) for f in FIELDS},
            }) + '\n')

    n_matched = sum(1 for k in all_test_pockets if k in best)
    print(f'{len(all_test_pockets)} test pockets total, {n_matched} with a ProBiS hit, '
          f'{len(all_test_pockets) - n_matched} with no hit (no candidate met the clique threshold)')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
