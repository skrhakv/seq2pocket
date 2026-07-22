"""
Removes sc-PDB_enhanced training entries whose sequence is too similar to a
LIGYSIS test sequence, so train/test overlap is ruled out at the sequence
level rather than only at the UniProt-ID level (see
../extract-scPDB.ipynb, cell 8, for the existing UniProt-based filter this
supplements).

For every (unique) train sequence, aligns it against every (unique) test
sequence with sequence_identity_utils.compute_identity() and keeps the
maximum identity_shortest_gapped score. Train rows whose max identity is
>= --threshold are dropped.

Usage:
    python filter_train_by_identity.py
    python filter_train_by_identity.py --threshold 0.4 --max_workers 32
"""
import argparse
import sys
from pathlib import Path
from multiprocessing import Pool

import pandas as pd

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'
sys.path.append(f'{PROJECT_DIRECTORY}/src/utils')
from sequence_identity_utils import compute_identity  # noqa: E402

DATA_DIRECTORY = f'{PROJECT_DIRECTORY}/data/data-extraction'
TRAIN_PATH = f'{DATA_DIRECTORY}/scPDB_enhanced_binding_sites_translated_filtered.csv'
TEST_PATH = f'{DATA_DIRECTORY}/ligysis_without_unobserved.csv'
OUT_PATH = f'{DATA_DIRECTORY}/scPDB_enhanced_binding_sites_translated_filtered_to_40_identity.csv'
REPORT_PATH = f'{DATA_DIRECTORY}/scPDB_enhanced_binding_sites_translated_filtered_to_40_identity_report.csv'

SEQUENCE_COL = 4  # both CSVs are ';'-delimited, headerless, sequence is the last column
CHECKPOINT_FLUSH_EVERY = 200


def _init_worker(test_sequences, metric):
    global _TEST_SEQS, _METRIC
    _TEST_SEQS = test_sequences
    _METRIC = metric


def _max_identity_against_test_set(args):
    train_idx, train_seq = args
    best_identity, best_test_idx = 0.0, -1
    for test_idx, test_seq in enumerate(_TEST_SEQS):
        identity = compute_identity((train_seq, test_seq))[_METRIC]
        if identity > best_identity:
            best_identity, best_test_idx = identity, test_idx
    return train_idx, best_identity, best_test_idx


def load_checkpoint(checkpoint_path):
    if not Path(checkpoint_path).exists():
        return {}
    checkpoint_df = pd.read_csv(checkpoint_path)
    return {
        row.train_idx: (row.max_identity, row.best_test_idx)
        for row in checkpoint_df.itertuples()
    }


def flush_checkpoint(results: dict, checkpoint_path):
    pd.DataFrame(
        [(idx, ident, test_idx) for idx, (ident, test_idx) in results.items()],
        columns=['train_idx', 'max_identity', 'best_test_idx'],
    ).to_csv(checkpoint_path, index=False)


def compute_max_identities(train_seqs, test_seqs, metric, max_workers, chunk_size, checkpoint_path):
    results = load_checkpoint(checkpoint_path)
    if results:
        print(f'Resuming from checkpoint: {len(results)}/{len(train_seqs)} train sequences already done')

    remaining_tasks = [(i, seq) for i, seq in enumerate(train_seqs) if i not in results]
    if not remaining_tasks:
        return results

    with Pool(max_workers, initializer=_init_worker, initargs=(test_seqs, metric)) as pool:
        done = 0
        for train_idx, best_identity, best_test_idx in pool.imap_unordered(
            _max_identity_against_test_set, remaining_tasks, chunksize=chunk_size
        ):
            results[train_idx] = (best_identity, best_test_idx)
            done += 1
            if done % CHECKPOINT_FLUSH_EVERY == 0:
                print(f'{done}/{len(remaining_tasks)} remaining train sequences processed')
                flush_checkpoint(results, checkpoint_path)

    flush_checkpoint(results, checkpoint_path)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train_path', type=str, default=TRAIN_PATH)
    parser.add_argument('--test_path', type=str, default=TEST_PATH)
    parser.add_argument('--out_path', type=str, default=OUT_PATH)
    parser.add_argument('--report_path', type=str, default=REPORT_PATH)
    parser.add_argument('--threshold', type=float, default=0.4, help='Max allowed identity to any test sequence before a train row is dropped')
    parser.add_argument('--metric', type=str, default='identity_shortest_gapped',
                         choices=['identity_global', 'identity_shortest', 'identity_shortest_gapped'])
    parser.add_argument('--max_workers', type=int, default=32)
    parser.add_argument('--chunk_size', type=int, default=4)
    args = parser.parse_args()
    checkpoint_path = f'{args.out_path}.checkpoint.csv'

    train_df = pd.read_csv(args.train_path, sep=';', header=None)
    test_df = pd.read_csv(args.test_path, sep=';', header=None)

    unique_train_seqs = train_df[SEQUENCE_COL].unique().tolist()
    unique_test_seqs = test_df[SEQUENCE_COL].unique().tolist()
    seq_to_unique_idx = {seq: i for i, seq in enumerate(unique_train_seqs)}

    print(f'Train: {len(train_df)} rows, {len(unique_train_seqs)} unique sequences')
    print(f'Test: {len(test_df)} rows, {len(unique_test_seqs)} unique sequences')
    print(f'Aligning {len(unique_train_seqs) * len(unique_test_seqs)} unique sequence pairs '
          f'using metric={args.metric}, {args.max_workers} workers')

    results = compute_max_identities(
        unique_train_seqs, unique_test_seqs, args.metric, args.max_workers, args.chunk_size, checkpoint_path
    )

    max_identity = [results[seq_to_unique_idx[seq]][0] for seq in train_df[SEQUENCE_COL]]
    best_test_idx = [results[seq_to_unique_idx[seq]][1] for seq in train_df[SEQUENCE_COL]]
    train_df['max_identity_to_test'] = max_identity
    train_df['closest_test_id'] = [
        test_df.iloc[i, 0] if i >= 0 else '' for i in best_test_idx
    ]

    report_df = train_df[[0, 1, 'max_identity_to_test', 'closest_test_id']].copy()
    report_df.columns = ['pdb_id', 'chain', 'max_identity_to_test', 'closest_test_id']
    report_df.to_csv(args.report_path, index=False)

    keep_mask = train_df['max_identity_to_test'] < args.threshold
    filtered_df = train_df.loc[keep_mask, train_df.columns[:5]]
    filtered_df.to_csv(args.out_path, sep=';', header=False, index=False)

    print(f'Kept {keep_mask.sum()}/{len(train_df)} rows (threshold={args.threshold}, metric={args.metric})')
    print(f'Filtered train set written to {args.out_path}')
    print(f'Identity report written to {args.report_path}')

    Path(checkpoint_path).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
