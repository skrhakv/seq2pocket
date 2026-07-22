# THIS CODE IS BASED ON A SNIPPET PROVIDED BY SAMUEL FANCI (GitHub: https://github.com/nexuso1/)

import numpy as np
import pandas as pd

from multiprocessing import Pool
from pathlib import Path
from Bio.Align import PairwiseAligner
from Bio import SeqIO
from Bio.Seq import Seq


def load_sequences(path_or_seq, num=None):
    """Load sequences from a FASTA file, or wrap a single raw sequence string
    into the same one-row DataFrame shape."""
    recs = []
    try:
        with open(path_or_seq, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                recs.append((record.id, record.seq))

        return pd.DataFrame.from_records(recs, columns=['id', 'sequence'])

    except OSError or FileNotFoundError:
        id = f'Sequence {num}'
        seq = Seq(path_or_seq)
        return pd.DataFrame.from_records([(id, seq)], columns=['id', 'sequence'])


def get_aligner():
    aligner = PairwiseAligner()
    aligner.mode = 'global'

    return aligner


def compute_identity(sequences: tuple[str, str], verbose=0):
    """Globally align a pair of sequences and return the alignment score
    together with three %identity variants:
      - identity_global: matches / full alignment length
      - identity_shortest: matches / length of the shorter raw sequence
      - identity_shortest_gapped: matches / length of the shorter sequence's
        aligned region with terminal gaps stripped
    """
    # 1. Set up the aligner and perform the alignment
    seqA, seqB = sequences

    aligner = get_aligner()
    alignments = aligner.align(seqA, seqB)

    # 2. Grab the highest scoring alignment (the first one returned)
    best_alignment = alignments[0]

    # 3. Extract the aligned strings with gaps
    # Index 0 is the target (seqA), Index 1 is the query (seqB).
    # The ':' slice gets the entire sequence.
    seqA_aligned = best_alignment[0, :]
    seqB_aligned = best_alignment[1, :]

    # 4. Count the identical positions
    matches = sum(1 for a, b in zip(seqA_aligned, seqB_aligned) if a == b and a != '-')

    # 5. Calculate Sequence Identity

    # Method A: Divide by the total alignment length
    alignment_length = len(seqA_aligned)
    identity_by_alignment = matches / alignment_length

    # Method B: Divide by the length of the shortest sequence
    min_length = min(len(seqA), len(seqB))
    identity_by_shortest = matches / min_length

    # Method C: Divide by the length of the gapped alignment of the shorter sequence
    shorter_aligned_seq = seqA_aligned if len(seqA) < len(seqB) else seqB_aligned
    stripped = str(shorter_aligned_seq).strip('-')
    identity_by_shortest_gapped_length = matches / len(stripped)

    if verbose > 0:
        # Display Results
        print(f"Aligned Sequence 1: {seqA_aligned}")
        print(f"Aligned Sequence 2: {seqB_aligned}")
        print(f"Total Matches:      {matches}")
        print(f"Identity (Alignment Length): {identity_by_alignment:.1%}")
        print(f"Identity (Shortest Length):  {identity_by_shortest:.1%}")
        print(f"Identity (Shortest Aligned Gapped Length):  {identity_by_shortest_gapped_length:.1%}")

    return {
        'score': best_alignment.score,
        'identity_global': identity_by_alignment,
        'identity_shortest': identity_by_shortest,
        'identity_shortest_gapped': identity_by_shortest_gapped_length,
        'seqA_aligned': seqA_aligned,
        'seqB_aligned': seqB_aligned
    }


def _task_func(args: tuple[int, int, Seq, Seq]):
    i, j, seqA, seqB = args
    return i, j, compute_identity((seqA, seqB))


def compute_similarity_matrix(sequencesA, sequencesB, max_workers=None, chunk_size=None):
    """Compute the full pairwise identity matrices between two sequence
    lists. Materializes one task per (i, j) pair and keeps the aligned
    strings for every pair in the returned DataFrame, so this is only
    suitable for small-to-medium sequence lists -- for large cross products
    (e.g. filtering one dataset against another) compute identities directly
    with compute_identity() instead, keeping only the summary you need."""
    res_global = np.zeros(shape=(len(sequencesA), len(sequencesB)))
    res_shortest, res_shortest_gapped = np.zeros_like(res_global), np.zeros_like(res_global)
    records = []

    tasks = [(i, j, sequencesA[i], sequencesB[j]) for j in range(len(sequencesB)) for i in range(len(sequencesA))]
    count = 0
    with Pool(max_workers) as pool:
        for i, j, result in pool.imap_unordered(_task_func, tasks, chunksize=chunk_size):

            result['idxA'] = i
            result['idxB'] = j

            res_global[i, j] = result['identity_global']
            res_shortest[i, j] = result['identity_shortest']
            res_shortest_gapped[i, j] = result['identity_shortest_gapped']
            records.append(result)
            count += 1

            if count % 10000 == 0:
                print(count)

    return res_global, res_shortest, res_shortest_gapped, pd.DataFrame.from_records(records)


def save_detailed_results(result_df: pd.DataFrame, idxA, idxB, idsA, idsB, out_path):
    result_df['id_seqA'] = [idsA[i] for i in idxA]
    result_df['id_seqB'] = [idsB[i] for i in idxB]
    result_df.to_parquet(out_path.with_suffix('.parquet.zst'), compression='zstd')


def compute_matrix_info(matrix):
    metrics = {
        'mean': np.mean,
        'std': np.std,
        'median': np.median,
        'max': np.max
    }

    return {name: metric(matrix) for name, metric in metrics.items()}


def save_matrix(matrix, idsA, idsB, out_path: Path):
    df = pd.DataFrame(matrix, index=idsA, columns=idsB)
    df.to_parquet(out_path.with_suffix('.parquet.zst'), compression='zstd')


def save_result_summary(glob_matrix, loc_matrix, align_matrix, out_path: Path):
    pd.DataFrame.from_dict({
        'global': compute_matrix_info(glob_matrix),
        'local': compute_matrix_info(loc_matrix),
        'local_gapped': compute_matrix_info(align_matrix),
    }).T.to_csv(out_path.with_suffix('.csv'), float_format='%.4f', index=False)
