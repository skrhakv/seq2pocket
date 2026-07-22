"""
RoG is computed directly from a residue-residue distance matrix via the
classical identity

    Rg^2 = (1/(2N^2)) * sum_{i,j} d_ij^2

which needs only pairwise distances, not absolute 3D coordinates (this is
why CBS -- for which only distance matrices, not coordinates, are available
in this pipeline -- can still be analyzed).
"""
import sys
import numpy as np
from scipy.stats import wilcoxon

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'
sys.path.append(f'{PROJECT_DIRECTORY}/src/utils')
sys.path.append(f'{PROJECT_DIRECTORY}/../cryptic-nn/src')

import eval_utils  # noqa: E402
import cryptoshow_utils  # noqa: E402
import clustering_utils  # noqa: E402

DATA_DIRECTORY = f'{PROJECT_DIRECTORY}/data'
GBS_RAW_PATH = f'{DATA_DIRECTORY}/stats/hole-metrics/raw-gbs.pkl'
CBS_RAW_PATH = f'{DATA_DIRECTORY}/stats/hole-metrics/raw-cbs.pkl'
GBS_COORDINATES_DIR = f'{DATA_DIRECTORY}/coordinates/ligysis'
CBS_DISTANCE_MATRIX_DIR = '/work/skrhakv/cryptic-nn/distance-matrices'

CLUSTER_BANDWIDTH = 9.0             # Angstrom, same MeanShift eps as the manuscript's clustering
CLUSTER_SCORING_METHOD = 'sum_of_squares'
MIN_WITHOUT_RESIDUES = 2            # need >=2 residues in a without-cluster for RoG to be meaningful


def load_distance_matrix(task: str, protein_id: str):
    if task == 'gbs':
        coordinates = np.load(f'{GBS_COORDINATES_DIR}/{protein_id}.npy')
        return eval_utils.compute_distance_matrix(coordinates)
    else:
        return np.load(f'{CBS_DISTANCE_MATRIX_DIR}/{protein_id}.npy')


def rog(distance_matrix: np.ndarray, idx: np.ndarray) -> float:
    """Radius of gyration of residue set `idx`, computed from pairwise
    distances only (equivalent to the usual sum-of-squared-deviations-from-
    centroid definition, but needs no absolute coordinates)."""
    n = len(idx)
    if n < 2:
        return 0.0
    sub = distance_matrix[np.ix_(idx, idx)]
    return float(np.sqrt(np.sum(sub ** 2) / (2 * n * n)))


def cluster_predicted_residues(pdb_id: str, chain_id: str, predicted_mmcif_indices: np.ndarray,
                                probabilities: np.ndarray):
    """Clusters predicted-positive residues into predicted pockets using the
    real seq2pocket clustering code (clustering_utils.execute_atom_clustering,
    same as table3_core.py's cluster_variant), and returns each cluster's
    residues as a 0-based mmCIF-position index array (same indexing as
    y_test/pred_without/pred_with/distance_matrix). Returns [] if there are
    no predicted residues or no resulting surface clusters."""
    if len(predicted_mmcif_indices) == 0:
        return []
    clusters, cluster_residues, cluster_scores, atom_coords, residue_coords = clustering_utils.execute_atom_clustering(
        pdb_id, chain_id, predicted_mmcif_indices, probabilities,
        eps=CLUSTER_BANDWIDTH, scoring_method=CLUSTER_SCORING_METHOD,
    )
    if cluster_residues is None:
        return []
    cluster_residues_mmcif = cryptoshow_utils.map_auth_to_mmcif_numbering_array(
        pdb_id, chain_id, cluster_residues.values(), binding_residues_are_integers=True, numbers_only=True
    )[0]
    return [np.array(sorted(residues)) for residues in cluster_residues_mmcif]


def best_matching_cluster(reference_idx: np.ndarray, candidates: list):
    """The cluster in `candidates` sharing the largest fraction of
    `reference_idx`'s own residues (i.e. "where did most of this cluster's
    residues end up"). None if no candidate overlaps at all."""
    if not candidates or len(reference_idx) == 0:
        return None
    reference_set = set(reference_idx.tolist())
    best_overlap, best_cluster = 0.0, None
    for c in candidates:
        overlap = len(reference_set & set(c.tolist())) / len(reference_set)
        if overlap > best_overlap:
            best_overlap, best_cluster = overlap, c
    return best_cluster


def iter_pocket_footprints(task: str, protein_id: str, d: dict):
    """Yields one dict per WITHOUT-variant predicted cluster that has a
    matching WITH-variant cluster and enough residues:
    protein_id, pocket_index, distance_matrix (shared reference),
    pos_without, pos_with (residue index arrays of the matched cluster
    pair). Downstream scripts compute their own RoG / baselines from
    these."""
    y_test, pred_without, pred_with = d['y_test'], d['pred_without'], d['pred_with']
    probabilities = d['probabilities']
    try:
        distance_matrix = load_distance_matrix(task, protein_id)
    except FileNotFoundError:
        return
    if distance_matrix.shape[0] != len(y_test):
        return

    pdb_id, chain_id = protein_id[:4], protein_id[4:]

    clusters_by_variant = {}
    for variant_name, pred in (('without', pred_without), ('with', pred_with)):
        indices_above_threshold = np.where(pred == 1)[0]
        try:
            clusters_by_variant[variant_name] = cluster_predicted_residues(
                pdb_id, chain_id, indices_above_threshold, probabilities[indices_above_threshold]
            )
        except Exception:
            clusters_by_variant[variant_name] = []

    for pocket_i, cluster_without in enumerate(clusters_by_variant['without']):
        if len(cluster_without) < MIN_WITHOUT_RESIDUES:
            continue
        matched_with = best_matching_cluster(cluster_without, clusters_by_variant['with'])
        if matched_with is None:
            # smoothing reshaped the point cloud enough that no with-variant
            # cluster contains any of this cluster's residues -- shouldn't
            # happen in practice (pred_with is always a superset of
            # pred_without at the residue level) but exclude rather than guess
            continue

        yield {
            'protein_id': protein_id,
            'pocket_index': pocket_i,
            'distance_matrix': distance_matrix,
            'pos_without': cluster_without,
            'pos_with': matched_with,
        }


def paired_test(before, after, alternative='greater'):
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    diff = before - after
    if len(diff) == 0:
        return None, None, 0, float('nan')
    median_diff = float(np.median(diff))
    if np.allclose(diff, 0):
        return None, None, len(diff), median_diff
    stat, p = wilcoxon(diff, alternative=alternative)
    return float(stat), float(p), len(diff), median_diff
