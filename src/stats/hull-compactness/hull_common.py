import sys
import numpy as np
from scipy.spatial import Delaunay

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'
sys.path.append(f'{PROJECT_DIRECTORY}/src/utils')
sys.path.append(f'{PROJECT_DIRECTORY}/../cryptic-nn/src')

import cryptoshow_utils  # noqa: E402
import clustering_utils  # noqa: E402

DATA_DIRECTORY = f'{PROJECT_DIRECTORY}/data'
GBS_RAW_PATH = f'{DATA_DIRECTORY}/stats/hole-metrics/raw-gbs.pkl'
CBS_RAW_PATH = f'{DATA_DIRECTORY}/stats/hole-metrics/raw-cbs.pkl'

CLUSTER_BANDWIDTH = 9.0             # Angstrom, same MeanShift eps as the manuscript's clustering
CLUSTER_SCORING_METHOD = 'sum_of_squares'
MIN_RESIDUES_FOR_HULL = 4           # a 3D convex hull needs >= 4 non-coplanar points


def get_auth_residue_ids(pdb_id, chain_id, mmcif_indices):
    """AUTH-numbered residue ids for a set of mmCIF-position (sequence-index)
    residues -- no clustering, just the numbering lookup already used
    internally by clustering_utils.execute_atom_clustering."""
    if len(mmcif_indices) == 0:
        return set()
    return set(cryptoshow_utils.map_mmcif_numbering_to_auth(pdb_id, chain_id, np.asarray(mmcif_indices)))


def iter_pocket_gap_fill(protein_id: str, d: dict):
    """Yields one dict per WITHOUT-variant predicted cluster (>=
    MIN_RESIDUES_FOR_HULL residues, non-degenerate hull): protein_id,
    pocket_index, n_without, gap_residues (AUTH ids), filled_residues
    (AUTH ids, subset of gap_residues predicted positive after smoothing)."""
    pred_without, pred_with = d['pred_without'], d['pred_with']
    probabilities = d['probabilities']
    pdb_id, chain_id = protein_id[:4], protein_id[4:]

    indices_without = np.where(pred_without == 1)[0]
    if len(indices_without) == 0:
        return

    try:
        clusters, cluster_residues, cluster_scores, atom_coords, residue_coords = clustering_utils.execute_atom_clustering(
            pdb_id, chain_id, indices_without, probabilities[indices_without],
            eps=CLUSTER_BANDWIDTH, scoring_method=CLUSTER_SCORING_METHOD,
        )
    except Exception:
        return
    if cluster_residues is None:
        return

    try:
        with_positive_auth = get_auth_residue_ids(pdb_id, chain_id, np.where(pred_with == 1)[0])
    except Exception:
        return

    all_coords = {auth_id: coord.get_array() for auth_id, coord in residue_coords.items()}

    for cluster_label in range(len(cluster_residues)):
        without_auth = set(cluster_residues[cluster_label])
        if len(without_auth) < MIN_RESIDUES_FOR_HULL:
            continue

        hull_points = np.array([all_coords[r] for r in without_auth if r in all_coords])
        if len(hull_points) < MIN_RESIDUES_FOR_HULL:
            continue
        try:
            hull = Delaunay(hull_points)
        except Exception:
            continue  # degenerate/coplanar point set

        gap_residues = {
            auth_id for auth_id, coord in all_coords.items()
            if auth_id not in without_auth and hull.find_simplex(coord) >= 0
        }
        if len(gap_residues) == 0:
            continue

        filled_residues = gap_residues & with_positive_auth

        yield {
            'protein_id': protein_id,
            'pocket_index': cluster_label,
            'n_without': len(without_auth),
            'gap_residues': gap_residues,
            'filled_residues': filled_residues,
        }
