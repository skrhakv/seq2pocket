import pickle
import sys
import time

import numpy as np

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'
sys.path.append(f'{PROJECT_DIRECTORY}/src/utils')
sys.path.append(f'{PROJECT_DIRECTORY}/../cryptic-nn/src')

import cryptoshow_utils  # noqa: E402
import clustering_utils  # noqa: E402

CACHE_PATH = f'{PROJECT_DIRECTORY}/src/stats/clustering-hyperparam-sweep/val_predictions_cache.pkl'
COORDINATES_DIR = f'{PROJECT_DIRECTORY}/data/coordinates/cryptobench'

DCC_HIT_THRESHOLD = 12.0
K = 2

DEFAULT_BANDWIDTH = 9
DEFAULT_PROBE_RADIUS = 1.6
DEFAULT_POINT_DENSITY = 50

GRIDS = {
    'bandwidth': [6, 7, 9, 12, 15],
    'probe_radius': [1.4, 1.6, 2.0],
    'point_density': [20, 50, 100],
}


def evaluate_config(cache, bandwidth, probe_radius, point_density):
    clustering_utils.PROBE_RADIUS = probe_radius
    clustering_utils.POINTS_DENSITY_PER_ATOM = point_density

    hits = 0
    total_pockets = 0
    pfi_counts = []
    clustering_time = 0.0
    n_clustering_calls = 0

    for protein_id, entry in cache.items():
        pdb_id, chain_id = entry['pdb_id'], entry['chain_id']
        predictions = entry['predictions_with_smoothing']
        probabilities = entry['probabilities']

        indices_above_threshold = np.where(predictions == 1.0)[0]
        total_pockets += len(entry['pockets'])
        if len(indices_above_threshold) == 0:
            continue

        t0 = time.perf_counter()
        clusters, cluster_residues, cluster_scores, atom_coords, residue_coords = clustering_utils.execute_atom_clustering(
            pdb_id, chain_id, indices_above_threshold, probabilities[indices_above_threshold], eps=bandwidth
        )
        clustering_time += time.perf_counter() - t0
        n_clustering_calls += 1
        if cluster_residues is None:
            continue

        cluster_residues_mmcif = cryptoshow_utils.map_auth_to_mmcif_numbering_array(
            pdb_id, chain_id, cluster_residues.values(), binding_residues_are_integers=True, numbers_only=True
        )[0]

        try:
            coordinates = np.load(f'{COORDINATES_DIR}/{pdb_id}{chain_id}.npy')
        except FileNotFoundError:
            continue

        predicted_centers = {}
        for cluster_label, atom_indices in clusters.items():
            cluster_coords = np.array([atom_coords[a] for a in atom_indices])
            predicted_centers[cluster_label] = np.mean(cluster_coords, axis=0).get_array()

        cluster_order = np.argsort(cluster_scores)[::-1]
        N = len(entry['pockets'])
        selected_clusters_N_plus_K = cluster_order[:N + K]

        for pocket in entry['pockets']:
            pocket_set = set(pocket)
            actual_center = np.mean(coordinates[pocket], axis=0)

            d = min(
                (np.linalg.norm(actual_center - predicted_centers[j]) for j in selected_clusters_N_plus_K),
                default=float('inf'),
            )
            if d < DCC_HIT_THRESHOLD:
                hits += 1

            overlap_count = sum(1 for residues in cluster_residues_mmcif if pocket_set.intersection(residues))
            if overlap_count > 0:
                pfi_counts.append(overlap_count)

    dcc_rate = hits / total_pockets if total_pockets else float('nan')
    pfi = np.mean(pfi_counts) if pfi_counts else float('nan')
    sec_per_protein = clustering_time / n_clustering_calls if n_clustering_calls else float('nan')
    return dcc_rate, pfi, sec_per_protein


def main():
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    print(f'{len(cache)} cached proteins')

    print(f'\n{"hyperparameter":<15}{"value":<10}{"DCCtop-(N+2)":<15}{"PFI":<10}{"sec/protein":<12}')
    for param, grid in GRIDS.items():
        bandwidth, probe_radius, point_density = DEFAULT_BANDWIDTH, DEFAULT_PROBE_RADIUS, DEFAULT_POINT_DENSITY
        for value in grid:
            if param == 'bandwidth':
                bandwidth = value
            elif param == 'probe_radius':
                probe_radius = value
            elif param == 'point_density':
                point_density = value

            dcc_rate, pfi, sec_per_protein = evaluate_config(cache, bandwidth, probe_radius, point_density)
            marker = ' <- paper value' if value in (
                DEFAULT_BANDWIDTH, DEFAULT_PROBE_RADIUS, DEFAULT_POINT_DENSITY
            ) and value == {'bandwidth': DEFAULT_BANDWIDTH, 'probe_radius': DEFAULT_PROBE_RADIUS,
                             'point_density': DEFAULT_POINT_DENSITY}[param] else ''
            print(f'{param:<15}{value:<10}{dcc_rate:<15.4f}{pfi:<10.4f}{sec_per_protein:<12.4f}{marker}')


if __name__ == '__main__':
    main()
