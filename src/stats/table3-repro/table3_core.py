import pickle
import sys
import numpy as np
import torch
from transformers import AutoTokenizer
from scipy.stats import wilcoxon, binomtest

torch.manual_seed(420)

PROJECT_DIRECTORY = '/home/skrhakv/Projects/seq2pocket'
sys.path.append(f'{PROJECT_DIRECTORY}/src/utils')
sys.path.append(f'{PROJECT_DIRECTORY}/../cryptic-nn/src')

import eval_utils  # noqa: E402
import cryptoshow_utils  # noqa: E402
import clustering_utils  # noqa: E402
from eval_utils import CryptoBenchClassifier
import __main__
setattr(__main__, "CryptoBenchClassifier", CryptoBenchClassifier)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DECISION_THRESHOLD = 0.7
POSITIVE_DISTANCE_THRESHOLD = 15
CLUSTER_EPS = 9
K = 2
DCC_HIT_THRESHOLD_MAIN = 12.0
DCC_HIT_THRESHOLD_STRICT = 4.0
DCC_INF_CAP = 999.0


def apply_smoothing(predictions, X_test, distance_matrix, smoothing_model):
    """Identical logic to the notebooks' smoothing loop."""
    predictions_with = predictions.copy()
    for residue_idx in np.where(predictions == 0.0)[0]:
        current_residue_embedding = X_test[residue_idx]
        close_residues_indices = np.where(distance_matrix[residue_idx] < POSITIVE_DISTANCE_THRESHOLD)[0]
        close_binding_residues_indices = np.intersect1d(close_residues_indices, np.where(predictions == 1.0)[0])
        if len(close_binding_residues_indices) == 0:
            continue
        elif len(close_binding_residues_indices) == 1:
            surrounding_embedding = X_test[close_binding_residues_indices].reshape(-1)
        else:
            surrounding_embedding = np.mean(X_test[close_binding_residues_indices], axis=0).reshape(-1)

        concatenated_embedding = torch.tensor(
            np.concatenate((current_residue_embedding, surrounding_embedding), axis=0), dtype=torch.float32
        ).to(DEVICE)
        smoothing_logit = smoothing_model(concatenated_embedding).squeeze()
        if (torch.sigmoid(smoothing_logit) > eval_utils.SMOOTHING_DECISION_THRESHOLD).float() == 1:
            predictions_with[residue_idx] = 1
    return predictions_with


def cluster_variant(pdb_id, chain_id, predictions, probabilities, actual_binding_sites):
    """Identical logic to the notebooks' clustering + DCC/RRO setup for one variant."""
    indices_above_threshold = np.where(predictions > DECISION_THRESHOLD)[0]
    if len(indices_above_threshold) == 0:
        return None

    clusters, cluster_residues, cluster_scores, atom_coords, residue_coords = clustering_utils.execute_atom_clustering(
        pdb_id, chain_id, indices_above_threshold, probabilities[indices_above_threshold], eps=CLUSTER_EPS
    )
    if cluster_residues is None:
        return None

    cluster_residues_mmcif = cryptoshow_utils.map_auth_to_mmcif_numbering_array(
        pdb_id, chain_id, cluster_residues.values(), binding_residues_are_integers=True, numbers_only=True
    )[0]

    cluster_order = np.argsort(cluster_scores)[::-1]
    N = len(actual_binding_sites)

    predicted_centers = {}
    for cluster_label, atom_indices in clusters.items():
        cluster_coords = np.array([atom_coords[a] for a in atom_indices])
        predicted_centers[cluster_label] = np.mean(cluster_coords, axis=0).get_array()

    return {
        'predicted_centers': predicted_centers,
        'selected_clusters_N': cluster_order[:N],
        'selected_clusters_N_plus_K': cluster_order[:N + K],
        'cluster_residues_mmcif': cluster_residues_mmcif,
    }


def min_dist_to_clusters(center, cluster_ids, predicted_centers):
    d = float('inf')
    for j in cluster_ids:
        dist = np.linalg.norm(center - predicted_centers[j])
        if dist < d:
            d = dist
    return d


def run_evaluation(model_path, esm_model_name, embeddings_dir, coordinates_dir, annotation_path,
                    pocket_types, output_path,
                    smoothing_model_path=f'{PROJECT_DIRECTORY}/data/models/smoother.pt', limit=None):
    model = torch.load(model_path, weights_only=False).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(esm_model_name)

    smoothing_model = torch.load(smoothing_model_path, weights_only=False).to(DEVICE)
    smoothing_model.eval()

    binding_residues, sequences = eval_utils.read_test_binding_residues(data_path=annotation_path, pocket_types=pocket_types)
    number_of_pockets = sum(len(v) for v in binding_residues.values())
    print(f'Total annotated pockets (denominator, matching the notebook): {number_of_pockets}')

    protein_ids = list(sequences.keys())
    if limit is not None:
        protein_ids = protein_ids[:limit]

    records = []
    n_skipped = 0

    for i, protein_id in enumerate(protein_ids):
        pdb_id, chain_id = protein_id.split('_')
        sequence = sequences[protein_id]
        embeddings_path = f'{embeddings_dir}/{pdb_id}{chain_id}.npy'
        coordinates_path = f'{coordinates_dir}/{pdb_id}{chain_id}.npy'
        actual_binding_sites = [np.array([int(x.split('_')[1]) for x in pocket]) for pocket in binding_residues[protein_id]]

        # deliberate fix vs. the notebook -- see module docstring point 1
        probabilities = eval_utils.compute_prediction(sequence, model, tokenizer)

        try:
            coordinates = np.load(coordinates_path)
        except FileNotFoundError:
            print(f'[{i+1}/{len(protein_ids)}] {protein_id}: SKIP (no coordinates)')
            n_skipped += 1
            continue
        distance_matrix = eval_utils.compute_distance_matrix(coordinates)

        if distance_matrix.shape[0] != probabilities.shape[0]:
            print(f'[{i+1}/{len(protein_ids)}] {protein_id}: SKIP (length mismatch: '
                  f'probs={probabilities.shape[0]}, coords={distance_matrix.shape[0]})')
            n_skipped += 1
            continue

        try:
            X_test = np.load(embeddings_path)
        except FileNotFoundError:
            print(f'[{i+1}/{len(protein_ids)}] {protein_id}: SKIP (no embeddings)')
            n_skipped += 1
            continue
        if X_test.shape[0] != distance_matrix.shape[0]:
            print(f'[{i+1}/{len(protein_ids)}] {protein_id}: SKIP (embedding length mismatch)')
            n_skipped += 1
            continue

        predictions_without = (probabilities > DECISION_THRESHOLD).astype(float)
        predictions_with = apply_smoothing(predictions_without, X_test, distance_matrix, smoothing_model)

        variant_data = {}
        for variant_name, preds in [('without', predictions_without), ('with', predictions_with)]:
            variant_data[variant_name] = cluster_variant(pdb_id, chain_id, preds, probabilities, actual_binding_sites)

        if variant_data['without'] is None and variant_data['with'] is None:
            print(f'[{i+1}/{len(protein_ids)}] {protein_id}: no predicted binding residues/surface clusters in either variant')
            continue

        actual_centers = []
        is_small_pocket = []
        for pocket in actual_binding_sites:
            actual_centers.append(np.mean(coordinates[pocket], axis=0))
            is_small_pocket.append(len(pocket) < 10)

        for pocket_i, (pocket, center, small) in enumerate(zip(actual_binding_sites, actual_centers, is_small_pocket)):
            record = {'protein_id': protein_id, 'pocket_index': pocket_i, 'is_small_pocket': small}

            for variant_name in ('without', 'with'):
                v = variant_data[variant_name]
                if v is None:
                    record[f'dcc_n_{variant_name}'] = float('inf')
                    record[f'dcc_n_plus_k_{variant_name}'] = float('inf')
                    record[f'dcc_max_{variant_name}'] = float('inf')
                    record[f'rro_{variant_name}'] = None
                    continue

                record[f'dcc_n_{variant_name}'] = min_dist_to_clusters(center, v['selected_clusters_N'], v['predicted_centers'])
                record[f'dcc_n_plus_k_{variant_name}'] = min_dist_to_clusters(center, v['selected_clusters_N_plus_K'], v['predicted_centers'])

                # DCC_MAX + RRO, Javier et al. Table 4 convention: RRO is
                # computed on the SAME cluster matched by minimum centroid
                # distance (DCC_MAX), not an independently chosen
                # best-overlap cluster, and only counted when that match is
                # <= DCC_HIT_THRESHOLD_MAIN (a "correctly predicted" site).
                # Misses are excluded from RRO (None), not zeroed -- a
                # self-selected denominator, matching Table 4's "mean % RRO
                # across all correctly predicted sites".
                match = clustering_utils.compute_dcc_matched_rro(
                    [pocket], [center], v['predicted_centers'], v['cluster_residues_mmcif'],
                    dcc_hit_threshold=DCC_HIT_THRESHOLD_MAIN,
                )[0]
                record[f'dcc_max_{variant_name}'] = match['dcc']
                record[f'rro_{variant_name}'] = match['rro']

            records.append(record)

        if (i + 1) % 50 == 0:
            print(f'[{i+1}/{len(protein_ids)}] proteins processed, {len(records)} pockets recorded so far')

    with open(output_path, 'wb') as f:
        pickle.dump({'records': records, 'number_of_pockets': number_of_pockets}, f)

    print(f'\nProcessed {len(protein_ids)} proteins from the annotation file ({n_skipped} skipped); '
          f'{len(records)} pockets have >=1 prediction in at least one variant.')
    print(f'(Percentages below use {number_of_pockets} as the denominator, matching the notebook -- '
          f'proteins skipped or with zero predictions implicitly count as misses.)')

    print_and_test(records, number_of_pockets)
    return records, number_of_pockets


def print_and_test(records, number_of_pockets):
    for variant_name in ('without', 'with'):
        dcc_max = np.array([r[f'dcc_max_{variant_name}'] for r in records])
        dcc_n = np.array([r[f'dcc_n_{variant_name}'] for r in records])
        dcc_n_plus_k = np.array([r[f'dcc_n_plus_k_{variant_name}'] for r in records])
        rro_raw = [r[f'rro_{variant_name}'] for r in records]
        rro = np.array([x for x in rro_raw if x is not None])  # Javier et al. Table 4 convention: DCC_MAX<=threshold-correct sites only, misses excluded (not zeroed)

        print(f'\n=== {variant_name} smoothing ===')
        print(f'RRO (Javier et al. Table 4 convention, mean % over DCC_MAX<={DCC_HIT_THRESHOLD_MAIN}A-correct sites only): '
              f'mean={rro.mean():.4f}, median={np.median(rro):.4f}, n={len(rro)}/{len(records)}')
        print(f'DCC_MAX  < 12: {np.sum(dcc_max < 12) / number_of_pockets:.4f} '
              f'(< 4: {np.sum(dcc_max < 4) / number_of_pockets:.4f})')
        print(f'DCC_N+K  < 12: {np.sum(dcc_n_plus_k < 12) / number_of_pockets:.4f} '
              f'(< 4: {np.sum(dcc_n_plus_k < 4) / number_of_pockets:.4f})')
        print(f'DCC_N    < 12: {np.sum(dcc_n < 12) / number_of_pockets:.4f} '
              f'(< 4: {np.sum(dcc_n < 4) / number_of_pockets:.4f})')

        small_mask = np.array([r['is_small_pocket'] for r in records])
        for label, mask in [('Small pockets (<10 residues)', small_mask), ('Large pockets (>=10 residues)', ~small_mask)]:
            if mask.sum() == 0:
                continue
            print(f'  {label}: DCC_MAX < 12: {np.sum(dcc_max[mask] < 12) / mask.sum():.4f} '
                  f'(< 4: {np.sum(dcc_max[mask] < 4) / mask.sum():.4f}), n={int(mask.sum())}')

    # --- paired significance tests -- the added value on top of the notebook ---
    dcc_max_without = np.array([r['dcc_max_without'] for r in records])
    dcc_max_with = np.array([r['dcc_max_with'] for r in records])
    rro_without_raw = [r['rro_without'] for r in records]
    rro_with_raw = [r['rro_with'] for r in records]

    hit_without = dcc_max_without < DCC_HIT_THRESHOLD_MAIN
    hit_with = dcc_max_with < DCC_HIT_THRESHOLD_MAIN
    n_hh = int(np.sum(hit_without & hit_with))
    n_hm = int(np.sum(hit_without & ~hit_with))
    n_mh = int(np.sum(~hit_without & hit_with))
    n_mm = int(np.sum(~hit_without & ~hit_with))
    print(f'\nDCC_MAX hit/miss transition (threshold {DCC_HIT_THRESHOLD_MAIN} A, among '
          f'{len(records)} pockets with >=1 prediction in either variant):')
    print(f'  hit -> hit:   {n_hh}')
    print(f'  hit -> miss:  {n_hm}  (broken by smoothing)')
    print(f'  miss -> hit:  {n_mh}  (rescued by smoothing)')
    print(f'  miss -> miss: {n_mm}')
    if n_hm + n_mh > 0:
        mcnemar_p = binomtest(n_mh, n_hm + n_mh, p=0.5, alternative='greater').pvalue
    else:
        mcnemar_p = None
    print(f'  McNemar exact test (H1: rescued pockets outnumber broken pockets): p = {mcnemar_p}')

    dcc_max_without_c = np.where(np.isinf(dcc_max_without), DCC_INF_CAP, dcc_max_without)
    dcc_max_with_c = np.where(np.isinf(dcc_max_with), DCC_INF_CAP, dcc_max_with)
    diff = dcc_max_without_c - dcc_max_with_c
    if not np.allclose(diff, 0):
        stat, p = wilcoxon(diff, alternative='greater')
        print(f'\nPaired Wilcoxon DCC_MAX (H1: without > with, i.e. smoothing reduces distance): '
              f'n={len(diff)}, median={np.median(diff):.4f}, statistic={stat}, p={p}')

    # RRO is now conditional on DCC_MAX correctness (Javier et al. Table 4 convention), so
    # 'without' and 'with' no longer necessarily flag the same sites as correct -- a valid
    # PAIRED test needs the subset that is correct in BOTH variants (same "properly-paired,
    # quality-given-detected subset" fix used elsewhere in this project).
    both_correct = [a is not None and b is not None for a, b in zip(rro_without_raw, rro_with_raw)]
    n_both_correct = sum(both_correct)
    rro_without = np.array([a for a, m in zip(rro_without_raw, both_correct) if m])
    rro_with = np.array([b for b, m in zip(rro_with_raw, both_correct) if m])
    print(f'\nRRO paired-test subset: {n_both_correct}/{len(records)} sites are DCC_MAX<={DCC_HIT_THRESHOLD_MAIN}A-correct '
          f'in BOTH variants (only these can be validly paired)')

    diff_rro = rro_with - rro_without
    if len(diff_rro) > 0 and not np.allclose(diff_rro, 0):
        stat, p = wilcoxon(diff_rro, alternative='greater')
        print(f'Paired Wilcoxon RRO (H1: with > without, i.e. smoothing increases overlap): '
              f'n={len(diff_rro)}, median={np.median(diff_rro):.4f}, statistic={stat}, p={p}')

    return {
        'transition_table': {'hit_hit': n_hh, 'hit_miss': n_hm, 'miss_hit': n_mh, 'miss_miss': n_mm},
        'mcnemar_p_value': mcnemar_p,
    }
