import csv
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
sys.path.append('/home/skrhakv/cryptoshow-analysis/src')
sys.path.append('/home/vit/Projects/cryptoshow-analysis/src')
import cryptoshow_utils

DATA_PATH = '/home/skrhakv/cryptoshow-analysis/data/A-cluster-ligysis-data/clustered-binding-sites.txt'

def read_test_binding_residues(data_path=DATA_PATH, pocket_types=['CRYPTIC']) -> set[int]:
    cryptic_binding_residues = {}
    sequences = {}

    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            chain_id = row[0][4:]
            protein_id = f'{row[0][:4]}_{chain_id}'
            sequence = row[4]

            if row[3] == '':
                continue

            binding_residue_indices = [f'{chain_id}_{int(i[1:])}'for i in row[3].split(' ')]
            if row[2] in pocket_types:
                if protein_id not in cryptic_binding_residues:
                    cryptic_binding_residues[protein_id] = []
                cryptic_binding_residues[protein_id].append(binding_residue_indices)
            sequences[protein_id] = sequence

    return cryptic_binding_residues, sequences

DCC_THRESHOLD = 4.0  # in Angstroms

def print_plots(DCCs, coverages, dice_coefficients, binding_prediction_scores, number_of_pockets, model):
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    _, axs = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(DCCs, bins=20, color='skyblue', edgecolor='black', alpha=0.7, ax=axs[0])
    axs[0].set_title(f'{model}: Pocket Center Distance (DCC) distribution\n(median={np.median(DCCs):.2f} Å), DCC ≤ {DCC_THRESHOLD} Å: {np.sum(np.array(DCCs) < DCC_THRESHOLD)} / {number_of_pockets}', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('DCC (Å)', fontsize=11)
    axs[0].set_ylabel('Count', fontsize=11)
    axs[0].set_xlim(0, 50)

    sns.histplot(coverages, bins=20, color='salmon', edgecolor='black', alpha=0.7, ax=axs[1])
    axs[1].set_title(f'{model}: Residues Covered Percentage\n(median={np.median(coverages):.1f}%)', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Residues Covered (%)', fontsize=11)
    axs[1].set_ylabel('Count', fontsize=11)

    plt.tight_layout()
    plt.show()

    _, axs = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(dice_coefficients, bins=20, color='mediumseagreen', edgecolor='black', alpha=0.7, ax=axs[0])
    axs[0].set_title(f'{model}: Dice Coefficient Distribution\n(median={np.median(dice_coefficients):.2f})', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('Dice Coefficient', fontsize=11)
    axs[0].set_ylabel('Count', fontsize=11)

    sns.histplot(binding_prediction_scores, bins=20, color='gold', edgecolor='black', alpha=0.7, ax=axs[1])
    axs[1].axvline(0.7, color='red', linestyle='--', linewidth=2, label='Threshold = 0.7')
    axs[1].set_title(f'{model}: Prediction Scores of Test Set CBSs\n(median={np.median(binding_prediction_scores):.2f})', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Prediction Score', fontsize=11)
    axs[1].set_ylabel('Count', fontsize=11)
    axs[1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def compute_pocket_level_metrics(cryptic_binding_residues, predicted_binding_sites, prediction_scores, coordinates_dir):
    DCCs = []
    coverages = []
    dice_coefficients = []
    binding_prediction_scores = []
    number_of_pockets = 0
    for protein_id in cryptic_binding_residues.keys():
        number_of_pockets += len(cryptic_binding_residues[protein_id])
        
        assert len(cryptic_binding_residues[protein_id]) > 0, f"No cryptic binding residues for protein_id: {protein_id}"

        coordinates = np.load(f'{coordinates_dir}/{protein_id.replace("_", "")}.npy')

        # loop over each cryptic binding site
        for actual_cryptic_binding_residue_indices in cryptic_binding_residues[protein_id]:
            dcc = float('inf')
            actual_cryptic_binding_residue_indices = [int(i.split('_')[1]) for i in actual_cryptic_binding_residue_indices]

            # loop over each predicted binding site and select the one with the lowest DCC
            for predicted_cryptic_binding_residue_indices in predicted_binding_sites[protein_id]:
                dcc = min(dcc, cryptoshow_utils.compute_center_distance(coordinates, actual_cryptic_binding_residue_indices, predicted_cryptic_binding_residue_indices))

            if dcc != float('inf'):
                DCCs.append(dcc)

        concatenated_cryptic_binding_residues = np.array(np.unique(np.concatenate(cryptic_binding_residues[protein_id])))
        concatenated_cryptic_binding_residues = [int(i.split('_')[1]) for i in concatenated_cryptic_binding_residues]
        concatenated_predicted_binding_residues = np.array(np.unique(np.concatenate(predicted_binding_sites[protein_id]))) if len(predicted_binding_sites[protein_id]) > 0 else np.array([])
        residues_covered = np.intersect1d(np.array(concatenated_cryptic_binding_residues), concatenated_predicted_binding_residues)

        this_coverage = len(residues_covered) / len(concatenated_cryptic_binding_residues) * 100
        this_dice_coefficient = 2 * len(residues_covered) / (len(concatenated_cryptic_binding_residues) + len(concatenated_predicted_binding_residues)) if (len(concatenated_cryptic_binding_residues) + len(concatenated_predicted_binding_residues)) > 0 else 0

        coverages.append(this_coverage)
        dice_coefficients.append(this_dice_coefficient)
        if protein_id in prediction_scores:
            binding_prediction_scores.extend(prediction_scores[protein_id][concatenated_cryptic_binding_residues])
    return DCCs, coverages, dice_coefficients, binding_prediction_scores, number_of_pockets
