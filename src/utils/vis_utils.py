import os
import sys
import numpy as np

def read_predictions(data_path: str, protein_ids: list[str]) -> dict[str, np.ndarray]:
    '''Read prediction pickle files for given protein IDs from the specified data path.
     Args:
        data_path (str): Path to the directory containing prediction pickle files.
        protein_ids (list of str): List of protein IDs to read predictions for.
    Returns:
        dict: A dictionary mapping protein IDs to their loaded predictions.
    '''
    import pickle
    predictions = {}
    for protein_id in protein_ids:
        if f'{protein_id}.pkl' not in os.listdir(data_path):
            continue
        filename = protein_id.replace('_', '')
        with open(f'{data_path}/{filename}.pkl', 'rb') as f:
            predictions[protein_id] = pickle.load(f)
    return predictions

def reformat_binding_residues(binding_residues: dict) -> dict:
    reformated = {}
    for protein_id, residues in binding_residues.items():
        reformated_protein_id = protein_id.replace('_', '')
        reformated_residues = [np.array([int(residue.split('_')[1]) for residue in pocket]) for pocket in residues]
        reformated[reformated_protein_id] = reformated_residues
    return reformated

def generate_pymol_algebra_selection(protein_id: str, residues: np.ndarray, are_atom_ids=False) -> str:
    if are_atom_ids:
        return f'{protein_id} and id {"+".join([str(i) for i in residues])}'
    else:
        return f'{protein_id} and resi {"+".join([str(i) for i in residues])}'

def get_intersection(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.array(list(set(arr1).intersection(set(arr2))))

DCC_THRESHOLD = 4.0  # Angstroms

def count_successful_predictions(DCCs, threshold=DCC_THRESHOLD):
    return sum(1 for dcc in DCCs if dcc <= threshold)