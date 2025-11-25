import sys
import numpy as np

COORDINATES_DIR = '/home/vit/Projects/cryptoshow-analysis/data/A-cluster-ligysis-data/coordinates'

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

def draw_pocket_centers(binding_residues, predicted_residues, protein_id, coordinates_dir=COORDINATES_DIR):
    from pymol import cmd
    from pymol.cgo import COLOR, SPHERE
    coordinates = np.load(f'{coordinates_dir}/{protein_id.replace("_", "")}.npy')
    for i, pocket in enumerate(binding_residues):
        center = coordinates[pocket].mean(axis=0)
        spherelist = [
           COLOR,    1.000,    0.000,    0.000,
           SPHERE,   center[0],   center[1],   center[2], 0.30,
            ]
        cmd.load_cgo(spherelist, f'true_pocket_{i}', 1)
        cmd.pseudoatom(f'true_pocket_center_{i}', pos=center.tolist())

    for i, pocket in enumerate(predicted_residues):
        center = coordinates[pocket].mean(axis=0)
        spherelist = [
           COLOR,    0.000,    0.000,    1.000,
           SPHERE,   center[0],   center[1],   center[2], 0.30,
            ]
        cmd.load_cgo(spherelist, f'predicted_pocket_{i}', 1)
        cmd.pseudoatom(f'predicted_pocket_center_{i}', pos=center.tolist())

def draw_pocket_center_lines(binding_residues, predicted_residues, protein_id):
    from pymol import cmd
    for i, true_pocket in enumerate(binding_residues):
        for j, predicted_pocket in enumerate(predicted_residues):
            cmd.distance(f'distance_{i}_{j}', f'true_pocket_center_{i}', f'predicted_pocket_center_{j}')

DCC_THRESHOLD = 4.0  # Angstroms

def compute_DCCs(protein_id, true_binding_residues, predicted_binding_residues, coordinates_dir=COORDINATES_DIR):
    sys.path.append('/home/skrhakv/cryptoshow-analysis/src')
    sys.path.append('/home/vit/Projects/cryptoshow-analysis/src')
    import cryptoshow_utils
    
    DCCs = []
    for true_pocket in true_binding_residues:
        dcc = float('inf')
        coordinates = np.load(f'{coordinates_dir}/{protein_id.replace("_", "")}.npy')
        for predicted_pocket in predicted_binding_residues:
            this_dcc = cryptoshow_utils.compute_center_distance(coordinates, true_pocket, predicted_pocket)
            print(this_dcc)
            dcc = min(dcc, this_dcc)
        DCCs.append(dcc)
    return DCCs

def count_successful_predictions(DCCs, threshold=DCC_THRESHOLD):
    return sum(1 for dcc in DCCs if dcc <= threshold)

def compute_avg_successful_pocket_size(binding_pockets, predicted_pockets, coordinates_dir='/home/vit/Projects/cryptoshow-analysis/data/A-cluster-ligysis-data/coordinates'):
    avg_success_pocket_size = []
    for protein_id in binding_pockets.keys(): 
        this_binding_residues = binding_pockets[protein_id]
        this_p2rank_predictions = predicted_pockets[protein_id]
        sys.path.append('/home/skrhakv/cryptoshow-analysis/src')
        sys.path.append('/home/vit/Projects/cryptoshow-analysis/src')
        import cryptoshow_utils


        for true_pocket in this_binding_residues:
            dcc = float('inf')
            coordinates = np.load(f'{coordinates_dir}/{protein_id.replace("_", "")}.npy')
            for predicted_pocket in this_p2rank_predictions:
                this_dcc = cryptoshow_utils.compute_center_distance(coordinates, true_pocket, predicted_pocket)
                dcc = min(dcc, this_dcc)
            if dcc < 4.0:
                avg_success_pocket_size.append(len(true_pocket))
    
    return avg_success_pocket_size