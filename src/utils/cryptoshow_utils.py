import numpy as np
from multipledispatch import dispatch

CIF_FILES_PATH = '/home/vit/Projects/deeplife-project/data/cif_files'
# CIF_FILES_PATH = '/home/skrhakv/cryptoshow-analysis/data/cif_files'

mapping = {'Aba': 'A', 'Ace': 'X', 'Acr': 'X', 'Ala': 'A', 'Aly': 'K', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cas': 'C',
           'Ccs': 'C', 'Cme': 'C', 'Csd': 'C', 'Cso': 'C', 'Csx': 'C', 'Cys': 'C', 'Dal': 'A', 'Dbb': 'T', 'Dbu': 'T',
           'Dha': 'S', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'Glz': 'G', 'His': 'H', 'Hse': 'S', 'Ile': 'I', 'Leu': 'L',
           'Llp': 'K', 'Lys': 'K', 'Men': 'N', 'Met': 'M', 'Mly': 'K', 'Mse': 'M', 'Nh2': 'X', 'Nle': 'L', 'Ocs': 'C',
           'Pca': 'E', 'Phe': 'F', 'Pro': 'P', 'Ptr': 'Y', 'Sep': 'S', 'Ser': 'S', 'Thr': 'T', 'Tih': 'A', 'Tpo': 'T',
           'Trp': 'W', 'Tyr': 'Y', 'Unk': 'X', 'Val': 'V', 'Ycm': 'C', 'Sec': 'U', 'Pyl': 'O', 'Mhs': 'H', 'Snm': 'S',
           'Mis': 'S', 'Seb': 'S', 'Hic': 'H', 'Fme': 'M', 'Asb': 'D', 'Sah': 'C', 'Smc': 'C', 'Tpq': 'Y', 'Onl': 'X',
           'Tox': 'W', '5x8': 'X', 'Ddz': 'A'}


def three_to_one(three_letter_code):
    if three_letter_code[0].upper() + three_letter_code[1:].lower() not in mapping:
        return 'X'
    return mapping[three_letter_code[0].upper() + three_letter_code[1:].lower()]

def map_auth_to_mmcif_numbering(pdb_id: str, chain_id: str, binding_residues: set, auth=True) -> tuple[list[str], str]:
    """
    Map the binding residues from auth labeling to the mmCIF numbering (zero-based).
    Args:
        pdb_id (str): PDB ID of the protein.
        chain_id (str): Chain ID of the protein.
        binding_residues (set): Set of binding residues in the PDB numbering.
        auth (bool): Whether to use auth labeling (True) or PDB assigned sequential numbering (False).
    Returns:
        List[str]: List of binding residues in the mmCIF numbering.
        str: The amino acid sequence of the chain.
    """
    import biotite.database.rcsb as rcsb
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdbx import get_structure
    from biotite.structure import get_residues

    cif_file_path = rcsb.fetch(pdb_id, "cif", CIF_FILES_PATH)
    cif_file = pdbx.CIFFile.read(cif_file_path)
    
    protein = get_structure(cif_file, model=1, use_author_fields=auth)
    protein = protein[(protein.atom_name == "CA") 
                        & (protein.element == "C") 
                        & (protein.chain_id == chain_id) ]
    residue_ids, residue_types = get_residues(protein)

    sequence = ''
    mapped_binding_residues = []
    for i in range(len(residue_ids)):
        residue_id = str(residue_ids[i])
        amino_acid = three_to_one(residue_types[i])

        if residue_id in binding_residues:
            mapped_binding_residues.append(f'{amino_acid}{i}')

        sequence += amino_acid

    return mapped_binding_residues, sequence

def map_auth_to_mmcif_numbering_array(pdb_id: str, chain_id: str, binding_residues_list: list[set], auth=True, numbers_only=False, binding_residues_are_integers=False) -> tuple[list[str], str]:
    """
    Map the binding residues from auth labeling to the mmCIF numbering (zero-based).
    Args:
        pdb_id (str): PDB ID of the protein.
        chain_id (str): Chain ID of the protein.
        binding_residues (set): Set of binding residues in the PDB numbering.
        auth (bool): Whether to use auth labeling (True) or PDB assigned sequential numbering (False).
    Returns:
        List[str]: List of binding residues in the mmCIF numbering.
        str: The amino acid sequence of the chain.
    """
    import biotite.database.rcsb as rcsb
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdbx import get_structure
    from biotite.structure import get_residues

    cif_file_path = rcsb.fetch(pdb_id, "cif", CIF_FILES_PATH)
    cif_file = pdbx.CIFFile.read(cif_file_path)
    
    protein = get_structure(cif_file, model=1, use_author_fields=auth)
    protein = protein[(protein.atom_name == "CA") 
                        & (protein.element == "C") 
                        & (protein.chain_id == chain_id) ]
    residue_ids, residue_types = get_residues(protein)

    sequence = ''
    mapped_binding_residues = [[] for _ in range(len(binding_residues_list))]
    for i in range(len(residue_ids)):
        if binding_residues_are_integers:
            residue_id = int(residue_ids[i])
        else:
            residue_id = str(residue_ids[i])
        amino_acid = three_to_one(residue_types[i])
        for binding_site_index, binding_residues in enumerate(binding_residues_list):
            if residue_id in binding_residues:
                if numbers_only:
                    mapped_binding_residues[binding_site_index].append(i)
                else:
                    mapped_binding_residues[binding_site_index].append(f'{amino_acid}{i}')

        sequence += amino_acid

    return mapped_binding_residues, sequence

def map_mmcif_numbering_to_auth(pdb_id: str, chain_id: str, binding_residues: np.ndarray, auth=True, binding_scores=None) -> list[int]:
    """
    Map the binding residues from mmCIF numbering (zero-based) to the auth labeling.
    Args:
        pdb_id (str): PDB ID of the protein.
        chain_id (str): Chain ID of the protein.
        binding_residues (np.ndarray): Set of binding residues in the PDB numbering.
    Returns:
        list[int]: List of binding residues in the auth labeling.
    """
    import biotite.database.rcsb as rcsb
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdbx import get_structure
    from biotite.structure import get_residues

    if binding_scores is not None:
        assert len(binding_residues) == len(binding_scores), "Length of binding residues and binding scores must be the same"
    
    cif_file_path = rcsb.fetch(pdb_id, "cif", CIF_FILES_PATH)
    cif_file = pdbx.CIFFile.read(cif_file_path)
    
    protein = get_structure(cif_file, model=1, use_author_fields=auth)
    protein = protein[(protein.atom_name == "CA") 
                        & (protein.element == "C") 
                        & (protein.chain_id == chain_id) ]
    residue_ids, _ = get_residues(protein)

    mapped_binding_residues = []
    scores = []
    for i in range(len(residue_ids)):

        residue_index = np.where(binding_residues == i)[0]
        if len(residue_index) > 0:
            residue_id = int(residue_ids[i])
            mapped_binding_residues.append(residue_id)
            if binding_scores is not None:
                scores.append(binding_scores[residue_index[0]])

    if binding_scores is None:
        return mapped_binding_residues
    return mapped_binding_residues, scores

def map_mmcif_numbering_to_auth_array(pdb_id: str, chain_id: str, binding_residues_list: list[np.ndarray], auth=True, binding_scores=None) -> list[int]:
    """
    Map the binding residues from mmCIF numbering (zero-based) to the auth labeling.
    Args:
        pdb_id (str): PDB ID of the protein.
        chain_id (str): Chain ID of the protein.
        binding_residues (np.ndarray): Set of binding residues in the PDB numbering.
    Returns:
        list[int]: List of binding residues in the auth labeling.
    """
    import biotite.database.rcsb as rcsb
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdbx import get_structure
    from biotite.structure import get_residues
    
    cif_file_path = rcsb.fetch(pdb_id, "cif", CIF_FILES_PATH)
    cif_file = pdbx.CIFFile.read(cif_file_path)
    
    protein = get_structure(cif_file, model=1, use_author_fields=auth)
    protein = protein[(protein.atom_name == "CA") 
                        & (protein.element == "C") 
                        & (protein.chain_id == chain_id) ]
    residue_ids, _ = get_residues(protein)

    mapped_binding_residues = [[] for _ in range(len(binding_residues_list))]
    mapped_scores = [[] for _ in range(len(binding_residues_list))]

    for i in range(len(residue_ids)):
        for binding_site_index, binding_residues in enumerate(binding_residues_list):
            residue_index = np.where(binding_residues == i)[0]
            if len(residue_index) > 0:
                residue_id = int(residue_ids[i])
                mapped_binding_residues[binding_site_index].append(residue_id)
                if binding_scores is not None:
                    mapped_scores[binding_site_index].append(binding_scores[residue_index[0]])

    if binding_scores is None:
        return mapped_binding_residues
    return mapped_binding_residues, mapped_scores
@dispatch(np.ndarray, list, np.ndarray)
def compute_center_distance(points: np.ndarray, expected_pocket_ids: list[int], actual_pocket_ids: np.ndarray) -> float:
    """
    Compute the distance from the predicted pocket center to the expected pocket center.

    Args:
        points (np.ndarray): Array of shape (N, 3) containing the coordinates of the points.
        expected_pocket_ids (list[int]): List of indices corresponding to the expected pocket.
        actual_pocket_ids (list[int]): List of indices corresponding to the predicted pocket.
    Returns:
        float: The DCC value, which is the distance from the predicted pocket center to the ligand center.
    """
    expected_coords = points[expected_pocket_ids].mean(axis=0)
    actual_coords = points[actual_pocket_ids].mean(axis=0)

    dist = np.linalg.norm(expected_coords - actual_coords)
    return dist

@dispatch(np.ndarray, np.ndarray)
def compute_center_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    Compute the distance from the predicted pocket center to the expected pocket center.

    Args:
        points (np.ndarray): Array of shape (N, 3) containing the coordinates of the points.
        expected_pocket_ids (list[int]): List of indices corresponding to the expected pocket.
        actual_pocket_ids (list[int]): List of indices corresponding to the predicted pocket.
    Returns:
        float: The DCC value, which is the distance from the predicted pocket center to the ligand center.
    """
    expected_coords = points1.mean(axis=0)
    actual_coords = points2.mean(axis=0)
    dist = np.linalg.norm(expected_coords - actual_coords)
    return dist

def get_distance_matrix(pdb_id, chain_id):
    from scipy.spatial import distance_matrix

    coords = get_coordinates(pdb_id, chain_id)
    dist_matrix = distance_matrix(coords, coords)

    return dist_matrix

def get_coordinates(pdb_id, chain_id, auth=True):
    import biotite.database.rcsb as rcsb
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdbx import get_structure
    from biotite.structure import get_residues
    
    cif_file_path = rcsb.fetch(pdb_id, "cif", CIF_FILES_PATH)
    cif_file = pdbx.CIFFile.read(cif_file_path)
    
    protein = get_structure(cif_file, model=1, use_author_fields=auth)
    protein = protein[(protein.atom_name == "CA") 
                        & (protein.element == "C") 
                        & (protein.chain_id == chain_id) ]
    residue_ids, residue_types = get_residues(protein)

    coords = protein.coord

    assert len(residue_ids) == len(coords) == len(residue_types), "Number of residues and coordinates do not match"

    return coords