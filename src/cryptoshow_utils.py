import numpy as np

CIF_FILES_PATH = '/home/vit/Projects/deeplife-project/data/cif_files'

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

def map_residues_to_mmcif(pdb_id, chain_id, binding_residues):
    """
    Map the binding residues to the mmCIF numbering.
    Args:
        pdb_id (str): PDB ID of the protein.
        chain_id (str): Chain ID of the protein.
        binding_residues (set): Set of binding residues in the PDB numbering.
    Returns:
        set: Set of binding residues in the mmCIF numbering.
    """
    import biotite.database.rcsb as rcsb
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure.io.pdbx import get_structure
    from biotite.structure import get_residues

    cif_file_path = rcsb.fetch(pdb_id, "cif", CIF_FILES_PATH)
    cif_file = pdbx.CIFFile.read(cif_file_path)
    
    protein = get_structure(cif_file, model=1)
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

def compute_center_distance(points: np.ndarray, expected_pocket_ids: list[int], actual_pocket_ids: list[int]) -> float:
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