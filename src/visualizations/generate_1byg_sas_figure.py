"""
Computes the actual SAS points for 1BYG chain A's predicted residues, using
the exact same method and parameters as clustering_utils.py's
get_protein_surface_points (probe_radius=1.6, n_points=50, this repo's
patched SASA.py - stock Biopython's ShrakeRupley doesn't expose
atom.sasa_points at all), then writes a standalone .pml with the real point
coordinates baked in as a CGO object - no biopython needed to just view it.

Predicted residues = the union of both clusters in the model's actual saved
output for this structure (data/visualizations/predictions/ligysis/1bygA.pkl),
converted from the pkl's mmCIF-positional numbering to real auth residue
numbers via the local .cif (same as before). Colored as a single pocket
color, not per-cluster. The bound Staurosporine ligand (resn STU) is kept
(only waters were ever removed) and shown explicitly as magenta sticks.

Usage:
    python3 src/visualizations/generate_1byg_sas_figure.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdbx import get_structure
from biotite.structure import get_residues

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))  # for the repo's patched SASA.py
import Bio.PDB.SASA
import SASA as patched_sasa
Bio.PDB.SASA.ShrakeRupley = patched_sasa.ShrakeRupley  # adds atom.sasa_points, stock biopython doesn't
from Bio.PDB import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley

CIF_PATH = Path('/home/vit/cif_files/1byg.cif')
PKL_PATH = REPO_ROOT / 'data' / 'visualizations' / 'predictions' / 'ligysis' / '1bygA.pkl'
OUT_PML = Path(__file__).parent / '1byg_sas_points_figure.pml'

PROBE_RADIUS = 1.6           # clustering_utils.PROBE_RADIUS
POINTS_DENSITY_PER_ATOM = 50  # clustering_utils.POINTS_DENSITY_PER_ATOM

# clustering_utils.aal_prot, verbatim
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASH", "GLH", "HIE", "HID", "HIP", "LYN", "CYX", "CYM", "TYM",
}


def predicted_auth_residues():
    """Union of both predicted clusters (mmCIF positions -> auth ids)."""
    cluster_residues, _additional = pickle.load(open(PKL_PATH, 'rb'))
    cif_file = pdbx.CIFFile.read(str(CIF_PATH))
    structure = get_structure(cif_file, model=1, use_author_fields=True)
    ca = structure[(structure.chain_id == 'A') & (structure.atom_name == 'CA') & (structure.element == 'C')]
    residue_ids, _ = get_residues(ca)
    return {int(residue_ids[p]) for cluster in cluster_residues for p in cluster if p < len(residue_ids)}


def compute_sas_points(auth_residues):
    """SAS points (Nx3) for the given auth residue ids, same method/params
    as clustering_utils.get_protein_surface_points."""
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure('1byg', str(CIF_PATH))[0]['A']
    for residue in list(struct):
        if residue.get_resname() not in STANDARD_RESIDUES:
            struct.detach_child(residue.id)

    sr = ShrakeRupley(n_points=POINTS_DENSITY_PER_ATOM, probe_radius=PROBE_RADIUS)
    sr.compute(struct, level='A')

    points = []
    for residue in struct.get_residues():
        if residue.get_id()[1] not in auth_residues:
            continue
        for atom in residue.get_atoms():
            points.extend(atom.sasa_points)
    return np.array(points)


def write_pml(points, auth_residues):
    resi_selection = '+'.join(str(r) for r in sorted(auth_residues))
    cgo_coords = ',\n    '.join(
        f'COLOR, 1.0, 0.85, 0.0,\n    SPHERE, {x:.3f}, {y:.3f}, {z:.3f}, 0.15'
        for x, y, z in points
    )

    OUT_PML.write_text(f'''# Reproduces the "SAS points on predicted residues" SI figure (1BYG chain A,
# human C-terminal Src kinase, Staurosporine site) with the real SAS points
# (same probe_radius={PROBE_RADIUS}, n_points={POINTS_DENSITY_PER_ATOM} as
# clustering_utils.py), not PyMOL's own dot-solvent surface, and no water
# contamination (this repo's SASA pipeline strips non-standard residues
# before computing anything). {len(points)} points on {len(auth_residues)}
# predicted residues, colored as a single pocket (not per-cluster).
#
# Usage:
#   pymol {OUT_PML.name}

reinitialize
fetch 1byg, async=0
remove not chain A
remove solvent
hide everything

bg_color white
set two_sided_lighting, on
show surface, polymer
color grey80, polymer

select predicted_pocket, resi {resi_selection}
color skyblue, predicted_pocket

# Staurosporine (resn STU) - the bound inhibitor, kept and shown explicitly
select ligand, resn STU
show sticks, ligand
color magenta, ligand
util.cnc ligand

python
from pymol import cmd
from pymol.cgo import *
sas_points = [
    {cgo_coords}
]
cmd.load_cgo(sas_points, "sas_points")
python end

zoom predicted_pocket or ligand
deselect
''')
    print(f'wrote {OUT_PML} ({len(points)} SAS points, {len(auth_residues)} predicted residues)')


if __name__ == '__main__':
    residues = predicted_auth_residues()
    pts = compute_sas_points(residues)
    write_pml(pts, residues)
