"""
Parse the train (scPDB-sites.csv) and test (ligysis_for_pocket_level_evaluation.csv)
pocket annotation files into a common, auth-numbered format.

- Train: residue numbers are already auth numbers (confirmed by tracing
  cryptoshow_utils.map_auth_to_mmcif_numbering, which takes exactly this kind
  of input) - used as-is, one row = one pocket, nothing merged.
- Test: residue numbers are mmCIF (zero-based, per-chain positional) indices,
  NOT auth numbers (confirmed via eval_utils.read_test_binding_residues and
  how those indices are later used to index data/coordinates/ligysis/*.npy).
  Mapped to auth numbers here from each structure's local .cif file (same
  logic as cryptoshow_utils.map_mmcif_numbering_to_auth, just reading
  straight from /home/vit/cif_files instead of fetching from RCSB).

Output (work/, gitignored):
    train_pockets.jsonl  - {pdb_id, chain, ligand, auth_residues}
    test_pockets.jsonl   - {pdb_id, chain, pocket_idx, auth_residues}
    prepare_data.log
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdbx import get_structure
from biotite.structure import get_residues

HERE = Path(__file__).parent
DATA_DIR = HERE.parent.parent.parent / 'data' / 'data-extraction'
CIF_DIR = Path('/home/vit/cif_files')
WORK_DIR = HERE / 'work'
WORK_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / 'scPDB-sites.csv'
TEST_CSV = DATA_DIR / 'ligysis_for_pocket_level_evaluation.csv'

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def chain_residue_ids(pdb_id, chain_id):
    """Auth residue id per CA position (position i -> auth id) - same
    convention as cryptoshow_utils.map_mmcif_numbering_to_auth."""
    cif_file = pdbx.CIFFile.read(str(CIF_DIR / f'{pdb_id}.cif'))
    protein = get_structure(cif_file, model=1, use_author_fields=True)
    protein = protein[(protein.atom_name == 'CA') & (protein.element == 'C') & (protein.chain_id == chain_id)]
    residue_ids, _ = get_residues(protein)
    return residue_ids


def load_train():
    """pdb_id;chain;ligand;'res1 res2 ...' (auth numbers already)."""
    n_rows = n_skipped = 0
    with open(TRAIN_CSV) as f, open(WORK_DIR / 'train_pockets.jsonl', 'w') as out:
        for pdb_id, chain, ligand, residues_raw, *_ in csv.reader(f, delimiter=';'):
            n_rows += 1
            try:
                auth_residues = [int(r) for r in residues_raw.split()]
            except ValueError:
                auth_residues = []
            if not auth_residues:
                n_skipped += 1
                continue
            out.write(json.dumps({'pdb_id': pdb_id, 'chain': chain, 'ligand': ligand,
                                   'auth_residues': auth_residues}) + '\n')
    log(f'train: {n_rows} rows read, {n_rows - n_skipped} pockets written, {n_skipped} skipped (empty/bad residues)')


def load_test():
    """pdbid+chain;label;type;'X44 Y12 ...';sequence - the leading letter of
    each residue token is the amino-acid code, not a chain id (discarded,
    same as eval_utils.read_test_binding_residues); the number is a
    zero-based mmCIF positional index into that structure's chain."""
    pockets_by_protein = defaultdict(list)
    with open(TEST_CSV) as f:
        for protein_id, _label, _type, residues_raw, *_ in csv.reader(f, delimiter=';'):
            if residues_raw.strip():
                positions = np.array(sorted(set(int(tok[1:]) for tok in residues_raw.split())))
                pockets_by_protein[protein_id].append(positions)

    n_written = n_structures_failed = n_pockets_failed = 0
    with open(WORK_DIR / 'test_pockets.jsonl', 'w') as out:
        for i, (protein_id, pockets) in enumerate(pockets_by_protein.items()):
            pdb_id, chain = protein_id[:4], protein_id[4:]
            if (i + 1) % 200 == 0:
                log(f'  test: {i + 1}/{len(pockets_by_protein)} structures processed')
            try:
                residue_ids = chain_residue_ids(pdb_id, chain)
            except Exception as e:
                log(f'  FAILED structure {protein_id}: {e}')
                n_structures_failed += 1
                n_pockets_failed += len(pockets)
                continue
            for pocket_idx, positions in enumerate(pockets):
                valid = positions[positions < len(residue_ids)]
                if len(valid) < len(positions):
                    log(f'  {protein_id} pocket {pocket_idx}: '
                        f'{len(positions) - len(valid)}/{len(positions)} positions out of range, dropped')
                auth_residues = [int(residue_ids[p]) for p in valid]
                if not auth_residues:
                    n_pockets_failed += 1
                    continue
                out.write(json.dumps({'pdb_id': pdb_id, 'chain': chain, 'pocket_idx': pocket_idx,
                                       'auth_residues': auth_residues}) + '\n')
                n_written += 1

    log(f'test: {len(pockets_by_protein)} structures, {n_written} pockets written, '
        f'{n_structures_failed} structures failed, {n_pockets_failed} pockets dropped')


if __name__ == '__main__':
    log('=== train (scPDB-sites.csv) ===')
    load_train()
    log('=== test (ligysis_for_pocket_level_evaluation.csv) ===')
    load_test()
    (WORK_DIR / 'prepare_data.log').write_text('\n'.join(log_lines) + '\n')
