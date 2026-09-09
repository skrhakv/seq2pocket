"""
For each LIGYSIS test pocket, compare it (ProBiS motif-vs-motif) against every
scPDB pocket annotated on each of its Foldseek-shortlisted candidate structures.
Keeps the best (highest z-score) match per test pocket - this is what defines
the "low pocket similarity" subset.

Converts only the structures actually needed (test structures + the union of
their shortlisted candidates) from local .cif to legacy .pdb, on demand,
cached in work/pdb_cache/.

Usage:
    python3 03_run_probis.py [--workers 12] [--limit N]
"""
import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb_io
from biotite.structure.io.pdbx import get_structure

HERE = Path(__file__).parent
CIF_DIR = Path('/home/vit/cif_files')
WORK_DIR = HERE / 'work'
PDB_CACHE = WORK_DIR / 'pdb_cache'
PROBIS_TMP = WORK_DIR / 'probis_tmp'
PROBIS = HERE / 'vendor' / 'probis-src' / 'probis'
NO_MATCH = {'z_score': None, 'e_value': None, 'rmsd': None, 'aligned_vertices': None}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def ensure_pdb(pdb_id):
    """Convert local .cif -> legacy .pdb once, cache the result. ProBiS needs
    legacy PDB format; auth numbering is preserved by biotite's PDB writer.
    Writes to a per-process temp file then renames atomically, since many
    worker processes can race to convert the same (popular) structure."""
    PDB_CACHE.mkdir(exist_ok=True)
    out_path = PDB_CACHE / f'{pdb_id}.pdb'
    if out_path.exists():
        return out_path
    cif_path = CIF_DIR / f'{pdb_id}.cif'
    if not cif_path.exists():
        return None
    tmp_path = PDB_CACHE / f'.{pdb_id}.{os.getpid()}.pdb.tmp'
    try:
        structure = get_structure(pdbx.CIFFile.read(str(cif_path)), model=1, use_author_fields=True)
        pdb_file = pdb_io.PDBFile()
        pdb_file.set_structure(structure)
        pdb_file.write(str(tmp_path))
        tmp_path.rename(out_path)
    except Exception as e:
        print(f'  cif->pdb FAILED for {pdb_id}: {e}')
        tmp_path.unlink(missing_ok=True)
        return None
    return out_path


def motif_string(chain, residues):
    """"[:A and (12-19,129-131)]" style ProBiS motif selector for a residue set."""
    residues = sorted(set(residues))
    if not residues:
        return None
    ranges, start = [], residues[0]
    prev = start
    for r in residues[1:] + [None]:
        if r == prev + 1:
            prev = r
            continue
        ranges.append(str(start) if start == prev else f'{start}-{prev}')
        start = prev = r
    return f'[:{chain} and ({",".join(ranges)})]'


def run_probis_pair(test_pdb, test_chain, test_residues, train_pdb, train_chain, train_residues):
    test_pdb_path, train_pdb_path = ensure_pdb(test_pdb), ensure_pdb(train_pdb)
    motif1, motif2 = motif_string(test_chain, test_residues), motif_string(train_chain, train_residues)
    if not (test_pdb_path and train_pdb_path and motif1 and motif2):
        return None

    with tempfile.TemporaryDirectory(dir=PROBIS_TMP) as tmp:
        tmp = Path(tmp)
        nosql, result_json = tmp / 'pair.nosql', tmp / 'pair.json'
        try:
            # NOTE: do not pass -super together with -nosql - it silently produces
            # an empty alignment database even when a real alignment was found
            # (confirmed during the spike on the cluster).
            subprocess.run([
                str(PROBIS), '-compare', '-dist', '3.0',
                '-motif1', motif1, '-f1', str(test_pdb_path), '-c1', test_chain,
                '-motif2', motif2, '-f2', str(train_pdb_path), '-c2', train_chain,
                '-nosql', str(nosql), '-z_score', '0.0',
            ], check=True, capture_output=True, timeout=60, cwd=tmp)
            subprocess.run([
                str(PROBIS), '-results', '-f1', str(test_pdb_path), '-c1', test_chain,
                '-nosql', str(nosql), '-json', str(result_json), '-z_score', '0.0',
            ], check=True, capture_output=True, timeout=60, cwd=tmp)
            data = json.loads(result_json.read_text())
        except Exception:
            return None

    best = None
    for entry in data:
        for aln in entry.get('alignment', []):
            if best is None or aln['scores']['z_score'] > best['z_score']:
                best = aln['scores']
    if best is None:
        return NO_MATCH
    return {k: best[k] for k in ('z_score', 'e_value', 'rmsd', 'aligned_vertices')}


def find_tasks(test_pockets, train_pockets, candidates_by_pdb, limit=None):
    """Yields (task_key, run_probis_pair-kwargs) for every (test pocket,
    candidate structure's pocket) pair to compare."""
    train_by_structure = defaultdict(list)
    train_chains_by_pdb = defaultdict(set)
    for p in train_pockets:
        train_by_structure[(p['pdb_id'], p['chain'])].append(p)
        train_chains_by_pdb[p['pdb_id']].add(p['chain'])

    def resolve_candidates(pdb_id, chain):
        """Foldseek only suffixes a db entry name with "_{chain}" when more
        than one chain from that structure survived its own filtering; a
        structure with exactly one surviving chain gets no suffix ('' key)
        - true for the vast majority of both train and test structures here."""
        by_chain = candidates_by_pdb.get(pdb_id, {})
        if chain in by_chain:
            return by_chain[chain]
        if '' in by_chain:
            return by_chain['']
        return next(iter(by_chain.values())) if len(by_chain) == 1 else []

    def resolve_train_chains(pdb_id, chain):
        """Same ambiguity, mirrored for a Foldseek *target* name against the
        known scPDB-sites.csv chains for that pdb_id. If genuinely ambiguous
        (structure has >1 chain, none matching), try every chain with pockets."""
        if (pdb_id, chain) in train_by_structure:
            return [chain]
        return list(train_chains_by_pdb.get(pdb_id, ()))

    for tp in (test_pockets[:limit] if limit else test_pockets):
        seen_structures = set()
        for c in resolve_candidates(tp['pdb_id'], tp['chain']):
            target_pdb, target_chain_raw = c['target'][:4], c['target'][4:]
            target_chain_raw = target_chain_raw[1:] if target_chain_raw.startswith('_') else target_chain_raw
            for train_chain in resolve_train_chains(target_pdb, target_chain_raw):
                if (target_pdb, train_chain) in seen_structures:
                    continue
                seen_structures.add((target_pdb, train_chain))
                for train_pocket in train_by_structure.get((target_pdb, train_chain), []):
                    key = (tp['pdb_id'], tp['chain'], tp['pocket_idx'], target_pdb, train_chain, train_pocket['ligand'])
                    yield key, dict(
                        test_pdb=tp['pdb_id'], test_chain=tp['chain'], test_residues=tp['auth_residues'],
                        train_pdb=target_pdb, train_chain=train_chain, train_residues=train_pocket['auth_residues'],
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int, default=None, help='only process the first N test pockets (debugging)')
    args = parser.parse_args()

    test_pockets = load_jsonl(WORK_DIR / 'test_pockets.jsonl')
    train_pockets = load_jsonl(WORK_DIR / 'train_pockets.jsonl')
    candidates_by_pdb = {c['query_pdb_id']: c['by_chain'] for c in load_jsonl(WORK_DIR / 'foldseek_top_candidates.jsonl')}

    # a list, not a dict: task keys can legitimately repeat (many structures
    # have several pockets sharing the same ligand name, e.g. multiple Mg2+
    # sites), so keying by task_key would silently collapse distinct pockets
    tasks = list(find_tasks(test_pockets, train_pockets, candidates_by_pdb, args.limit))
    n_test_pockets = min(args.limit, len(test_pockets)) if args.limit else len(test_pockets)
    print(f'{n_test_pockets} test pockets, {len(tasks)} pairwise ProBiS comparisons to run')
    PROBIS_TMP.mkdir(exist_ok=True)

    out_path = WORK_DIR / 'probis_pair_results.jsonl'
    done = 0
    with open(out_path, 'w') as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_probis_pair, **kwargs): key for key, kwargs in tasks}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = None
                print(f'  task {key} raised: {e}')
            done += 1
            if result is not None:
                out.write(json.dumps({
                    'test_pdb': key[0], 'test_chain': key[1], 'test_pocket_idx': key[2],
                    'train_pdb': key[3], 'train_chain': key[4], 'train_ligand': key[5],
                    **result,
                }) + '\n')
            if done % 500 == 0:
                print(f'  {done}/{len(tasks)} pairs done')
                out.flush()

    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
