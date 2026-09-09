"""
Foldseek prescreen: for each unique LIGYSIS test structure, find its top-K
most structurally similar training structures (whole-chain search, not
pocket-specific - this is only a candidate shortlist; ProBiS does the actual
pocket-similarity judgment in step 3).

Usage:
    python3 02_run_foldseek.py [--top-k 20]
"""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CIF_DIR = Path('/home/vit/cif_files')
WORK_DIR = HERE / 'work'
FOLDSEEK = HERE / 'vendor' / 'foldseek' / 'bin' / 'foldseek'


def unique_pdb_ids(jsonl_path):
    with open(jsonl_path) as f:
        return {json.loads(line)['pdb_id'] for line in f}


def link_cifs(pdb_ids, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    n_missing = 0
    for pdb_id in pdb_ids:
        src, dst = CIF_DIR / f'{pdb_id}.cif', target_dir / f'{pdb_id}.cif'
        if not src.exists():
            n_missing += 1
        elif not dst.exists():
            dst.symlink_to(src)
    if n_missing:
        print(f'  WARNING: {n_missing} structures missing from {CIF_DIR}, skipped')
    return len(pdb_ids) - n_missing


def run(cmd):
    print('+', ' '.join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def split_name(name):
    """Foldseek db entry names are cif filenames without extension, suffixed
    "_{chain}" ONLY when more than one chain from that structure survived its
    own filtering - a structure with exactly one surviving chain gets no
    suffix. PDB ids are always 4 chars, so split on that (not a plain
    rsplit('_', 1)); chain '' means "the file's sole chain"."""
    pdb_id, rest = name[:4], name[4:]
    return pdb_id, (rest[1:] if rest.startswith('_') else rest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--threads', type=int, default=12)
    args = parser.parse_args()

    train_pdb_ids = unique_pdb_ids(WORK_DIR / 'train_pockets.jsonl')
    test_pdb_ids = unique_pdb_ids(WORK_DIR / 'test_pockets.jsonl')
    print(f'{len(train_pdb_ids)} unique train structures, {len(test_pdb_ids)} unique test structures')

    train_cif_dir, test_cif_dir = WORK_DIR / 'train_cifs', WORK_DIR / 'test_cifs'
    n_train = link_cifs(train_pdb_ids, train_cif_dir)
    n_test = link_cifs(test_pdb_ids, test_cif_dir)
    print(f'linked {n_train} train cifs, {n_test} test cifs')

    fs_dir = WORK_DIR / 'foldseek'
    (fs_dir / 'tmp').mkdir(parents=True, exist_ok=True)
    train_db, test_db, result_db = fs_dir / 'train_db', fs_dir / 'test_db', fs_dir / 'result_db'
    tsv_path = fs_dir / 'search_results.tsv'

    if not train_db.exists():
        run([FOLDSEEK, 'createdb', train_cif_dir, train_db])
    if not test_db.exists():
        run([FOLDSEEK, 'createdb', test_cif_dir, test_db])
    run([
        FOLDSEEK, 'search', test_db, train_db, result_db, fs_dir / 'tmp',
        '--alignment-type', '2',  # TM-align
        '-a',                     # backtrace, needed by convertalis for alntmscore
        '-e', 'inf',              # keep hits regardless of e-value, we rank by score ourselves
        '--max-seqs', str(max(args.top_k * 5, 100)),  # over-fetch, then trim to top-k below
        '--threads', str(args.threads),
    ])
    run([
        FOLDSEEK, 'convertalis', test_db, train_db, result_db, tsv_path,
        '--format-output', 'query,target,alntmscore,fident,alnlen,evalue',
        '--threads', str(args.threads),
    ])

    hits = defaultdict(list)
    with open(tsv_path) as f:
        for line in f:
            query, target, tmscore, fident, alnlen, evalue = line.rstrip('\n').split('\t')
            hits[query].append((float(tmscore), target, fident, alnlen, evalue))

    # group by pdb_id -> {chain (possibly '' meaning "sole chain"): candidates}
    by_pdb_id = defaultdict(dict)
    for query, rows in hits.items():
        rows.sort(key=lambda r: -r[0])
        query_pdb, query_chain = split_name(query)
        by_pdb_id[query_pdb][query_chain] = [
            {'target': t, 'tmscore': s, 'fident': fi, 'alnlen': al, 'evalue': ev}
            for s, t, fi, al, ev in rows[:args.top_k]
        ]

    out_path = WORK_DIR / 'foldseek_top_candidates.jsonl'
    with open(out_path, 'w') as out:
        for query_pdb, chain_map in by_pdb_id.items():
            out.write(json.dumps({'query_pdb_id': query_pdb, 'by_chain': chain_map}) + '\n')
    print(f'{len(by_pdb_id)} query structures with candidates written to {out_path}')


if __name__ == '__main__':
    main()
