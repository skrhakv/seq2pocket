#!/usr/bin/env python3
"""Seq2Pocket inference: a structure -> ranked 3D binding pockets (JSON).

    python run_seq2pocket.py proteins.txt --task gbs --size 3B -o out.json
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


# HF repo holding the model / smoother files (override for a fork or private copy).
MODELS_REPO = os.environ.get('SEQ2POCKET_MODELS_REPO', 'skrhakv/seq2pocket')

# (task, size) -> model file, size-matched smoother, ESM2 backbone.
MODEL_CATALOGUE = {
    ('gbs', '3B'):   {'model': 'gbs-model.pt',      'smoother': 'smoother.pt',
                      'backbone': 'facebook/esm2_t36_3B_UR50D'},
    ('gbs', '650M'): {'model': 'gbs-model-650M.pt', 'smoother': 'smoother-650M.pt',
                      'backbone': 'facebook/esm2_t33_650M_UR50D'},
    ('cbs', '3B'):   {'model': 'cbs-model.pt',      'smoother': 'smoother.pt',
                      'backbone': 'facebook/esm2_t36_3B_UR50D'},
    ('cbs', '650M'): {'model': 'cbs-model-650M.pt', 'smoother': 'smoother-650M.pt',
                      'backbone': 'facebook/esm2_t33_650M_UR50D'},
}

MAX_LENGTH                   = 1024
SEQUENCE_MAX_LENGTH          = MAX_LENGTH - 2      # room for [CLS]/[SEP]
DECISION_THRESHOLD           = 0.7                 # GBS/CBS positive cutoff
SMOOTHING_DECISION_THRESHOLD = 0.4                 # smoother positive cutoff
POSITIVE_DIST_THR            = 15                  # smoother neighbourhood (Å)
CLUSTER_BANDWIDTH            = 10.0                # MeanShift bandwidth (Å)
POINTS_DENSITY_PER_ATOM      = 50                  # SASA points per atom
PROBE_RADIUS                 = 1.6                 # SASA probe radius (Å)

_STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'ASH', 'GLH', 'HIE', 'HID', 'HIP', 'LYN', 'CYX', 'CYM', 'TYM',
}


class CryptoBenchClassifier(nn.Module):
    """Smoothing classifier. Layer dims come from the loaded weights, so one
    definition serves both sizes (650M input 1280*2, 3B input 2560*2)."""
    def __init__(self, dim=2048, dropout=0.5, input_dim=2560 * 2):
        super().__init__()
        self.layer_1  = nn.Linear(input_dim, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_2  = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_3  = nn.Linear(dim, 1)
        self.relu     = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.dropout2(self.relu(
            self.layer_2(self.dropout1(self.relu(self.layer_1(x)))))))


def _register_unpickle_classes():
    """Register the classes the pickled models reference: FinetunedEsmModel
    (from tutorial/finetuning_utils.py) and CryptoBenchClassifier (under __main__)."""
    import __main__
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'tutorial'))
    from finetuning_utils import FinetunedEsmModel
    setattr(__main__, 'FinetunedEsmModel', FinetunedEsmModel)
    setattr(__main__, 'CryptoBenchClassifier', CryptoBenchClassifier)


_AA3 = {
    'Aba': 'A', 'Ace': 'X', 'Acr': 'X', 'Ala': 'A', 'Aly': 'K', 'Arg': 'R',
    'Asn': 'N', 'Asp': 'D', 'Cas': 'C', 'Ccs': 'C', 'Cme': 'C', 'Csd': 'C',
    'Cso': 'C', 'Csx': 'C', 'Cys': 'C', 'Dal': 'A', 'Dbb': 'T', 'Dbu': 'T',
    'Dha': 'S', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'Glz': 'G', 'His': 'H',
    'Hse': 'S', 'Ile': 'I', 'Leu': 'L', 'Llp': 'K', 'Lys': 'K', 'Men': 'N',
    'Met': 'M', 'Mly': 'K', 'Mse': 'M', 'Nh2': 'X', 'Nle': 'L', 'Ocs': 'C',
    'Pca': 'E', 'Phe': 'F', 'Pro': 'P', 'Ptr': 'Y', 'Sep': 'S', 'Ser': 'S',
    'Thr': 'T', 'Tih': 'A', 'Tpo': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Unk': 'X',
    'Val': 'V', 'Ycm': 'C', 'Sec': 'U', 'Pyl': 'O', 'Mhs': 'H', 'Snm': 'S',
    'Mis': 'S', 'Seb': 'S', 'Hic': 'H', 'Fme': 'M', 'Asb': 'D', 'Sah': 'C',
    'Smc': 'C', 'Tpq': 'Y', 'Onl': 'X', 'Tox': 'W', '5x8': 'X', 'Ddz': 'A',
}


def _three_to_one(code: str) -> str:
    return _AA3.get(code[0].upper() + code[1:].lower(), 'X')


def _get_parser(path: Path):
    from Bio.PDB import PDBParser, MMCIFParser
    if path.suffix.lower() in ('.cif', '.mmcif'):
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def parse_chain(pdb_path: Path, chain_id: str):
    """Return (auth_res_ids, sequence, CA_coords) for the requested chain."""
    model = _get_parser(pdb_path).get_structure('prot', str(pdb_path))[0]
    if chain_id not in model:
        available = ', '.join(c.id for c in model.get_chains())
        raise ValueError(f"chain '{chain_id}' not found (available: {available})")

    res_ids, seq, coords = [], [], []
    for residue in model[chain_id].get_residues():
        het, seqnum, _ = residue.get_id()
        if het.strip() or 'CA' not in residue:      # skip HETATM/water and gaps
            continue
        res_ids.append(seqnum)
        seq.append(_three_to_one(residue.get_resname()))
        coords.append(residue['CA'].get_vector().get_array())
    return res_ids, ''.join(seq), np.array(coords, dtype=np.float32)


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    coords = np.array(coords)
    return np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)


def compute_prediction(sequence, model, tokenizer, device) -> np.ndarray:
    """Per-residue GBS/CBS probabilities, chunked for sequences > 1022."""
    out_all = []
    for i in range(0, len(sequence), SEQUENCE_MAX_LENGTH):
        chunk = sequence[i: i + SEQUENCE_MAX_LENGTH]
        enc = tokenizer(chunk, max_length=MAX_LENGTH, padding='max_length',
                        truncation=True, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(enc)
        if isinstance(out, tuple):                  # CBS model returns a tuple
            out = out[0]
        out = out.squeeze(0)[enc['attention_mask'].squeeze(0).bool()][1:-1]
        out_all.extend(torch.sigmoid(out).detach().cpu().float().numpy())
    return np.array(out_all).flatten()


def generate_embedding(sequence, esm_model, tokenizer, device) -> np.ndarray:
    """Per-residue standalone ESM2 embeddings for the smoother (same chunking)."""
    chunks = []
    for i in range(0, len(sequence), SEQUENCE_MAX_LENGTH):
        enc = tokenizer(sequence[i: i + SEQUENCE_MAX_LENGTH], return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = esm_model(**enc)
        chunks.append(out.last_hidden_state[0, 1:-1].detach().cpu().float().numpy())
    return np.concatenate(chunks, axis=0)


def smooth_predictions(probabilities, embedding, distance_matrix, smoother, device):
    """Promote label-0 residues near a pocket if the smoother agrees."""
    predictions = (probabilities > DECISION_THRESHOLD).astype(float)
    for residue_idx in np.where(predictions == 0.0)[0]:
        close_idx = np.where(distance_matrix[residue_idx] < POSITIVE_DIST_THR)[0]
        close_binding = np.intersect1d(close_idx, np.where(predictions == 1.0)[0])
        if len(close_binding) == 0:
            continue
        elif len(close_binding) == 1:
            surrounding = embedding[close_binding].reshape(-1)
        else:
            surrounding = np.mean(embedding[close_binding], axis=0).reshape(-1)

        concat = torch.tensor(
            np.concatenate([embedding[residue_idx], surrounding]),
            dtype=torch.float32).to(device)
        if torch.sigmoid(smoother(concat).squeeze()) > SMOOTHING_DECISION_THRESHOLD:
            predictions[residue_idx] = 1.0
    return predictions


# SASA surface MeanShift clustering — self-contained port of get_protein_surface_points
# + cluster_atoms_by_surface (src/utils/clustering_utils.py) working on a parsed chain.
def _surface_points_for_chain(chain, predicted_binding_sites: set):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import Bio.PDB.SASA
    import SASA as _custom_sasa
    Bio.PDB.SASA.ShrakeRupley = _custom_sasa.ShrakeRupley   # sets atom.sasa_points

    # Drop non-standard residues in place (chain is a throwaway re-parse).
    for residue in list(chain):
        het, _, _ = residue.get_id()
        if het.strip() or residue.get_resname() not in _STANDARD_RESIDUES:
            chain.detach_child(residue.get_id())
    _custom_sasa.ShrakeRupley(n_points=POINTS_DENSITY_PER_ATOM,
                              probe_radius=PROBE_RADIUS).compute(chain, level='A')

    surface_points, map_point_to_atom, map_atoms_to_residue_id = [], [], {}
    for residue in chain.get_residues():
        residue_id = residue.get_id()[1]
        if residue_id not in predicted_binding_sites:
            continue
        representative = residue['CA'] if 'CA' in residue else next(residue.get_atoms())
        n_points = 0
        for atom in residue.get_atoms():
            atom_id = atom.get_serial_number()
            n_points += len(atom.sasa_points)
            surface_points.append(atom.sasa_points)
            map_point_to_atom.extend([atom_id] * len(atom.sasa_points))
            map_atoms_to_residue_id[atom_id] = residue_id
        if n_points == 0:                           # fully buried: fall back to CA
            rep_id = representative.get_serial_number()
            surface_points.append(representative.get_vector().get_array().reshape(1, -1))
            map_point_to_atom.append(rep_id)
            map_atoms_to_residue_id[rep_id] = residue_id

    if not surface_points:
        return np.empty((0, 3)), np.array([]), {}
    return np.vstack(surface_points), np.array(map_point_to_atom), map_atoms_to_residue_id


def cluster_binding_residues_sasa(chain, binding_auth_ids, probs_by_resnum,
                                  eps=CLUSTER_BANDWIDTH):
    from sklearn.cluster import MeanShift
    from collections import Counter

    all_points, map_point_to_atom, map_atoms_to_residue_id = \
        _surface_points_for_chain(chain, set(binding_auth_ids))
    if all_points.shape[0] == 0:
        return None, None

    point_labels = MeanShift(bandwidth=eps, bin_seeding=True, n_jobs=-1) \
        .fit(all_points).labels_

    atom_labels = {}                                # majority-vote each atom
    for atom_id in np.unique(map_point_to_atom):
        idx = np.where(map_point_to_atom == atom_id)[0]
        atom_labels[atom_id] = Counter(point_labels[idx]).most_common(1)[0][0]

    residue_votes = defaultdict(lambda: defaultdict(int))
    for atom_id, label in atom_labels.items():
        residue_votes[map_atoms_to_residue_id[atom_id]][label] += 1

    cluster_residues = defaultdict(list)
    for resnum, votes in residue_votes.items():
        cluster_residues[max(votes, key=votes.get)].append(resnum)

    n_clusters = max(atom_labels.values()) + 1
    cluster_scores = [
        sum(probs_by_resnum.get(r, 0.0) ** 2 for r in cluster_residues.get(i, []))
        for i in range(n_clusters)]
    return dict(cluster_residues), cluster_scores


def get_model_path(filename, models_dir: Path) -> Path:
    """Local file in models_dir if present, else download just this file from HF."""
    local = models_dir / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    print(f'Fetching {filename} from HF ({MODELS_REPO}) ...', flush=True)
    return Path(hf_hub_download(repo_id=MODELS_REPO, filename=filename))


def _to_device(module, device):
    module.to(device)
    if device == 'cpu':
        module.float()      # models saved as fp16; most CPU ops need fp32
    return module.eval()


def load_models(task, size, models_dir: Path, device, use_smoother: bool):
    from transformers import AutoTokenizer, EsmModel

    spec = MODEL_CATALOGUE[(task, size)]
    backbone = spec['backbone']

    print(f'Loading tokenizer ({backbone}) ...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    print(f'Loading {task.upper()} {size} model: {spec["model"]} ...', flush=True)
    gbs_model = _to_device(
        torch.load(get_model_path(spec['model'], models_dir),
                   weights_only=False, map_location='cpu'), device)

    esm_standalone = smoother = None
    if use_smoother:
        print('Loading standalone ESM2 backbone for smoother embeddings ...', flush=True)
        esm_standalone = _to_device(EsmModel.from_pretrained(backbone), device)
        print(f'Loading smoother: {spec["smoother"]} ...', flush=True)
        smoother = _to_device(
            torch.load(get_model_path(spec['smoother'], models_dir),
                       weights_only=False, map_location='cpu'), device)

    return tokenizer, gbs_model, esm_standalone, smoother


def predict_structure(pdb_path, chain_id, tokenizer, gbs_model, esm_standalone,
                      smoother, device, use_smoother) -> dict:
    """Run the pipeline on one structure -> {structure_id, chain, ranked_pockets}."""
    res_ids, sequence, coords = parse_chain(pdb_path, chain_id)

    probs = compute_prediction(sequence, gbs_model, tokenizer, device)
    if use_smoother:
        embedding = generate_embedding(sequence, esm_standalone, tokenizer, device)
        preds = smooth_predictions(probs, embedding, compute_distance_matrix(coords),
                                   smoother, device)
    else:
        preds = (probs > DECISION_THRESHOLD).astype(float)

    binding_indices  = [i for i, v in enumerate(preds) if v == 1.0]
    binding_auth_ids = [res_ids[i] for i in binding_indices]
    probs_by_resnum  = {res_ids[i]: float(probs[i]) for i in binding_indices}

    ranked_pockets = []
    if binding_auth_ids:
        chain = _get_parser(pdb_path).get_structure('prot', str(pdb_path))[0][chain_id]
        try:
            cluster_residues, cluster_scores = cluster_binding_residues_sasa(
                chain, binding_auth_ids, probs_by_resnum)
        except Exception as e:
            print(f'WARN clustering {pdb_path.name}: {e}', flush=True)
            cluster_residues = None
        if not cluster_residues:                    # fallback: one pocket
            cluster_residues = {0: binding_auth_ids}
            cluster_scores   = [sum(probs_by_resnum[r] ** 2 for r in binding_auth_ids)]

        ranked = sorted(zip(cluster_residues.values(), cluster_scores),
                        key=lambda x: x[1], reverse=True)
        for rank, (resnums, _score) in enumerate(ranked, 1):
            ranked_pockets.append({
                'rank': rank,
                'residues': [f'{chain_id}:{r}' for r in sorted(resnums)],
                'probability': float(np.mean([probs_by_resnum.get(r, 0.0)
                                              for r in resnums])),
            })

    return {'structure_id': pdb_path.stem, 'chain': chain_id,
            'ranked_pockets': ranked_pockets}


def read_batch(batch_file) -> list:
    """Parse the batch manifest into structure paths (one path per line).

    '#' comments and blank lines are ignored. A single protein is just a
    one-line file. Every protein chain in each structure is predicted.
    """
    paths = []
    for line in Path(batch_file).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            paths.append(Path(line))
    return paths


def list_protein_chains(pdb_path) -> list:
    """Chain IDs that contain at least one standard residue with a CA atom."""
    model = _get_parser(pdb_path).get_structure('prot', str(pdb_path))[0]
    return [chain.id for chain in model
            if any(not r.get_id()[0].strip() and 'CA' in r for r in chain)]


def run(args) -> dict:
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print('WARNING: --device cuda but no GPU visible; using CPU.', flush=True)
        device = 'cpu'
    print(f'Device: {device}', flush=True)

    inputs = read_batch(args.batch)
    if not inputs:
        raise SystemExit(f'Batch file {args.batch} lists no structures.')

    use_smoother = not args.no_smooth
    spec = MODEL_CATALOGUE[(args.task, args.size)]

    _register_unpickle_classes()
    tokenizer, gbs_model, esm_standalone, smoother = load_models(
        args.task, args.size, Path(args.models_dir), device, use_smoother)

    predictions = []
    for i, pdb_path in enumerate(inputs, 1):
        try:
            chains = list_protein_chains(pdb_path)
        except Exception as e:
            print(f'[{i}/{len(inputs)}] WARN {pdb_path.name}: {e}', flush=True)
            continue
        for chain in chains:
            try:
                pred = predict_structure(pdb_path, chain, tokenizer, gbs_model,
                                         esm_standalone, smoother, device, use_smoother)
            except Exception as e:
                print(f'[{i}/{len(inputs)}] WARN {pdb_path.name} [{chain}]: {e}',
                      flush=True)
                continue
            print(f'[{i}/{len(inputs)}] {pred["structure_id"]} [{chain}]: '
                  f'{len(pred["ranked_pockets"])} pocket(s).', flush=True)
            predictions.append(pred)

    return {
        'metadata': {
            'tool': 'seq2pocket', 'task': args.task, 'model_size': args.size,
            'model_file': spec['model'], 'smoothing': use_smoother, 'device': device,
            'decision_threshold': DECISION_THRESHOLD,
            'cluster_bandwidth': CLUSTER_BANDWIDTH,
        },
        'predictions': predictions,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Seq2Pocket: predict 3D ligand binding pockets for a batch of structures.')
    p.add_argument('batch',
                   help='Batch manifest: one structure path per line '
                        '(a single protein = a one-line file).')
    p.add_argument('--task', choices=['gbs', 'cbs'], default='gbs',
                   help='General (gbs) or cryptic (cbs) binding sites.')
    p.add_argument('--size', choices=['650M', '3B'], default='3B',
                   help='ESM2 backbone size.')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--no-smooth', action='store_true',
                   help='Skip the embedding-supported smoothing step.')
    p.add_argument('--models-dir', default='/models',
                   help='Directory holding the model .pt files.')
    p.add_argument('-o', '--output', help='Combined JSON path (default: stdout).')
    return p.parse_args(argv)


def main():
    args = parse_args()
    text = json.dumps(run(args), indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
        print(f'Wrote {args.output}', flush=True)
    else:
        print(text)


if __name__ == '__main__':
    main()
