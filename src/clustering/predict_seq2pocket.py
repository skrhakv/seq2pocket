#!/usr/bin/env python3
"""
predict_seq2pocket.py — Run the seq2pocket CBS model and save predictions
in the DeepLife submission protocol format (predictions.json).

This script reproduces the pipeline from:
  seq2pocket/src/clustering/atom-pocket-level-evaluation-cryptobench.ipynb
  Cell 2 (with smoothing) / Cell 6 (without smoothing)

and writes results as a protocol-conforming predictions.json file that can be
fed directly to deeplife-2026/src/evaluate.py.

Usage (with smoothing, i.e. reproducing cell 2):
    python predict_seq2pocket.py \\
        --seq2pocket-path /path/to/seq2pocket \\
        --model-path /path/to/cbs-model.pt \\
        --smoothing-model-path /path/to/smoother.pt \\
        --cb-path /path/to/cryptobench-clustered-binding-sites.csv \\
        --cif-dir /path/to/cif_files \\
        --embeddings-dir /path/to/embeddings/cryptobench \\
        --coordinates-dir /path/to/coordinates/cryptobench \\
        --output predictions.json

Usage (without smoothing, reproducing cell 6):
    python predict_seq2pocket.py \\
        --seq2pocket-path /path/to/seq2pocket \\
        --model-path /path/to/cbs-model.pt \\
        --cb-path /path/to/cryptobench-clustered-binding-sites.csv \\
        --cif-dir /path/to/cif_files \\
        --coordinates-dir /path/to/coordinates/cryptobench \\
        --output predictions.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run seq2pocket CBS model and save predictions.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seq2pocket-path", required=True,
                   help="Root of the seq2pocket repository (for src/utils imports)")
    p.add_argument("--finetuning-path", required=True,
                   help="Root of the cryptic-nn repository (for src/utils imports)")
    p.add_argument("--model-path", required=True,
                   help="Path to the main CBS model .pt file")
    p.add_argument("--smoothing-model-path", default=None,
                   help="Path to the smoothing classifier .pt file. "
                        "If omitted, smoothing is skipped.")
    p.add_argument("--cb-path", required=True,
                   help="Path to cryptobench-clustered-binding-sites.csv "
                        "(seq2pocket format: pdbid+chain;ligands;CRYPTIC|NON_CRYPTIC;residues;sequence)")
    p.add_argument("--cif-dir", required=True,
                   help="Directory containing {pdb_id}.cif structure files "
                        "(used by atom-level clustering)")
    p.add_argument("--embeddings-dir", default=None,
                   help="Directory containing {pdbid}{chain}.npy ESM-2 embeddings "
                        "(required when using smoothing)")
    p.add_argument("--coordinates-dir", required=True,
                   help="Directory containing {pdbid}{chain}.npy Cα coordinate arrays "
                        "(required for distance matrix used in smoothing)")
    p.add_argument("--output", default="predictions.json",
                   help="Path to write the output predictions.json")
    p.add_argument("--team-name", default="seq2pocket-baseline",
                   help="Team name written into predictions.json metadata")
    p.add_argument("--model-name", default="facebook/esm2_t36_3B_UR50D",
                   help="HuggingFace model name for the tokenizer")
    p.add_argument("--decision-threshold", type=float, default=0.7,
                   help="Probability threshold for classifying residues as binding")
    p.add_argument("--smoothing-threshold", type=float, default=0.4,
                   help="Probability threshold for the smoothing classifier")
    p.add_argument("--positive-distance-threshold", type=float, default=15.0,
                   help="Distance (Å) for searching neighbors for MeanShift algorithm in smoothing step")
    p.add_argument("--eps", type=float, default=9.0,
                   help="MeanShift bandwidth (Å) for atom-level clustering")
    p.add_argument("--pocket-types", nargs="+", default=["CRYPTIC"],
                   help="Pocket types to include from the CB CSV")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def compute_prediction(sequence, model, tokenizer, device):
    """Compute per-residue binding probabilities. Handles long sequences."""
    import torch
    import numpy as np
    MAX_LENGTH = 1024
    SEQ_MAX = MAX_LENGTH - 2

    model.eval()
    final_output = []

    for i in range(0, len(sequence), SEQ_MAX):
        chunk = sequence[i: i + SEQ_MAX]
        tokenized = tokenizer(
            chunk, max_length=MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            output = model(tokenized)
            if isinstance(output, tuple):
                output = output[0]

        output = output.squeeze(0)
        mask = tokenized["attention_mask"].squeeze(0).bool()
        output = output[mask][1:-1]  # strip [CLS] and [EOS]
        final_output.extend(torch.sigmoid(output).detach().cpu().numpy())

    return np.array(final_output, dtype=float).flatten()


def apply_smoothing(probabilities, predictions, coordinates, embeddings,
                    smoothing_model, device, pos_dist_threshold, smooth_threshold):
    """
    Add nearby residues that are predicted binding by the smoothing classifier.
    Returns a modified copy of predictions.
    """
    import torch
    import numpy as np
    from eval_utils import compute_distance_matrix

    distance_matrix = compute_distance_matrix(coordinates)
    predictions_copy = predictions.copy()

    for residue_idx in np.where(predictions == 0.0)[0]:
        current_emb = embeddings[residue_idx]
        close = np.where(distance_matrix[residue_idx] < pos_dist_threshold)[0]
        close_binding = np.intersect1d(close, np.where(predictions == 1.0)[0])

        if len(close_binding) == 0:
            continue
        elif len(close_binding) == 1:
            surround_emb = embeddings[close_binding].reshape(-1)
        else:
            surround_emb = np.mean(embeddings[close_binding], axis=0).reshape(-1)

        concat = torch.tensor(
            np.concatenate((current_emb, surround_emb)), dtype=torch.float32
        ).to(device)

        logit = smoothing_model(concat).squeeze()
        if (torch.sigmoid(logit) > smooth_threshold).float() == 1:
            predictions_copy[residue_idx] = 1.0

    return predictions_copy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- set up sys.path for seq2pocket utilities ----
    seq2pocket_path = Path(args.seq2pocket_path)
    for sub in ["src/utils"]:
        p = str(seq2pocket_path / sub)
        if p not in sys.path:
            sys.path.insert(0, p)

    seq2pocket_path = args.finetuning_path
    if seq2pocket_path not in sys.path:
        sys.path.insert(0, seq2pocket_path)

    import eval_utils
    import clustering_utils
    import cryptoshow_utils
    from eval_utils import CryptoBenchClassifier

    import __main__
    setattr(__main__, "CryptoBenchClassifier", CryptoBenchClassifier)

    # Patch the hardcoded CIF path used by clustering_utils and cryptoshow_utils
    clustering_utils.CIF_FILES = args.cif_dir
    cryptoshow_utils.CIF_FILES_PATH = args.cif_dir

    # ---- load model ----
    from transformers import AutoTokenizer
    print(f"Loading model from {args.model_path} ...")
    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    smoothing_model = None
    if args.smoothing_model_path:
        if args.embeddings_dir is None:
            print("WARNING: --smoothing-model-path given but --embeddings-dir not "
                  "provided; smoothing will be skipped.", file=sys.stderr)
        else:
            print(f"Loading smoothing model from {args.smoothing_model_path} ...")
            smoothing_model = torch.load(
                args.smoothing_model_path, weights_only=False, map_location=device
            )
            smoothing_model.to(device)
            smoothing_model.eval()

    # ---- load ground truth CSV (for sequences and protein list) ----
    binding_residues, sequences = eval_utils.read_test_binding_residues(
        args.cb_path, pocket_types=args.pocket_types
    )
    print(f"Proteins to process: {len(sequences)}")

    # ---- run prediction for each protein ----
    all_predictions = []
    skipped = []

    for i, protein_id in enumerate(sequences.keys()):
        pdb_id, chain_id = protein_id.split("_")
        sequence = sequences[protein_id]
        print(f"[{i+1}/{len(sequences)}] {protein_id} ...", end=" ", flush=True)

        coordinates_path = Path(args.coordinates_dir) / f"{pdb_id}{chain_id}.npy"
        if not coordinates_path.exists():
            print(f"SKIP — coordinates not found: {coordinates_path}")
            skipped.append(f"{protein_id}: coordinates not found")
            continue

        # ---- per-residue inference ----
        probabilities = compute_prediction(sequence, model, tokenizer, device)

        # ---- apply threshold ----
        coordinates = np.load(str(coordinates_path))
        if probabilities.shape[0] != coordinates.shape[0]:
            print(f"SKIP — embedding/coordinate size mismatch "
                  f"({probabilities.shape[0]} vs {coordinates.shape[0]})")
            skipped.append(f"{protein_id}: size mismatch")
            continue

        predictions = (probabilities > args.decision_threshold).astype(float)

        # ---- optional smoothing ----
        if smoothing_model is not None:
            emb_path = Path(args.embeddings_dir) / f"{pdb_id}{chain_id}.npy"
            if not emb_path.exists():
                print(f"(no embeddings, skipping smoothing)", end=" ", flush=True)
            else:
                embeddings = np.load(str(emb_path))
                if embeddings.shape[0] == predictions.shape[0]:
                    predictions = apply_smoothing(
                        probabilities, predictions, coordinates, embeddings,
                        smoothing_model, device,
                        args.positive_distance_threshold, args.smoothing_threshold
                    )

        indices_above_threshold = np.where(predictions > args.decision_threshold)[0]
        if len(indices_above_threshold) == 0:
            print("SKIP — no binding residues predicted")
            skipped.append(f"{protein_id}: no binding residues above threshold")
            continue

        # ---- atom-level clustering ----
        clusters, residue_clusters, cluster_scores, atom_coords, _ = \
            clustering_utils.execute_atom_clustering(
                pdb_id, chain_id,
                indices_above_threshold,
                probabilities[indices_above_threshold],
                eps=args.eps
            )

        if residue_clusters is None:
            print("SKIP — no surface binding residues")
            skipped.append(f"{protein_id}: no surface binding residues")
            continue

        # ---- build ranked pockets ----
        # cluster_scores[k] = mean probability for cluster k
        # residue_clusters[k] = list of auth_seq_id integers for cluster k
        # clusters[k] = list of atom serial numbers for cluster k
        # atom_coords[serial] = BioPython Vector
        cluster_order = np.argsort(cluster_scores)[::-1]  # descending by score

        ranked_pockets = []
        for rank, cluster_id in enumerate(cluster_order, start=1):
            auth_residues = residue_clusters[cluster_id]
            if not auth_residues:
                continue

            atom_indices = clusters[cluster_id]
            cluster_coords = np.array(
                [atom_coords[a].get_array() for a in atom_indices]
            )
            center = np.mean(cluster_coords, axis=0).tolist()

            ranked_pockets.append({
                "rank": rank,
                "probability": float(cluster_scores[cluster_id]),
                "residues": [f"{chain_id}:{res_id}" for res_id in auth_residues],
                "center": center,
            })

        all_predictions.append({
            "pdb_id": pdb_id,
            "chain": chain_id,
            "ranked_pockets": ranked_pockets,
        })
        print(f"OK — {len(ranked_pockets)} pockets")

    # ---- write output ----
    smoothing_note = (
        f" + smoothing (threshold={args.smoothing_threshold})"
        if smoothing_model is not None else ""
    )
    output = {
        "metadata": {
            "team_name": args.team_name,
            "model_version": Path(args.model_path).stem,
            "submission_date": date.today().isoformat(),
            "description": (
                f"seq2pocket CBS model{smoothing_note}, "
                f"atom-level MeanShift clustering (eps={args.eps}), "
                f"decision_threshold={args.decision_threshold}"
            ),
        },
        "predictions": all_predictions,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(all_predictions)} proteins to {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
