# Seq2Pocket Docker image

Runs the Seq2Pocket pipeline and writes ranked 3D
pockets as JSON.

| Flag | Choices | Default | Meaning |
|------|---------|---------|---------|
| `--task`   | `gbs`, `cbs`   | `gbs` | General vs. cryptic binding sites |
| `--size`   | `650M`, `3B`   | `3B`  | ESM2 size |
| `--device` | `auto`, `cpu`, `cuda` | `auto` | device |
| `--no-smooth` | flag        | off   | Skip embedding-supported smoothing |
| `-o`       | directory      | `.` | Output dir; one `<structure_id>.json` per structure |

Pipeline: GBS/CBS prediction → embedding-supported smoothing → SASA MeanShift
clustering.

## Build

```bash
podman build -t seq2pocket .      # or: docker build -t seq2pocket .
```
## Models

On first run the pipeline downloads the
files the chosen `--task`/`--size` need (model + size-matched smoother) from the
Hugging Face repo [`skrhakv/seq2pocket`](https://huggingface.co/skrhakv/seq2pocket)
into the `/models` volume's HF cache; later runs reuse them. A `650M` run pulls
~1.3 GB, a `3B` run ~5.7 GB.

| task | size | model file | smoother |
|------|------|-----------|----------|
| gbs | 3B | `gbs-model.pt` | `smoother.pt` |
| gbs | 650M | `gbs-model-650M.pt` | `smoother-650M.pt` |
| cbs | 3B | `cbs-model.pt` | `smoother.pt` |
| cbs | 650M | `cbs-model-650M.pt` | `smoother-650M.pt` |

## Run

```bash
podman run --rm --gpus all \
  -v seq2pocket-models:/models \ # for storing models between runs
  -v /data/structures:/data/structures:ro,z \ # where your structures are stored & batch file points at them 
  -v ~/seq2pocket-work:/work:z -w /work \ # for the batch file + output - change '~/seq2pocket-work' if needed
  seq2pocket proteins.batch --task gbs --size 3B -o results
```

`~/seq2pocket-work/proteins.batch`:

```
/data/structures/4gqq.pdb
/data/structures/1crn.cif
```

CPU-only, small model, no smoothing (fast smoke test): same command with
`--size 650M --device cpu`.

Interactive shell (override the entrypoint to debug):

```bash
podman run --rm -it --entrypoint bash -v seq2pocket-models:/models \
  -v ~/seq2pocket-work:/work:z -w /work seq2pocket
```

## Output

One JSON file per structure, written to `-o` as `<structure_id>.json`. Residues
are `[chain, seqnum, insertion_code]` tuples.


```json
{
  "format": "0.1.0-beta",
  "metadata": {
    "tool": "seq2pocket",
    "structure_id": "4gqq",
    "input_file": "/data/structures/4gqq.pdb",
    "pdb_model_number": 1,
    "parameters": { "task": "gbs", "model_size": "3B",  "model_file": "gbs-model.pt",
                    "smoothing": true, "device": "cuda",
                    "decision_threshold": 0.7, "cluster_bandwidth": 10.0 }
  },
  "predictions": [
    {
      "ranked_pockets": [
        {
          "rank": 1,
          "auth_residues":  [["A", 14, ""], ["A", 16, ""]],
          "label_residues": [["A", 10, ""], ["A", 12, ""]],
          "probability": 0.83,
          "center": { "x": 12.3, "y": -4.5, "z": 8.1 }
        }
      ]
    }
  ]
}
```
