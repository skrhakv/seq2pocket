# Seq2Pocket Docker image

Runs the Seq2Pocket pipeline and writes ranked 3D
pockets as JSON.

| Flag | Choices | Default | Meaning |
|------|---------|---------|---------|
| `--task`   | `gbs`, `cbs`   | `gbs` | General vs. cryptic binding sites |
| `--size`   | `650M`, `3B`   | `3B`  | ESM2 size |
| `--device` | `auto`, `cpu`, `cuda` | `auto` | device |
| `--no-smooth` | flag        | off   | Skip embedding-supported smoothing |
| `-o`       | path           | stdout | Output JSON |

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
  -v seq2pocket-models:/models \
  -v /data/structures:/data/structures:ro,z \
  -v ~/seq2pocket-work:/work:z -w /work \
  seq2pocket proteins.batch --task gbs --size 3B -o out.json
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

One combined JSON (`-o` / stdout) with a `predictions` entry per structure chain:

```json
{
  "metadata": { "tool": "seq2pocket", "task": "gbs", "model_size": "3B",
                "smoothing": true, "device": "cuda", ... },
  "predictions": [
    {
      "structure_id": "4gqq",
      "chain": "A",
      "ranked_pockets": [
        { "rank": 1, "residues": ["A:14", "A:16", ...], "probability": 0.83 }
      ]
    }
  ]
}
```
