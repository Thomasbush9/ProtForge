# ESM-C parameter tuning

ESM-C (ESM Cambrian, EvolutionaryScale) per-residue embeddings/logits. ProtForge
runs it via `containers/run_batch_esmc.py` (`workflow/rules/esmc.smk`). ESM-C
needs only the sequence — no MSA — so it runs in parallel with MSA straight from
`input.fasta_dir`. The main "tuning" choice is **which model size**; the rest are
throughput knobs.

## Model size — the main quality lever (`esmc.models`)

`models` is any subset of `{6B, 600M, 300M}`. Each size is its own stage with its
own jobs, per-size SLURM resources (`slurm.resources.esmc_<size>`), and a
`.esmc_<size>_complete` sentinel.

| Size | Params | Layers | Embed dim | Quality (official framing) | Speed / VRAM | When to use |
|---|---|---|---|---|---|---|
| **300M** | ~300M | 30 | 960 | ≈ ESM2-650M | Fastest, lowest VRAM (fits ~16 GB) | Large-scale screening / high-throughput embedding |
| **600M** | ~600M | 36 | 1152 | ≈ ESM2-3B | Runs on a 16 GB T4 (per BioLM hosting) | Balanced default for most embedding work |
| **6B** | ~6B | 80 | 2560 | New SOTA, beats all prior PLMs | Heaviest; needs a large-memory GPU (A100/H100 40–80 GB) | Best embeddings, when quality > throughput/cost |

ProtForge default: `models: [600M]`. Embedding quality scales with size; choose
300M for screening, 600M as the middle ground, 6B for the highest-quality
embeddings. Output is per-residue embeddings (+ logits) -> `sequences/{seq}/esmc/{size}/`.

EvolutionaryScale does **not** publish exact per-size VRAM; the 16 GB figures
above are third-party (BioLM) / reasoned, not official. Set
`slurm.resources.esmc_6B` higher than the shared `esmc` default — 6B needs far
more GPU/host memory than 300M.

## Throughput knobs (`esmc:` block)

| Param | What it does | Effect | Default | When to change |
|---|---|---|---|---|
| `max_files_per_job` | Sequences per chunk (scheduling granularity). | Throughput only. | 25 | Tune via estimator. |
| `chunks_per_group` | Chunks served per GPU job; model loaded ONCE per group, then all sequences are length-bucketed into padded micro-batches. | Higher amortizes the weight load → better GPU utilization. Scale SLURM runtime accordingly. | 1 | Raise (e.g. 4) for many short sequences to amortize load. |
| `token_budget` | Padded-token cap per micro-batch (rows × max_len). | Caps peak VRAM per batch. | per size: 6B 8000, 600M 32000, 300M 49152 | Lower if a chunk OOMs. |

## SAE (sparse autoencoder) — `esmc.sae`

SAEs decompose ESM-C's dense activations into a large dictionary of sparse,
interpretable biological features (k=64 features fire per residue). Recomputed
from the sequence, so SAE can run standalone on a prior run's YAMLs. Output ->
`sequences/{seq}/sae/{size}/{sae_type}/`.

| Param | What it does | Default |
|---|---|---|
| `enabled` | Turn SAE extraction on. | false |
| `sae_type` | `all-layers` (residual-stream hidden states, codebook 16,384, one SAE per layer — "global understanding") or `mlp` (per-layer MLP output, codebook 131,072 — finer/specialized features). | all-layers |
| `layers` | `all` (every trained layer) or a comma list (e.g. `"18,36"`). | all |
| `sizes` | Subset of `models` to run SAE on. | all of `models` |
| `max_files_per_job`, `chunks_per_group` | As above (one SAE load per group). | 25, 1 |

Note: published open SAE weights are **6B-centric** (e.g. `biohub/ESMC-6B-sae-*`).
Confirm smaller-size SAE shards exist on HuggingFace before assuming 300M/600M SAEs.
`layers: all` is heaviest (one SAE per layer); restrict to specific layers to cut cost.

## Recommended recipes

- **Fast screen:** `models: [300M]`, default `chunks_per_group`, SAE off. Bump
  `chunks_per_group` for libraries of short sequences.
- **Balanced (default):** `models: [600M]`, SAE off.
- **Highest-quality embeddings:** `models: [6B]`; raise `slurm.resources.esmc_6B`
  memory; lower `token_budget` if it OOMs.
- **Interpretability:** `sae.enabled: true`, `sae_type: all-layers`,
  `layers: all` (or a few layers to save time), typically on 6B.

The resource estimator covers shared `esmc` but **not** per-size `esmc_6B/600M/300M`
or `esmc_sae` — set `slurm.resources.esmc_<size>` / `esmc_sae` by hand.

## Sources

- ESM-C announcement (sizes, layers, embed dims, benchmarks):
  https://www.evolutionaryscale.ai/blog/esm-cambrian
- ESM repo (outputs, 6B access): https://github.com/evolutionaryscale/esm
- SAE model cards (all-layers vs mlp, codebooks, k=64):
  https://huggingface.co/biohub/ESMC-6B-sae-k64-codebook16384 ,
  https://huggingface.co/biohub/ESMC-6B-sae-mlp-k64-codebook131072 ,
  https://huggingface.co/biohub/ESMC-6B
- 600M on 16 GB T4 (third-party VRAM data point): https://biolm.ai/models/esmc-600m/
