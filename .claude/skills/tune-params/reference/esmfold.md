# ESMFold2 parameter tuning

ESMFold2 (Biohub "fast" variant) single-sequence structure prediction. ProtForge
runs it via `containers/run_batch_esmfold.py` (`workflow/rules/esmfold.smk`),
passing `--num-loops`, `--num-sampling-steps`, and `--seed`. Like ESM-C it needs
**no MSA** and runs in parallel with MSA straight from `input.fasta_dir`. Outputs
(structure.cif, plddt.npy, metrics.pt) -> `sequences/{seq}/esmfold/fast/`.

ESMFold2 is diffusion-based (ESM-C embeddings + a diffusion structure module),
distinct from the original recycling-based ESMFold. It folds one complex per call
(no real batching), so `chunks_per_group` only amortizes the model load.

## ProtForge-exposed params (`esmfold:` block)

| Param | What it does | Quality / speed / VRAM | ProtForge default | When to change |
|---|---|---|---|---|
| `num_loops` | Refinement cycles (the recycling analogue). More = more refinement. | More = better, ~linear runtime. | **3** | Lower (1–2) for fast screens; 3 is the documented example value. |
| `num_sampling_steps` | Diffusion denoising iterations. More steps = finer denoising. | More = higher quality, slower; the main speed dial. | **50** | Lower (e.g. 32, the Biohub "Fast" example) for speed; raise for quality. |
| `seed` | RNG seed for the diffusion sampler. | Reproducibility only. | 0 | Change to draw a different sample. |
| `compile` | Opt-in `torch.compile` of the resident model (dynamic shapes; eager fallback). | Faster steady state after warmup; only worth it when a group serves many folds. | false | Enable when `chunks_per_group` is high. |
| `chunks_per_group` | Chunks served per GPU job; model loaded ONCE per group, folds stay serial. | Amortizes model load across serial folds. | 1 | Raise for many sequences to amortize load. |
| `max_files_per_job` | Sequences per chunk. | Throughput only. | 25 | Tune via estimator. |

VRAM scales with sequence length (attention is roughly O(L²) in length for this
model family). The model owns its bf16 precision.

## Recommended recipes

- **Fast screen:** `num_loops: 2`, `num_sampling_steps: 32`. Raise
  `chunks_per_group` (and optionally `compile: true`) for large short-sequence
  libraries.
- **Balanced (default):** `num_loops: 3`, `num_sampling_steps: 50`.
- **Higher quality:** raise `num_sampling_steps` (more steps), keep `num_loops: 3`.

Raising `num_loops` / `num_sampling_steps` raises per-job runtime — re-run the
resource estimator after changing them.

## Caveats (honest gaps)

The Biohub "ESMFold2-Fast" variant is sparsely documented. The values above
(`num_loops=3`, `num_sampling_steps` 32–50) are the **documented example values**
from the reference code / model cards — there is **no published "true default"**
when the argument is omitted, and no published speed/VRAM benchmarks or
peer-reviewed accuracy numbers for this variant. ProtForge's `esmfold:` defaults
(num_loops 3, num_sampling_steps 50) are the source of truth here.

For context, the *original* ESMFold used `--num-recycles` default 4 and
`--chunk-size` (128/64/32) to cut O(L²)→O(L) attention memory; ESMFold2 replaces
recycling with `num_loops` + diffusion `num_sampling_steps`.

## Sources

- ESMFold2 reference code (num_loops, num_sampling_steps, num_diffusion_samples):
  https://github.com/atong01/esmfold2
- ESMFold2 model cards (example configs, 32/50 steps):
  https://huggingface.co/Synthyra/ESMFold2 ,
  https://huggingface.co/Synthyra/ESMFold2-Experimental-Fast-Cutoff2025
- Original ESMFold (single-sequence, recycling default 4, chunk-size):
  https://github.com/facebookresearch/esm ,
  https://deepwiki.com/facebookresearch/esm/2.3-esmfold
- ESMFold paper (single-sequence, speed): Lin et al. 2023, Science 379:1123–1130
  https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf
