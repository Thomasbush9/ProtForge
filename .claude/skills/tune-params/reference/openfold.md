# OpenFold3 parameter tuning

OpenFold3 (AlphaFold3-style diffusion structure prediction). ProtForge runs it
via `run_openfold predict` (`workflow/rules/openfold.smk`), passing
`--num-diffusion-samples`, `--num-model-seeds`, and a generated `runner.yaml`.
OpenFold3 consumes the MSA-stage a3m (copied to `uniref90_hits.a3m`,
`--use-msa-server=False`), so it runs off MSA in parallel with Boltz. Outputs
(top-N samples' .cif + confidence JSONs) -> `sequences/{seq}/openfold/`.

Each chunk = one query JSON = one SLURM job; a job folds all its queries across
`gpus_per_job` GPUs on one node via PyTorch Lightning. **num jobs == num JSONs.**

## ProtForge-exposed params (`openfold:` block)

| Param | What it does | Quality / speed / VRAM | ProtForge default | When to change |
|---|---|---|---|---|
| `num_diffusion_samples` | Structures sampled per seed from the diffusion module. | More = more diversity + better best-of-N, ~linear runtime. VRAM depends on mode (batched vs sequential low-mem). | **5** | Lower to 2–3 for fast screens; OpenFold3 advises starting with 1 seed × 2–3 samples. |
| `num_model_seeds` | Independent random seeds; each is a full trunk+diffusion pass. **Dominant diversity lever** (more so than samples). | More = better hard-case coverage (e.g. antibodies), ~linear runtime; seeds run sequentially so no extra peak VRAM. | **1** | Raise (up to ~5) for difficult targets where diversity helps. |
| `seeds` | Explicit seed integers (overrides `num_model_seeds` count). | Reproducibility. | unset | Pin for reproducible runs. |
| `recycling_iters` | Trunk recycling iterations (refines embeddings before diffusion). | More = better accuracy, diminishing returns, ~linear cost; biggest trunk-side speed knob. | unset (uses the predict preset default) | Set 3–10 to trade speed for accuracy. |
| `samples_to_save` | Top-N samples (by ranking score) kept. int or `"all"`. | Disk only. | **1** | Raise to keep alternatives. |
| `gpus_per_job` | GPUs per job (1–4) → `pl_trainer_args.devices` + `--gpus-per-node`. | Splits a job's queries across GPUs → faster wallclock. | 1 | Raise when a chunk has many/large queries and a multi-GPU node is available. |
| `num_workers` | Dataloader workers. | Throughput only. | 10 | Tune to CPU allocation. |
| `num_batches` / `max_files_per_job` | Batch control — `num_batches` wins when set; else queries per JSON. | Scheduling only. | (num_batches unset) / 25 | Control job count. |
| `structure_format` | `cif` / `pdb` / `cif.gz`. | Output format only. | cif | Pick downstream-required format. |
| `write_full_confidence` | Write full per-residue confidence JSONs. | Disk only. | true | Set false to cut disk on big multi-seed jobs. |
| `advanced` | Raw dict deep-merged into `runner.yaml` (escape hatch, e.g. `low_mem` preset). | Varies. | unset | For memory presets / unexposed knobs. |

**Total structures = num_model_seeds × num_diffusion_samples × queries.** Default
1 × 5 = 5 structures per sequence; top-N kept by `samples_to_save`.

VRAM/compute scale steeply (≈O(N²) pairwise representations) with token count.
Reference AF3 hardware is a single 80 GB A100/H100. To fit larger complexes on
smaller cards: lower `num_diffusion_samples`, use a `low_mem` preset via
`advanced`, and set `write_full_confidence: false`.

## Recommended recipes

- **Fast screen:** `num_model_seeds: 1`, `num_diffusion_samples: 2`,
  `samples_to_save: 1`, `recycling_iters: 3` (if set). Cheapest first pass.
- **Balanced (default):** `num_model_seeds: 1`, `num_diffusion_samples: 5`.
- **High quality / hard targets:** raise `num_model_seeds` to 3–5 (the strongest
  diversity lever), keep `num_diffusion_samples: 5`, `recycling_iters: 10`.
- **Large complex on a small GPU:** lower `num_diffusion_samples`, add a low-mem
  preset via `advanced`, `write_full_confidence: false`, raise `gpus_per_job`.

The resource estimator does **not** size OpenFold — set `slurm.resources.openfold`
by hand, and re-estimate runtime/memory whenever you raise seeds/samples/recycling.

## Caveats (honest gaps)

OpenFold3's own docs do **not** publish a `recycling_iters` default (it inherits
the AF3 architecture); the AF3 paper uses `Ncycle=4` as the algorithmic default
but ran its benchmarks at 10 recycles. `num_diffusion_samples` default 5 and
`num_model_seeds` default 1 are from OpenFold3 docs (AF3's evaluation used 5
seeds, but its released-code default is also 1). Leave `recycling_iters` unset to
use OpenFold3's predict-preset default unless you have a reason to override.

## Sources

- OpenFold3 inference docs (num_diffusion_samples=5, num_model_seeds=1, seeds,
  devices, low_mem, totals): https://openfold-3.readthedocs.io/en/latest/inference.html
- OpenFold3 repo: https://github.com/aqlaboratory/openfold-3
- AlphaFold3 output/perf docs (5 samples/seed, ranking, scaling):
  https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md ,
  https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md
- AlphaFold3 paper suppl. (Ncycle=4 default, 10 recycles for benchmarks, 5 seeds):
  Abramson et al. 2024, Nature
- OpenFold3 practical guide (1 seed × 2–3 samples to start, low-mem):
  https://binaryverseai.com/openfold3-nim-demo-guide-use-protein-structure/
