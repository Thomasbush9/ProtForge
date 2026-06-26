# Boltz parameter tuning

Boltz-2 structure (and affinity) prediction via `boltz predict`. ProtForge's
rule (`workflow/rules/boltz.smk`) passes `--recycling_steps` and
`--diffusion_samples` explicitly, runs `--devices 1 --accelerator gpu`, and lets
any extra flags through `boltz.advanced` (see the flag table). Everything else
falls back to Boltz's own defaults.

## ProtForge-exposed params (`boltz:` block in config)

| Param | What it does | Quality / speed / VRAM | ProtForge default | When to change |
|---|---|---|---|---|
| `recycling_steps` | Trunk refinement iterations before diffusion. More passes refine the representation. | More = better accuracy, ~linear runtime cost, diminishing returns past ~10. Little VRAM impact. | **10** | Lower to 3 (Boltz's own default) for fast screens; 10 is the AlphaFold3-equivalent quality setting. |
| `diffusion_samples` | Independent structures sampled per input; rank/keep the best by confidence. | More = more diversity + better best-of-N, ~linear runtime. Peak VRAM is governed by Boltz's `--max_parallel_samples` (default 5), not the sample count itself. | **25** | Lower to 1–5 for fast screens; 25 is the AlphaFold3-equivalent setting. Raise when you want an ensemble / best-of-N. |
| `samples_to_save` | Top-N samples kept on disk (`model_0` = best confidence). int or `"all"`. | Disk only — no compute effect. | **1** | Raise to keep alternative conformations; `"all"` to inspect the full ensemble. |
| `num_runs` | Independent Boltz runs per sequence (separate output dirs `run_0/`, `run_1/`…). | Multiplies total runtime by `num_runs`. | **1** | Raise for run-to-run reproducibility checks. Distinct from `diffusion_samples` (which varies within one run). |
| `max_seq_len` | Drop sequences longer than N residues at chunk time (skipped ones logged to `boltz_chunks/skipped_sequences.tsv`). | Caps per-job VRAM/runtime by excluding long sequences. | unset | Set when long sequences OOM on the chosen GPU. |
| `max_files_per_job` | Sequences per chunk (scheduling granularity). | Throughput/scheduling only, not quality. | 25 | Tune with the resource estimator, not by hand. |

## `boltz.advanced` escape-hatch flags

`boltz.advanced` is deep-merged into a `boltz predict` flag string. Useful ones
(Boltz defaults shown; all OFF/unset unless you add them):

| Flag | What it does | Default |
|---|---|---|
| `sampling_steps` | Diffusion denoising steps. Lower = faster, slight quality loss. | 200 |
| `step_scale` | Sampling "temperature"; lower → more sample diversity. Keep 1–2. | 1.5 (Boltz-2) |
| `use_potentials` | Inference-time physics steering → more physically plausible poses. Adds runtime. | off |
| `subsample_msa` / `num_subsampled_msa` | Subsample deep MSAs to cut compute/VRAM (or induce diversity). | off / 1024 |
| `no_kernels` | Disable `trifast` CUDA kernels — compatibility fallback, slower. | off |
| `affinity_mw_correction` | Molecular-weight correction on the affinity head. | off |

## Recommended recipes

- **Fast screen / large library:** `recycling_steps: 3`, `diffusion_samples: 1`,
  `samples_to_save: 1`. Optionally `max_seq_len` to drop the long tail. Fastest,
  fine for ranking many candidates.
- **High-quality final:** `recycling_steps: 10`, `diffusion_samples: 25`,
  `samples_to_save: 1` (the ProtForge default — AlphaFold3-equivalent). Add
  `advanced: {use_potentials: true}` when pose realism matters. This "will take
  significantly longer" (Boltz docs).
- **Conformational ensemble:** raise `diffusion_samples`, set
  `samples_to_save: "all"`, lower `step_scale` toward 1.0 via `advanced`.

Note: raising `recycling_steps` / `diffusion_samples` raises per-job runtime and
VRAM, so re-run the resource estimator (`webapp/estimate_cli.py`) after changing
them — the calibrated runtime fits assume the defaults.

## Sources

- Boltz `boltz predict` options + defaults (recycling_steps, diffusion_samples,
  sampling_steps=200, step_scale, use_potentials, subsample_msa, no_kernels):
  https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md
- Boltz-2 paper (steering potentials, affinity): https://jeremywohlwend.com/assets/boltz2.pdf
- Boltz repo: https://github.com/jwohlwend/boltz
- VRAM figures are third-party (NVIDIA NIM ≥48 GB; ChimeraX ~1300 res on 24 GB),
  not published by Boltz — treat as indicative:
  https://docs.nvidia.com/nim/bionemo/boltz2/latest/support-matrix.html ,
  https://www.rbvi.ucsf.edu/chimerax/data/boltz-apr2025/boltz_help.html

ProtForge's own defaults (the `boltz:` block above) are the source of truth for
this pipeline; external numbers explain what each knob does.
