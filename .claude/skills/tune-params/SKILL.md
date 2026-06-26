---
name: tune-params
description: >-
  Advise on which model parameters to set for a ProtForge stage and recommend
  good values for a goal. Use when the user asks which parameters to use, how to
  tune boltz / esmc / esmfold / openfold / msa, the best settings for fast vs
  accurate runs, about recycling steps / diffusion samples / sampling steps /
  model size, or the speed-vs-quality (and VRAM/runtime) tradeoff for a stage.
  Advisory only — it recommends values; apply them via the config or the
  run-pipeline skill.
---

# Tune ProtForge stage parameters

Help a researcher pick the right tunable parameters for one pipeline stage and a
goal, then point them to apply the values. This skill is **advisory** — it does
not run anything. The durable knowledge lives in the per-stage reference docs;
this file just routes to them.

## Procedure

### 1. Pin down stage + goal

Ask only what you can't infer:

- **Which stage** — `msa`, `boltz`, `esmc`, `esmfold`, or `openfold`?
- **Goal** — fast screen vs high-quality final (the main axis)?
- **Dataset size** — how many sequences, and sequence lengths (long sequences
  drive VRAM/OOM)?
- **GPU budget** — which GPU/partition, or `auto`? Bigger models/more
  samples/recycling need more VRAM and time.

### 2. Read the matching reference doc

Read `reference/<stage>.md` for the stage in question (don't answer from memory —
the docs carry the ProtForge defaults and the external tool guidance):

- `reference/boltz.md` — recycling_steps, diffusion_samples, samples_to_save,
  num_runs, max_seq_len, `boltz.advanced` flags.
- `reference/esmc.md` — model size (300M/600M/6B), chunks_per_group, token_budget,
  and the SAE block.
- `reference/esmfold.md` — num_loops, num_sampling_steps, compile, chunks_per_group.
- `reference/openfold.md` — num_diffusion_samples, num_model_seeds, recycling_iters,
  gpus_per_job, samples_to_save.
- `reference/msa.md` — why MSA is a throughput stage, not a per-run quality dial.

### 3. Recommend concrete values

Give a specific set of `<stage>:` config values matched to the goal, using the
"recommended recipes" in the doc. Quote the ProtForge default and say what each
change costs (quality/speed/VRAM). When the docs flag an honest gap (e.g. no
published ESMFold2-Fast default, OpenFold recycling default), say so rather than
inventing a number — the `config.template.yaml` defaults are the source of truth
for ProtForge.

### 4. Tell them how to apply it

The values go in the `<stage>:` block of the run config. Point the user to:

- edit the config directly (`config.template.yaml` documents every block), or
- the **run-pipeline** skill, which preps the config and launches the run.

### 5. Flag the resource-estimation interaction

More samples / recycling / a bigger model size => more runtime and VRAM => the
calibrated resource fits no longer hold. Tell the user to **re-run the estimator**
after changing these (`python -m webapp.estimate_cli --config <cfg> --apply`).
Note the estimator does **not** size `openfold`, per-size `esmc_<size>`, or `sae`
— those `slurm.resources.*` entries are set by hand (see the reference docs).

## Notes

- Quality/speed knobs per stage: boltz `recycling_steps`+`diffusion_samples`;
  esmc `models` (size); esmfold `num_loops`+`num_sampling_steps`; openfold
  `num_model_seeds`+`num_diffusion_samples`(+`recycling_iters`); msa has none
  worth tuning per-run.
- ESM-C and ESMFold2 are single-sequence (no MSA); Boltz and OpenFold consume the
  MSA. Don't recommend MSA tuning for the ESM stages.
