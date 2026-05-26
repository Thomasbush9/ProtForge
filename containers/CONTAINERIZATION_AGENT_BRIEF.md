# Containerization Agent Brief

This is a handoff for the next agent working on ProtForge containerization. The current goal is Kempner-first Singularity execution: a user should be able to clone the repo, install the small host-side webapp/Snakemake environment, build or pull the GPU SIF, bind the external MSA/Boltz databases, launch Streamlit, and submit predictions with stable in-container paths.

## Status snapshot — 2026-05-26

| # | Title                                | Status         | Notes |
|---|--------------------------------------|----------------|-------|
| 1 | Build context leakage                | **done***      | `.singularityignore` is the gate. Old May-16 SIF leaked `.git`/`sifs/`/`sing_cache/` only because it predates the ignore file (May 19). Smoke `[3/7]` will pass after rebuild. Allowlist `%files` deferred (cost > value for current iteration speed). |
| 2 | Mutable supply chain                 | **deferred**   | User opted out (2026-05-26): not pinning HF revision, CUDA digest, mmseqs sha for now. Re-open when reproducibility becomes a higher-priority constraint. |
| 3 | Snakemake container contract drift   | **done**       | `containers.gpu` fallback; per-stage keys remain as overrides. |
| 4 | In-container script path mismatch    | **done**       | Rules call `/opt/protforge/slurm_scripts/run_*.py`. Smoke step 3 guards against regression. |
| 5 | Bind mounts too broad / writable     | **done**       | `_parse_bind` shell-quotes + validates mode; template defaults DBs to `:ro` at their actual host paths. |
| 6 | Baked weights bypass                 | **done**       | Rules inject `--env HF_HOME=` only when `cache_dir` set; otherwise `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Template `esm.cache_dir` / `esmfold.cache_dir` default empty. |
| 7 | Host webapp env spec                 | **done**       | `requirements-host.txt` + `pyproject.toml` `host` extras. |
| 8 | E2E test                             | **open**       | Stage-1 E2E (input → MSA → Boltz → ESMFold → ESM) agreed as the right target; not scripted yet. |

`*` Pending rebuild verification on cluster.

### 2026-05-26 changes

- **ESMFold weight prefetch fixed.** `.def` now uses `huggingface_hub.snapshot_download(repo_id='facebook/esmfold_v1', allow_patterns=['pytorch_model.bin','*.json','*.txt','*.model'])` instead of `EsmForProteinFolding.from_pretrained(..., use_safetensors=True)`. The previous form left the snapshot dir with only tokenizer/config files (smoke step 6 OSError). Resolves the blocker on the next rebuild.
- **Base image switched: `nvidia/cuda:12.4.1-runtime-ubuntu22.04` → `ubuntu:22.04`.** Rationale: PyTorch's `cu124` wheels bundle libcudart/libcublas/libcudnn under `torch/lib/`, so the CUDA layer was duplicated weight (~2 GB). The NVIDIA driver libs come from `singularity exec --nv` at runtime; Triton ships its own ptxas. Host `cuda/cudnn` modules remain in the bash/conda fallback (`workflow/rules/*.smk` `else` branches), not in container mode. Smoke step 2 (`torch=…cu124, cuda=True`) is still expected to pass — the test target is unchanged.
- **HF_TOKEN passthrough for builds.** First rebuild on Kempner hit HF Hub's per-IP anonymous rate limit on the ESM-C prefetch (`LocalEntryNotFoundError` from `EvolutionaryScale/esmc-600m-2024-12`). `build.sh` now accepts `--hf-token` or reads `$HF_TOKEN`; the token is staged to a mode-600 mktemp file under `SINGULARITY_TMPDIR` and bind-mounted at `/run/secrets/hf_token:ro`. `%post` reads it with `set +x` so it doesn't appear in the build log, then `unset HF_TOKEN`s before the rest of the build runs. The runtime `%environment` does not set HF_TOKEN, so it never reaches the image surface.

### Newly found in 2026-05-19 review (now addressed)

- **Path contract drift.** First-pass `bind_paths` redirected to `/data/colabfold_db` / `/data/boltz_db` (matching the def file's `%environment` comment), but stage configs (`msa.mmseq2_db`, `boltz.cache_dir`, ...) still passed host paths. Inside the container only `/data/*` would be visible, so the rule would call `colabfold_search` with a path that didn't exist. **Resolved** by switching the template default to "Design A": bind each DB at the SAME path inside the container, read-only. Stage configs work unchanged in container mode. The alternate `/data/*` convention is documented as an opt-in; the def file's existing comments remain valid for users who flip the contract.
- **Shell-quoting holes.** Initial pass only quoted bind args. **Resolved** by also `shlex.quote`-ing the SIF path in `container_cmd()`, shell-quoting `cache_dir` inside `container_cache_env` / `cache_dir_arg`, and rejecting `containers.runtime` outside `{auto,singularity,apptainer}` (it's interpolated into a bash command line, so cannot be free-form).

The remainder of this document is the original brief, kept verbatim for context. Items above supersede the recommendations below where they overlap.

---


## Current Architecture

- `containers/protforge-gpu.def` builds one GPU SIF for MSA, Boltz, ESM-C, and ESMFold.
- Model weights for ESM-C and ESMFold are baked under `/opt/weights/hf`.
- Large databases are intentionally not baked: ColabFold/MMseqs should bind to `/data/colabfold_db`, and Boltz checkpoint/cache should bind to `/data/boltz_db`.
- `Snakefile:container_cmd()` emits `singularity exec --nv --cleanenv ...`.
- The Streamlit app remains host-side and launches host `snakemake`; the SIF is only used by Snakemake stage rules.

## Design Principles

- Treat the SIF as immutable application/runtime state. User data, outputs, scratch, MSA databases, Boltz checkpoints, and optional override caches should be external bind mounts.
- Keep build context minimal. The def file currently copies `.` into `/opt/protforge`, so `.singularityignore` is a security control, not just a convenience.
- Prefer reproducible pulls over mutable source builds on the cluster. The long-term path should be CI or a trusted builder publishing an OCI image/SIF, with Kempner users running `singularity pull`.
- Run with least privilege and least filesystem exposure: `--cleanenv`, read-only database binds, explicit writable output/scratch binds, and no broad home/lab-tree binds once the exact path contract is known.
- Prefer node-local scratch for temp-heavy tools. Bind `${SLURM_TMPDIR:-/tmp}` to `/tmp` and set `TMPDIR=/tmp`; keep Triton/HF temp caches under that or a per-user writable cache.

## Issues To Fix

### 1. Build Context Leakage

Problem: `%files . /opt/protforge` can bake `.git`, `.venv`, `.sessions`, `config.yaml`, logs, test outputs, secrets, or huge local outputs into the SIF. A root `.singularityignore` has been added, but it is only the first guard.

Recommended action:

- Keep `.singularityignore` in sync with `.gitignore` plus Singularity-specific artifacts such as `*.sif`, `sing_cache/`, `sing_tmp/`, and smoke/e2e outputs.
- Prefer replacing `%files . /opt/protforge` with an explicit allowlist if the runtime file set is stable.
- Add a smoke assertion that the built image does not contain `.git`, `.venv`, `.sessions`, `config.yaml`, or `scripts/uniprot_fetch/.venv`.

### 2. Mutable Supply Chain

Problem: the image depends on mutable upstreams: a floating CUDA tag, MMseqs `latest`, unpinned PyPI packages, ColabFold from GitHub default branch, and HF models without explicit revisions.

Recommended action:

- Pin the base image by digest, not only tag.
- Pin Python packages in a dedicated container requirements/constraints file. For high assurance, use hashes where practical.
- Pin `colabfold` to a commit SHA.
- Replace `https://mmseqs.com/latest/...` with a versioned tarball and verify a SHA256 checksum before extracting.
- Pin HuggingFace downloads with explicit `revision=` values and document the expected snapshot IDs.
- Record all resolved versions in image labels or a build manifest under `/opt/protforge/container-build-manifest.txt`.

### 3. Snakemake Container Contract Drift

Problem: docs say to set `containers.gpu`, but `Snakefile:container_cmd(stage)` only checks per-stage keys like `containers.boltz`, `containers.esm`, and `containers.esmfold`. A fresh config following the README may silently run legacy host environments.

Recommended action:

- Either implement `containers.gpu` as a fallback for all GPU stages, or update all docs/templates/webapp UI to write the same SIF into each per-stage key.
- Add a dry-run test proving every enabled GPU rule expands to a non-empty `singularity exec` command.
- Update `config.template.yaml` for the single-SIF design and include read-only DB binds.

### 4. In-Container Script Path Mismatch

Problem: the image copies the repo to `/opt/protforge`, but the ESM and ESMFold rules call `/opt/protforge/run_esm.py` and `/opt/protforge/run_esmfold.py`. Those files live under `slurm_scripts/`.

Recommended action:

- Change containerized rules to call `/opt/protforge/slurm_scripts/run_esm.py` and `/opt/protforge/slurm_scripts/run_esmfold.py`.
- Or create explicit symlinks in `%post`, but prefer fixing the rule paths so the code layout remains honest.

### 5. Bind Mounts Are Too Broad And Writable

Problem: examples bind full `/n/holylfs06` and `/n/home06` trees and database binds are not marked read-only. This increases blast radius and makes the path contract less clear.

Recommended action:

- Bind databases as read-only: `host_colabfold_db:/data/colabfold_db:ro` and `host_boltz_db:/data/boltz_db:ro`.
- Bind only the repo/workdir, output dir, and required input dirs as writable.
- Consider `--no-home` or `--containall` once all required paths are explicit.
- Validate `containers.bind_paths` entries and shell-quote generated `-B` arguments. Avoid raw string interpolation from YAML/UI into shell commands.

### 6. Baked Weights Can Be Accidentally Bypassed

Problem: `%environment` defaults `HF_HOME=/opt/weights/hf`, but some rules pass `--env HF_HOME={params.cache_dir}` or `TORCH_HOME={params.cache_dir}`. If the config still contains old host cache paths, or blank values, the baked weights may not be used.

Recommended action:

- In container mode, default ESM/ESMFold cache config to `/opt/weights/hf`.
- Only pass `HF_HOME`/`TORCH_HOME` from the rule when the user explicitly chooses an override.
- For baked-cache runs, pass `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` where model loading supports it.
- Put writable per-user caches under `PROTFORGE_HOME` or another explicit bind, not inside the SIF.

### 7. Host Webapp Environment Is Not Specified

Problem: the container does not package the Streamlit app or Snakemake launcher. That is acceptable for Kempner, but the host environment must be documented and reproducible.

Recommended action:

- Add a host-side environment file for launching Streamlit and Snakemake (`environment.yml`, `requirements-webapp.txt`, or equivalent).
- Include `streamlit`, `streamlit-autorefresh`, `snakemake`, the SLURM profile dependencies/plugins, and the project package itself.
- Document that this host environment is separate from the GPU SIF.

### 8. Testing Does Not Cover The Real Runtime Path Yet

Problem: `containers/test/smoke.sh` validates GPU visibility, imports, baked weights, and ESMFold, but it does not exercise MSA, Boltz, DB bind mounts, Snakemake, or the webapp.

Recommended action:

- Script the Stage-1 E2E described in `containers/TESTING.md`.
- Include read-only DB binds and a one-protein FASTA.
- Add a Snakemake dry-run check for the webapp-generated/session config.
- Add a minimal Streamlit launch check that verifies a session config can point to the SIF and produce the expected Snakemake command.

## Suggested Implementation Order

1. Fix build-context safety: verify `.singularityignore`, then add image-content smoke checks.
2. Fix rule/doc contract: `containers.gpu` fallback or per-stage config only, plus the ESM/ESMFold script paths.
3. Tighten runtime binds: read-only DB binds, narrower writable mounts, quoted bind generation.
4. Add a host webapp/Snakemake environment spec.
5. Pin supply chain inputs and add build manifest labels.
6. Script E2E and webapp dry-run tests.

## Acceptance Checklist

- `singularity exec protforge-gpu.sif test ! -d /opt/protforge/.git` passes.
- `singularity exec protforge-gpu.sif test ! -d /opt/protforge/scripts/uniprot_fetch/.venv` passes.
- A config following the docs causes all enabled GPU rules to use the SIF.
- DB binds are documented and tested with `:ro`.
- ESM and ESMFold rules execute the correct scripts inside `/opt/protforge`.
- The image can load baked ESM-C and ESMFold weights without network access.
- A one-protein Snakemake E2E passes with MSA, Boltz, ESM, and ESMFold enabled.
- Streamlit can create or load a session config that points to the SIF and launches the same Snakemake path tested above.
