# TODO

## Snakemake Migration
- [x] Replace bash/SLURM orchestration with Snakemake workflow
- [x] Define rules for MSA, Boltz, ESM, ES stages
- [x] Use Snakemake's cluster execution and dependency management
- [x] Add dual-mode container support (Singularity / legacy fallback)
- [x] Fix ESM integration (shell variable resolution, model paths, logging)
- Branch: `snakemake`

## Containerization
- [x] Dual-mode Snakemake rules (`singularity exec` or `module load`)
- [x] `container_cmd()` helper and config template section
- [x] Container design doc (`docs/CONTAINERS.md`)
- [ ] Build Dockerfiles for each stage (ColabFold, Boltz, ESM, PDAnalysis)
- [ ] Test container images on Kempner cluster
- [ ] Publish to GHCR and add pull instructions
- [ ] CI/CD for automated image builds

## UX for Non-Expert Users
- [ ] Config validation script (pre-flight: check paths exist, envs have packages, model weights present)
- [ ] CLI wrapper (`protforge run`, `protforge status`, `protforge init`)
- [ ] `protforge init` — interactive config.yaml generator from template
- [ ] Progress dashboard (per-stage: N/total sequences done, failures)
- [ ] Complete README.md (installation, usage, examples)
- [ ] Getting-started tutorial (CSV → embeddings in 4 commands)
- [ ] Example configs: small test (5 seqs) + production

## Portability
- [ ] Remove hardcoded cluster paths from ESM library (patch or upstream PR)
- [ ] Make `module load` calls conditional on cluster detection
- [ ] Environment variable-based model path resolution (not hardcoded in esm package)
- [ ] Document setup for non-Kempner SLURM clusters

## Testing Framework
- [ ] pytest for Python utilities (utils/utils.py, etc.)
- [ ] bats for bash scripts (run.sh, generate_data.sh, checker scripts)
- [ ] CI integration

## Other Improvements
- [ ] Auto-retry with resource bumping (Snakemake `attempt` for OOM/timeout)
- [ ] Multi-chain support in data generation
- [ ] Unified logging across stages
- [ ] Output summary report (sequences processed, wall time, failures)
