# TODO

## Snakemake Migration
- [x] Replace bash/SLURM orchestration with Snakemake workflow
- [x] Define rules for MSA, Boltz, ESM, ES stages
- [x] Use Snakemake's cluster execution and dependency management
- [x] Add dual-mode container support (Singularity / legacy fallback)
- Branch: `snakemake`

## Containerization
- [x] Dual-mode Snakemake rules (`singularity exec` or `module load`)
- [x] `container_cmd()` helper and config template section
- [x] Container design doc (`docs/CONTAINERS.md`)
- [ ] Build Dockerfiles for each stage (ColabFold, Boltz, ESM, PDAnalysis)
- [ ] Test container images on Kempner cluster
- [ ] Publish to GHCR and add pull instructions
- [ ] CI/CD for automated image builds

## Testing Framework
- [ ] pytest for Python utilities (utils/utils.py, utils/generate_data.py, etc.)
- [ ] bats for bash scripts (run.sh, generate_data.sh, checker scripts)
- [ ] CI integration

## Other Improvements
- [ ] Dry-run flag (`--dry-run`) for run.sh and standalone scripts
- [ ] Progress dashboard (aggregate processed_paths.txt / total_paths.txt)
- [ ] Multi-chain support in data generation
- [ ] Config validation script (check paths exist, params in range)
- [ ] Unified logging across stages
