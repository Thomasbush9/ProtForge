# MSA parameter tuning

MSA generation via ColabFold `colabfold_search` over the shared MMseqs2/ColabFold
databases (`workflow/rules/msa.smk`). ProtForge invokes it as
`colabfold_search <fasta> <db> <out> --thread 4 --gpu 1`.

## Honest framing: MSA is a resource stage, not a quality dial

`colabfold_search` is best treated as a **throughput/resource** stage, not a
per-run quality-tuning stage. Its defining cost is the homology search over very
large memory-mapped databases (~256 GB host RAM for the mmap'd ColabFold DB),
which the estimator requests as `--mem`. The genuine quality levers are coarse
and recipe-level (which databases), and ColabFold already ships tuned defaults
you generally leave alone.

Crucially: **only Boltz and OpenFold consume the MSA.** ESMFold2 and ESM-C are
single-sequence (no MSA) — `colabfold_search` settings are irrelevant to them,
and running MSA buys those stages nothing.

## ProtForge-exposed params (`msa:` block)

| Param | What it does | Effect | ProtForge default | When to change |
|---|---|---|---|---|
| `max_files_per_job` | Sequences per chunk (scheduling granularity). | Throughput/scheduling only — **not** quality. | 25 | Tune via the resource estimator. |
| `array_max_concurrency` | Cap on simultaneous MSA array jobs. | Cluster-load throttle. | 10 | Lower to be polite under contention. |
| `mmseq2_db`, `colabfold_db` | Shared DB paths (UniRef30 + ColabFoldDB env DB). | Database choice **is** the real quality lever. | shared cluster paths | Don't change on Kempner (shared). |

The rule hardcodes `--thread 4 --gpu 1`; thread count and GPU only change speed,
not MSA content.

## What actually affects MSA quality (mostly not tuned here)

- **Database depth/diversity** — UniRef30 + the ColabFold environmental DB
  (`--use-env`, on by default) is what produces diverse MSAs, especially for
  orphan/shallow-MSA proteins. This is set by the shared DB paths, not a per-run knob.
- **"Deeper is always better" is a weak heuristic.** The ColabFold authors note
  an MSA with only ~30 sufficiently *diverse* sequences often suffices for
  high-quality predictions — diversity matters more than raw count, and
  ColabFold's default filtering samples sequence space evenly rather than
  maximizing depth.
- ColabFold exposes finer flags (`-s` sensitivity, `--filter`, `--diff`, `--qsc`,
  templates via `--use-templates`) but ProtForge does not surface them; they are
  recipe defaults you generally leave alone. Note GPU MMseqs is incompatible with
  `--use-templates`.

## Recommended recipe

There is no meaningful quality/speed knob to set per-run. Treat MSA as a
fixed-recipe upstream stage: keep the shared DBs, and tune only
`max_files_per_job` / concurrency for throughput via the resource estimator. If a
downstream model (Boltz/OpenFold) underperforms on orphan proteins, the issue is
MSA *depth from the databases*, not a tunable here. If you only run ESMFold2 /
ESM-C, you can disable the MSA stage entirely.

## Sources

- colabfold_search CLI defaults (-s, --filter, --diff, --qsc, --use-env, db1–4):
  https://github.com/sokrypton/ColabFold/blob/main/colabfold/mmseqs/search.py
- ColabFold README (GPU usage, DB sizes / RAM, setup_databases.sh):
  https://github.com/sokrypton/ColabFold/blob/main/README.md
- ColabFold paper (~30 diverse seqs often suffice; ColabFoldDB diversity):
  Mirdita et al. 2022, Nature Methods — https://pmc.ncbi.nlm.nih.gov/articles/PMC9184281/
- ESMFold/ESM-C are MSA-free: https://github.com/facebookresearch/esm
