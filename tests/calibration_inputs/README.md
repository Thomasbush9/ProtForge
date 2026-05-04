# Calibration inputs

`slurm_scripts/calibrate.sh` runs a single pipeline stage on a small,
real-world subset of FASTAs so its actual SLURM resource consumption
can be measured per GPU type. The numbers feed into
`webapp/scaling_models.calibrated.yaml` (later, once the simple sweep
has produced enough data to fit).

## Recommended workflow — sample from your real dataset

When you already have a directory of FASTAs you intend to process
(e.g. 7k UniProt-derived sequences), pick a stratified subset rather
than fabricating test data. The sampler weights toward the long tail
because that is where O(L²) memory diverges.

```bash
# 1. Pick ~20 FASTAs spanning the length distribution
python scripts/calibrate/subsample.py \
    --input_dir /path/to/your/7k_fastas \
    --output_dir tests/calibration_inputs/fastas \
    --n 20 \
    --seed 42

# 2. Run the calibration sweep on one GPU type
bash slurm_scripts/calibrate.sh boltz h100 tests/calibration_inputs/fastas

# 3. Repeat for the other GPU types you have access to
bash slurm_scripts/calibrate.sh boltz a100 tests/calibration_inputs/fastas
bash slurm_scripts/calibrate.sh esm   h100 tests/calibration_inputs/fastas
```

`manifest.csv` is written next to the sampled FASTAs and records
`(filename, length, bin, source_path)` so the analysis afterwards can
join sequence length back to the SLURM benchmarks.

## What lands in the output dir

After `calibrate.sh` finishes, `<output>/run/` contains:

```
<output>/run/
  msa_chunks/         (or boltz_chunks/, esm_chunks/, depending on stage)
    chunk_stats.tsv   chunk_id, num_seqs, mean_len, min_len, p95_len, max_len, total_residues
    manifest.txt
    chunk_0/ ...
  benchmarks/<stage>/
    <rule>_<chunk_id>.tsv   per-rule wall time + max RSS (Snakemake's benchmark: directive)
  logs/<stage>/<rule>_<chunk_id>.log
```

`chunk_stats.tsv × benchmarks/<stage>/*.tsv` join on `chunk_id`. That gives
you `(mean_len, p95_len, max_len) → (wall_time, max_rss)` per chunk —
exactly the shape needed to fit `runtime ~ L + L²` and `mem ~ L + L²`
later.

## Fallback — fixed UniProt accessions

If you do not yet have a real input directory (e.g. you're calibrating
before having data), `manifest.csv` here lists 5 fixed accessions
covering ~76, 238, 585, 1102, ~1500 residues. Fetch them with the
UniProt fetcher and use `slurm_scripts/calibrate.sh` against the
resulting dir:

```bash
cd scripts/uniprot_fetch/
python fetch_sequences.py \
    --input ../tests/calibration_inputs/manifest.csv \
    --output ../tests/calibration_inputs/fastas/
```

| Bin     | Length | Accession  | Name                                |
| ------- | ------ | ---------- | ----------------------------------- |
| ~100    | 76     | P0CG47     | Polyubiquitin-B (single ubiquitin)  |
| ~300    | 238    | P42212     | Green fluorescent protein           |
| ~600    | 585    | P02768     | Human serum albumin (mature chain)  |
| ~1000   | 1102   | P00533     | EGFR — kinase domain & flanking     |
| ~1500   | 1480   | (substitute) | any titin domain                  |

Sampling from your real data is preferred — this list is just a
starting set when no real distribution is available yet.
