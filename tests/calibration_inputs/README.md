# Calibration inputs

`slurm_scripts/calibrate.sh` runs a single pipeline stage on a small,
fixed set of proteins so its actual SLURM resource consumption can be
measured per GPU type. The numbers are then fit into
`webapp/scaling_models.calibrated.yaml` by
`webapp/estimator.py::recalibrate_from_benchmarks`.

## What sequences to use

Five real proteins covering the length distribution we care about
(short hormone → multi-domain enzyme). Recommended UniProt accessions:

| Bin     | Length | Accession  | Name                                |
| ------- | ------ | ---------- | ----------------------------------- |
| ~100    | 76     | P0CG47     | Polyubiquitin-B (single ubiquitin)  |
| ~300    | 238    | P42212     | Green fluorescent protein (Aequorea)|
| ~600    | 585    | P02768     | Human serum albumin (mature chain)  |
| ~1000   | 1102   | P00533     | EGFR — kinase domain & flanking     |
| ~1500   | 1480   | P01234     | (substitute with any titin domain)  |

To populate this directory:

```bash
# Use the existing UniProt fetcher (separate venv — see scripts/uniprot_fetch/README.md)
cd scripts/uniprot_fetch/
python fetch_sequences.py \
    --input ../tests/calibration_inputs/manifest.csv \
    --output ../tests/calibration_inputs/fastas/
```

`manifest.csv` lives next to this README and lists the accessions above.
Replace any of them with your own picks if you want — calibration is just
"run the pipeline on N inputs, observe SLURM, fit coefficients."

## Why these aren't checked in as FASTAs

Calibration needs *real* sequences so MSA produces realistic search
results. We list accessions instead of bundling FASTAs to keep the repo
size honest.
