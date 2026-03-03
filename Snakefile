"""
ProtForge Snakemake Workflow
============================
Orchestrates the 4-stage protein prediction pipeline:
  MSA -> Boltz -> ESM -> ES

Usage:
  snakemake --profile profiles/slurm/           # Full pipeline via SLURM
  snakemake --profile profiles/slurm/ -n        # Dry run
  snakemake --profile profiles/slurm/ --dag | dot -Tpng > dag.png
  snakemake --profile profiles/slurm/ --rerun-incomplete
"""

configfile: "config.yaml"

RUN_MSA   = config["pipeline"].get("msa", True)
RUN_BOLTZ = config["pipeline"].get("boltz", True)
RUN_ESM   = config["pipeline"].get("esm", True)
RUN_ES    = config["pipeline"].get("es", True)
OUTPUT    = config["output"]["parent_dir"]
SLURM_CFG = config.get("slurm", {})
SEQUENCES_DIR = f"{OUTPUT}/sequences"

# SLURM email notifications (reads slurm.email from config)
SLURM_EMAIL = SLURM_CFG.get("email", "")

def slurm_extra(gpu=False):
    """Build slurm_extra string with optional GPU and mail flags.

    Each sbatch flag must be individually quoted so the shell
    passes them as separate arguments to sbatch.
    """
    parts = []
    if gpu:
        parts.append("'--gpus-per-node=1'")
    if SLURM_EMAIL:
        parts.append(f"'--mail-type=END,FAIL'")
        parts.append(f"'--mail-user={SLURM_EMAIL}'")
    return " ".join(parts) if parts else "''"

# Container support (set .sif paths in config to enable)
CONTAINERS = config.get("containers", {})
BIND_PATHS = CONTAINERS.get("bind_paths", "/n/holylfs06,/n/home06")

def container_cmd(stage):
    """Return 'singularity exec --nv -B ... sif' prefix, or '' for legacy mode."""
    sif = CONTAINERS.get(stage, "")
    if sif:
        binds = " ".join(f"-B {p}" for p in BIND_PATHS.split(","))
        return f"singularity exec --nv {binds} {sif}"
    return ""

if RUN_MSA:
    include: "workflow/rules/msa.smk"
if RUN_BOLTZ:
    include: "workflow/rules/boltz.smk"
if RUN_ESM:
    include: "workflow/rules/esm.smk"
if RUN_ES:
    include: "workflow/rules/es.smk"


def get_targets():
    """Build the list of final sentinel files based on pipeline toggles."""
    targets = []
    if RUN_MSA:
        targets.append(f"{OUTPUT}/.msa_complete")
    if RUN_BOLTZ:
        targets.append(f"{OUTPUT}/.boltz_complete")
    if RUN_ESM:
        targets.append(f"{OUTPUT}/.esm_complete")
    if RUN_ES:
        targets.append(f"{OUTPUT}/es/.done")
    return targets


rule all:
    input:
        get_targets(),
