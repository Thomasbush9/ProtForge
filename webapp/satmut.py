"""
Saturation-mutagenesis: launch + derive + load.

A standalone single-sequence workflow (its own webapp tab, separate from the
pipeline config):

  1. submit_scan: write the query FASTA and submit a GPU job that runs the
     ESM-C container with --save-logits -> logits.npy. (Cluster-side; reuses the
     session's container/cache/partition settings.)
  2. derive_scan: once logits.npy lands, compute the Len x 20-AA wt-marginal LLR
     matrix with mutation_scan and write mutation_scan.csv. Pure NumPy — runs on
     the login node, no GPU/torch.
  3. load_matrix: read the CSV back for the heatmap.

Layout under <out_dir>:  input/<name>.fasta  and  <name>/<size>/{logits.npy,
aa_token_ids.json, mutation_scan.csv}.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from session import REPO_ROOT

# mutation_scan lives in workflow/scripts (shared with the pipeline side).
_WORKFLOW_SCRIPTS = REPO_ROOT / "workflow" / "scripts"
if str(_WORKFLOW_SCRIPTS) not in sys.path:
    sys.path.append(str(_WORKFLOW_SCRIPTS))

import mutation_scan  # noqa: E402

ESMC_SIZES = ["6B", "600M", "300M"]


# --- paths ----------------------------------------------------------------


def input_fasta(out_dir: str | Path, name: str) -> Path:
    return Path(out_dir) / "input" / f"{name}.fasta"


def size_dir(out_dir: str | Path, name: str, size: str) -> Path:
    # Matches run_batch_esmc.save_logits_outputs: <output-dir>/<name>/<size>/
    return Path(out_dir) / name / size


def logits_path(out_dir: str | Path, name: str, size: str) -> Path:
    return size_dir(out_dir, name, size) / "logits.npy"


def matrix_csv(out_dir: str | Path, name: str, size: str) -> Path:
    return size_dir(out_dir, name, size) / "mutation_scan.csv"


def write_query_fasta(out_dir: str | Path, name: str, seq: str) -> Path:
    p = input_fasta(out_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f">{name}\n{seq}\n")
    return p


# --- launch (cluster-side; not unit-tested) -------------------------------


def resolve_cluster_settings(cfg: dict) -> dict:
    """Pull the container/cache/SLURM settings a scan job needs from the session
    config. Returns {sif, cache, partition, account, runtime, log_dir} and a
    list of `errors` for anything missing."""
    containers = cfg.get("containers", {})
    sif = containers.get("esmc") or containers.get("gpu") or ""
    runtime = containers.get("runtime", "auto")
    if runtime == "auto":
        runtime = "singularity"
    esmc = cfg.get("esmc", {})
    slurm = cfg.get("slurm", {})
    res = slurm.get("resources", {}).get("esmc", {})
    settings = {
        "sif": sif,
        "cache": esmc.get("cache_dir", ""),
        "partition": slurm.get("esmc", {}).get("partition") or slurm.get("partition", ""),
        "account": slurm.get("account", ""),
        "runtime": runtime,
        "log_dir": slurm.get("log_dir", ""),
        "mem_mb": int(res.get("mem_mb", 128000)),
        "cpus": int(res.get("cpus_per_task", 8)),
        "runtime_min": int(res.get("runtime", 120)),
    }
    errors = []
    if not settings["sif"]:
        errors.append("No ESM container — set containers.esmc (or containers.gpu) in Configuration.")
    if not settings["cache"]:
        errors.append("No HF cache — set esmc.cache_dir in Configuration.")
    if not settings["partition"]:
        errors.append("No SLURM partition — set slurm.partition in Configuration.")
    settings["errors"] = errors
    return settings


def build_sbatch_command(cfg: dict, out_dir: str | Path, name: str, size: str) -> list[str]:
    """Construct the sbatch command for a scan job (no side effects).

    Submits scripts/run_satmut.sh with cluster settings from cfg. Exposed
    separately from submit_scan so the command can be inspected/tested.
    """
    s = resolve_cluster_settings(cfg)
    out_dir = Path(out_dir)
    cmd = [
        "sbatch", "--parsable",
        "--job-name", f"satmut_{name}_{size}",
        "--partition", s["partition"],
        "--gpus", "1",
        "--cpus-per-task", str(s["cpus"]),
        "--mem", f"{s['mem_mb']}M",
        "--time", str(s["runtime_min"]),
    ]
    if s["account"]:
        cmd += ["--account", s["account"]]
    if s["log_dir"]:
        cmd += ["--output", str(Path(s["log_dir"]) / f"satmut_{name}_{size}.%j.out")]
    cmd += [
        str(REPO_ROOT / "scripts" / "run_satmut.sh"),
        "--sif", s["sif"],
        "--cache", s["cache"],
        "--input", str((out_dir / "input").resolve()),
        "--output", str(out_dir.resolve()),
        "--runner", str(REPO_ROOT / "containers" / "run_batch_esmc.py"),
        "--size", size,
        "--runtime", s["runtime"],
    ]
    return cmd


def submit_scan(cfg: dict, out_dir: str | Path, name: str, seq: str, size: str) -> dict:
    """Write the FASTA and submit the scan job. Returns {job_id|None, command, error}."""
    write_query_fasta(out_dir, name, seq)
    cmd = build_sbatch_command(cfg, out_dir, name, size)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=REPO_ROOT)
    except Exception as e:
        return {"job_id": None, "command": " ".join(cmd), "error": str(e)}
    if r.returncode != 0:
        return {"job_id": None, "command": " ".join(cmd),
                "error": (r.stderr or r.stdout).strip()}
    job_id = r.stdout.strip().split(";")[0]  # --parsable -> "<jobid>[;cluster]"
    meta = {"job_id": job_id, "name": name, "size": size, "seq_len": len(seq),
            "submitted": datetime.now().isoformat()}
    (Path(out_dir) / name / size).mkdir(parents=True, exist_ok=True)
    (size_dir(out_dir, name, size) / "scan_meta.json").write_text(json.dumps(meta))
    return {"job_id": job_id, "command": " ".join(cmd), "error": None}


# --- derive matrix from logits (pure NumPy; tested) -----------------------


def derive_scan(out_dir: str | Path, name: str, size: str, seq: str) -> Path:
    """Compute the LLR matrix from a finished job's logits.npy and write the CSV.

    Returns the CSV path. Raises FileNotFoundError if logits.npy isn't there yet.
    """
    sd = size_dir(out_dir, name, size)
    lp = sd / "logits.npy"
    aap = sd / "aa_token_ids.json"
    if not lp.is_file() or not aap.is_file():
        raise FileNotFoundError(f"logits not ready in {sd} (need logits.npy + aa_token_ids.json)")
    logits = np.load(lp)
    aa_token_ids = json.loads(aap.read_text())
    llr = mutation_scan.wt_marginal_llr(logits, seq, aa_token_ids)
    return mutation_scan.write_scan_csv(matrix_csv(out_dir, name, size), seq, llr)


def load_matrix(csv_path: str | Path) -> tuple[str, list[str], np.ndarray]:
    """Read a mutation_scan.csv back into (wt_seq, aa_order, matrix LxA).

    CSV columns: position, wt_aa, sensitivity, then one column per amino acid.
    """
    csv_path = Path(csv_path)
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        aa_order = header[3:]  # after position, wt_aa, sensitivity
        wt, rows = [], []
        for r in reader:
            wt.append(r[1])
            rows.append([float(x) for x in r[3:]])
    return "".join(wt), aa_order, np.asarray(rows, dtype=float)
