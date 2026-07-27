#!/usr/bin/env python3
"""Build the per-job observations table that everything else fits against.

One row per SLURM job actually run, carrying BOTH the workload that job was
given and what it actually cost:

    workload   n_seqs, mean_len, max_len, p95_len, total_residues
    requested  mem_mb, runtime_min, cpus, gpu/partition
    observed   wall_s, host RAM peak, VRAM peak, SM utilization, io_in

Sources, and why each one is needed:
  * snakemake log  — the ONLY place that ties a SLURM jobid to its rule, its
                     wildcards (chunk_id), and the resources we requested.
  * chunk dirs     — the sequences that were in that chunk. Preferred over
                     chunk_stats.tsv, which is written as all-zeros for
                     FASTA-sourced stages (esmc, esmfold-from-fasta).
  * benchmark TSV  — snakemake's max_rss high-water mark. Trustworthy for host
                     RAM; jobstats undersamples it (see ../jobstats/README.md).
  * jobstats       — the only source of GPU SM% and VRAM on this cluster.

`jobstats` is slow (~1-2 s/job), so results are cached by jobid; re-runs are
instant and safe to iterate against.

Usage:
    python -m scripts.calibrate.optimize.dataset \
        --run OUTPUT_DIR:SNAKEMAKE_LOG [--run ...] --out observations.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

GIB = 2 ** 30

RULE_TO_STAGE = {
    "run_colabfold_search": "msa",
    "run_boltz_predict": "boltz",
    "run_esmc": "esmc",
    "run_esmfold": "esmfold",
    "run_openfold_predict": "openfold",
}

SUBMIT_RE = re.compile(r"Job (\d+) has been submitted with SLURM jobid (\d+)")
BLOCK_START_RE = re.compile(r"^(?:rule|checkpoint) ([A-Za-z_0-9]+):\s*$")


# --- snakemake log parsing -------------------------------------------------

def parse_snakemake_log(path: Path) -> list[dict]:
    """Extract one record per submitted job: rule, wildcards, requested resources.

    Snakemake writes an indented block per job, then a submission line carrying
    the SLURM id. We collect blocks keyed by snakemake's internal jobid, then
    attach SLURM ids from the submission lines.
    """
    text = path.read_text(errors="replace").splitlines()
    blocks: dict[str, dict] = {}
    sm_to_slurm: dict[str, str] = {}

    cur: dict | None = None
    for line in text:
        m = BLOCK_START_RE.match(line)
        if m:
            cur = {"rule": m.group(1), "fields": {}}
            continue
        if cur is not None:
            if line.startswith("    ") and ":" in line:
                key, _, val = line.strip().partition(":")
                cur["fields"][key.strip()] = val.strip()
                continue
            # block ended
            jid = cur["fields"].get("jobid")
            if jid:
                blocks[jid] = cur
            cur = None
        s = SUBMIT_RE.search(line)
        if s:
            sm_to_slurm[s.group(1)] = s.group(2)
    if cur is not None and cur["fields"].get("jobid"):
        blocks[cur["fields"]["jobid"]] = cur

    out = []
    for sm_id, slurm_id in sm_to_slurm.items():
        b = blocks.get(sm_id)
        if not b:
            continue
        f = b["fields"]
        out.append({
            "slurm_id": slurm_id,
            "rule": b["rule"],
            "stage": RULE_TO_STAGE.get(b["rule"], b["rule"]),
            "wildcards": _kv(f.get("wildcards", "")),
            "resources": _kv(f.get("resources", "")),
            "benchmark": f.get("benchmark", ""),
            "input": [p.strip() for p in f.get("input", "").split(",") if p.strip()],
        })
    return out


def _kv(s: str) -> dict:
    """Parse snakemake's 'a=1, b=foo' field lists. Values may contain quotes."""
    out = {}
    for part in s.split(","):
        k, _, v = part.strip().partition("=")
        if k and v:
            out[k.strip()] = v.strip()
    return out


# --- workload features -----------------------------------------------------

_LEN_CACHE: dict[str, int | None] = {}


def _fasta_len(p: Path) -> int | None:
    try:
        seq = "".join(l.strip() for l in p.read_text().splitlines()
                      if not l.startswith(">"))
        return len(seq) or None
    except OSError:
        return None


def _fasta_lens_multi(p: Path) -> list[int]:
    """A combined.fasta may hold many records."""
    lens, cur = [], 0
    try:
        for line in p.read_text().splitlines():
            if line.startswith(">"):
                if cur:
                    lens.append(cur)
                cur = 0
            else:
                cur += len(line.strip())
    except OSError:
        return []
    if cur:
        lens.append(cur)
    return lens


def _yaml_len(p: Path) -> int | None:
    """Boltz YAML: pull the protein sequence without a yaml dependency."""
    try:
        txt = p.read_text()
    except OSError:
        return None
    m = re.search(r"sequence:\s*([A-Za-z]+)", txt)
    return len(m.group(1)) if m else None


def _json_manifest_lens(p: Path) -> list[int]:
    """OpenFold chunk_N.json: {"queries": {name: {"chains": [{"sequence": ...}]}}}.

    OpenFold is handed a manifest rather than a chunk dir, so it needs its own
    reader; a multi-chain query contributes the sum of its chains.
    """
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    lens = []
    for q in (d.get("queries") or {}).values():
        total = sum(len(c.get("sequence", "") or "")
                    for c in (q.get("chains") or []))
        if total:
            lens.append(total)
    return lens


def lengths_for_inputs(inputs: list[str]) -> list[int]:
    """Sequence lengths for everything this job was handed.

    Handles the three shapes the pipeline uses: a combined.fasta, a chunk dir of
    per-sequence .fasta symlinks, and a chunk dir of boltz .yaml files.
    """
    lens: list[int] = []
    for raw in inputs:
        p = Path(raw)
        key = str(p)
        if key in _LEN_CACHE:
            cached = _LEN_CACHE[key]
            if isinstance(cached, list):
                lens.extend(cached)
                continue
        found: list[int] = []
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.name.endswith(".fasta"):
                    v = _fasta_len(f)
                    if v:
                        found.append(v)
                elif f.name.endswith((".yaml", ".yml")):
                    v = _yaml_len(f)
                    if v:
                        found.append(v)
        elif p.name.endswith(".fasta"):
            found = _fasta_lens_multi(p)
        elif p.name.endswith(".json"):
            found = _json_manifest_lens(p)
        elif p.name.endswith((".yaml", ".yml")):
            # runner.yaml is config, not a query — it has no `sequence:` key.
            v = _yaml_len(p)
            found = [v] if v else []
        _LEN_CACHE[key] = found
        lens.extend(found)
    return lens


def chunk_stats_fallback(output_dir: Path, stage: str, chunk_id: str) -> dict | None:
    """chunk_stats.tsv, used only when the chunk dirs are gone.

    Written as all-zeros for FASTA-sourced stages, so zero rows are rejected.
    """
    tsv = output_dir / f"{stage}_chunks" / "chunk_stats.tsv"
    if not tsv.exists():
        return None
    try:
        with tsv.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("chunk_id") == str(chunk_id):
                    n = int(float(row.get("num_seqs", 0) or 0))
                    if n <= 0:
                        return None
                    return {
                        "n_seqs": n,
                        "mean_len": float(row.get("mean_len", 0) or 0),
                        "max_len": float(row.get("max_len", 0) or 0),
                        "p95_len": float(row.get("p95_len", 0) or 0),
                        "total_residues": float(row.get("total_residues", 0) or 0),
                    }
    except (OSError, ValueError):
        return None
    return None


def summarize_lengths(lens: list[int]) -> dict:
    """Chunk features.

    `sum_len2` (ΣL²) matters: if per-sequence cost is a + b·L + c·L², a chunk's
    total cost is a·N + b·ΣL + c·ΣL². Carrying ΣL² makes the job-time model
    LINEAR in its parameters, so it fits by plain least squares with no
    distributional approximation. `max_len` drives peak VRAM (the longest
    sequence in the chunk sets the high-water mark).
    """
    if not lens:
        return {}
    s = sorted(lens)
    return {
        "n_seqs": len(s),
        "mean_len": round(statistics.mean(s), 1),
        "max_len": s[-1],
        "p95_len": s[min(len(s) - 1, int(0.95 * len(s)))],
        "total_residues": sum(s),
        "sum_len2": sum(v * v for v in s),
    }


# --- observed cost ---------------------------------------------------------

def read_benchmark(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    lines = p.read_text().strip().splitlines()
    if len(lines) < 2:
        return {}
    hdr, row = lines[0].split("\t"), lines[1].split("\t")
    rec = dict(zip(hdr, row))

    def num(k):
        try:
            return float(rec.get(k, "") or 0)
        except ValueError:
            return 0.0

    return {
        "wall_s": num("s"),
        "max_rss_gb": num("max_rss") / 1024,   # snakemake writes MB
        "io_in_gb": num("io_in") / 1024,
        "cpu_time_s": num("cpu_time"),
        "mean_load": num("mean_load"),
    }


def sh(cmd: list[str], timeout: int = 120) -> str:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else ""
    except (subprocess.SubprocessError, OSError):
        return ""


def sacct_placement(jids: list[str]) -> dict:
    """Node and start/end per job, for working out co-tenancy.

    An I/O-bound stage slows down when SLURM packs several of its jobs onto one
    node — MSA reads the shared colabfold database over Lustre, and two MSA jobs
    on a node contend for exactly the resource that dominates their runtime
    (measured: 1.19x slower at N=25). A model whose only inputs are chunk size
    and sequence length cannot see that, so record placement and let the fit
    account for it.
    """
    out: dict = {}
    if not jids:
        return out
    txt = sh(["sacct", "-X", "-n", "-P", "-j", ",".join(jids),
              "--format=JobID,NodeList,Start,End"])
    for line in txt.strip().splitlines():
        f = line.split("|")
        if len(f) >= 4:
            out[f[0]] = {"node": f[1], "start": f[2], "end": f[3]}
    return out


def annotate_cotenancy(rows: list[dict]) -> None:
    """Set `jobs_on_node` = harvested jobs that shared a node at the same time."""
    from datetime import datetime

    def ts(v):
        try:
            return datetime.strptime(v, "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError):
            return None

    for r in rows:
        a0, a1, node = ts(r.get("_start")), ts(r.get("_end")), r.get("node")
        if not (a0 and a1 and node):
            r["jobs_on_node"] = ""
            continue
        n = 1
        for other in rows:
            if other is r or other.get("node") != node:
                continue
            b0, b1 = ts(other.get("_start")), ts(other.get("_end"))
            if b0 and b1 and b0 < a1 and a0 < b1:      # overlapping intervals
                n += 1
        r["jobs_on_node"] = n


def jobstats_cached(jid: str, cache: dict) -> dict:
    """GPU SM%/VRAM from jobstats. Empty dict when the job was too short to sample."""
    if jid in cache:
        return cache[jid]
    out: dict = {}
    try:
        p = subprocess.run(["jobstats", "-j", jid], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=120)
        js = json.loads(p.stdout) if p.stdout.strip() else {}
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        js = {}
    node = next(iter(js.get("nodes", {}).values()), None)
    if node:
        out = {
            "gpu_util_pct": max(node.get("gpu_utilization", {}).values(), default=None),
            "vram_gb": max(node.get("gpu_used_memory", {}).values(), default=0) / GIB,
            "vram_total_gb": max(node.get("gpu_total_memory", {}).values(), default=0) / GIB,
            "js_host_gb": node.get("used_memory", 0) / GIB,
            "cpus": node.get("cpus", 0),
        }
    if js.get("total_time"):
        out["js_wall_s"] = js["total_time"]
    cache[jid] = out
    return out


# --- assembly --------------------------------------------------------------

def build(runs: list[tuple[Path, Path]], cache_path: Path) -> list[dict]:
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache = {}

    rows = []
    for output_dir, log in runs:
        jobs = parse_snakemake_log(log)
        print(f"  {log.name}: {len(jobs)} jobs", file=sys.stderr)
        placement = sacct_placement([j["slurm_id"] for j in jobs])
        for j in jobs:
            feats = summarize_lengths(lengths_for_inputs(j["input"]))
            if not feats:
                cid = j["wildcards"].get("chunk_id") or j["wildcards"].get("group_id")
                if cid is not None:
                    feats = chunk_stats_fallback(output_dir, j["stage"], cid) or {}
            if not feats:
                continue  # no workload features -> useless for fitting

            res = j["resources"]
            bench = read_benchmark(j["benchmark"])
            js = jobstats_cached(j["slurm_id"], cache)

            # Wall time MUST come from jobstats: it is keyed by SLURM jobid.
            # The benchmark TSV is keyed by chunk, so re-running a chunk
            # overwrites it and the same file would otherwise be attributed to
            # several distinct jobs.
            wall = js.get("js_wall_s") or bench.get("wall_s") or 0.0
            host_gb = max(bench.get("max_rss_gb", 0.0), js.get("js_host_gb", 0.0))

            pl = placement.get(j["slurm_id"], {})
            rows.append({
                "node": pl.get("node", ""),
                "_start": pl.get("start", ""),
                "_end": pl.get("end", ""),
                "run": output_dir.name,
                # The same output dir is reused across runs, so the log stem is
                # what actually distinguishes one run from another. Without it a
                # degraded run is indistinguishable from a healthy one.
                "log": log.name.replace(".snakemake.log", ""),
                "slurm_id": j["slurm_id"],
                "stage": j["stage"],
                "size": j["wildcards"].get("size", ""),
                **feats,
                "req_mem_gb": _f(res.get("mem_mb")) / 1024 if res.get("mem_mb") else 0.0,
                "req_runtime_min": _f(res.get("runtime")),
                "req_cpus": _f(res.get("cpus_per_task")),
                "partition": res.get("slurm_partition", ""),
                "wall_s": round(wall, 1),
                "host_gb": round(host_gb, 2),
                "vram_gb": round(js["vram_gb"], 2) if "vram_gb" in js else "",
                "gpu_util_pct": js.get("gpu_util_pct", ""),
                "io_in_gb": round(bench.get("io_in_gb", 0.0), 2),
                "cpu_time_s": round(bench.get("cpu_time_s", 0.0), 1),
            })

    annotate_cotenancy(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    return rows


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


FIELDS = ["run", "log", "slurm_id", "node", "jobs_on_node",
          "stage", "size", "n_seqs", "mean_len", "max_len",
          "p95_len", "total_residues", "sum_len2", "req_mem_gb",
          "req_runtime_min", "req_cpus", "partition", "wall_s", "host_gb",
          "vram_gb", "gpu_util_pct", "io_in_gb", "cpu_time_s"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="OUTDIR:LOG",
                    help="output dir and its snakemake log, colon-separated")
    ap.add_argument("--out", type=Path, default=Path("observations.csv"))
    ap.add_argument("--cache", type=Path, default=Path(".jobstats_cache.json"))
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        out, _, log = spec.rpartition(":")
        if not out or not log:
            print(f"bad --run {spec!r}, want OUTDIR:LOG", file=sys.stderr)
            return 2
        runs.append((Path(out), Path(log)))

    rows = build(runs, args.cache)
    if not rows:
        print("no usable job rows", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_stage: dict[str, int] = {}
    for r in rows:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
    print(f"\nwrote {args.out}  ({len(rows)} jobs)")
    for s, n in sorted(by_stage.items()):
        rs = [r for r in rows if r["stage"] == s]
        lens = [r["mean_len"] for r in rs]
        ns = [r["n_seqs"] for r in rs]
        print(f"  {s:<10} n={n:<4} N/job {min(ns)}-{max(ns)}  "
              f"mean_len {min(lens):.0f}-{max(lens):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
