import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

ESMFOLD2_REPO = "biohub/ESMFold2-Fast"
ESMFOLD_VARIANT = "fast"


def _enforce_offline(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return f"{cache_dir}/hub"


# TF32 tensor-core path for the fp32 matmuls only. ESMFold2Model.forward
# manages its OWN bf16 autocast internally and deliberately up-casts pair/z to
# fp32 before the distogram + pLDDT/pTM/ipTM confidence heads (Biohub
# transformers @3a8956fb: modeling_esmfold2.py:218-223, 987-1021). So we do NOT
# wrap fold() in an outer autocast — that would be redundant and risk overriding
# those fp32 confidence boundaries. TF32 backend flags are orthogonal to those
# boundaries and safe. No-op on CPU-only hosts.
def _enable_tf32() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _seq_from_yaml(path: Path) -> str:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or "sequences" not in data:
        raise KeyError(f"{path}: missing 'sequences'")
    entry = data["sequences"][0]
    if "protein" not in entry or "sequence" not in entry["protein"]:
        raise KeyError(f"{path}: expected sequences[0].protein.sequence")
    return str(entry["protein"]["sequence"]).strip()


def _seq_from_fasta(path: Path) -> str:
    """Return the sequence of the single record in a FASTA file."""
    records, cur = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                records.append("".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        records.append("".join(cur))
    if not records:
        raise ValueError(f"{path}: no sequence found")
    if len(records) > 1:
        raise ValueError(f"{path}: expected one sequence per FASTA, found {len(records)}")
    return records[0]


def parse_inputs(input_dir: str) -> dict[str, str]:
    """Read sequences keyed by file stem from .yaml and/or .fasta/.fa inputs."""
    base = Path(input_dir)
    paths = sorted([*base.glob("*.yaml"), *base.glob("*.fasta"), *base.glob("*.fa")])
    if not paths:
        raise FileNotFoundError(f"No *.yaml/*.fasta in {input_dir}")

    out: dict[str, str] = {}
    for p in paths:
        seq = _seq_from_yaml(p) if p.suffix == ".yaml" else _seq_from_fasta(p)
        stem = p.stem
        if stem in out:
            raise ValueError(f"Duplicate stem {stem!r} in {input_dir}")
        out[stem] = seq
    return out


def save_outputs(output_dir: str, name: str, variant: str, result) -> None:
    out_dir = Path(output_dir) / name / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "structure.cif").write_text(result.complex.to_mmcif())

    plddt = result.plddt
    if torch.is_tensor(plddt):
        plddt = plddt.detach().cpu().numpy()
    plddt = np.asarray(plddt)
    np.save(out_dir / "plddt.npy", plddt)

    meta = {"name": name, "variant": variant, "ptm": result.ptm}
    if result.iptm is not None:
        meta["iptm"] = result.iptm
    torch.save(meta, out_dir / "metrics.pt")

    # Light, torch-free copy of the scalar metrics so the dashboard (and
    # output_schema) can read ptm/pLDDT without loading the .pt pickle.
    def _scalar(v):
        return float(v.item()) if torch.is_tensor(v) else float(v)

    metrics_json = {
        "name": name, "variant": variant,
        "ptm": _scalar(result.ptm),
        "mean_plddt": float(np.mean(plddt)),
    }
    if result.iptm is not None:
        metrics_json["iptm"] = _scalar(result.iptm)
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json))


def load_model(hub_cache: str) -> ESMFold2Model:
    return ESMFold2Model.from_pretrained(
        ESMFOLD2_REPO,
        cache_dir=hub_cache,
        local_files_only=True,
    ).cuda().eval()


def _maybe_compile(model):
    """Best-effort torch.compile of the resident model. ESMFold2 folds variable
    lengths, so dynamic=True caps recompiles; identical-shape diffusion/loop
    forwards then reuse the fused graph across the many sequences one loaded
    model serves. Compile is opt-in (--compile) because the SDK's custom Triton
    kernels may not trace cleanly; on any failure we warn and fall back to eager
    rather than abort the job. NOT a precision change — the model still owns its
    internal bf16/fp32 boundaries."""
    try:
        compiled = torch.compile(model, dynamic=True)
        print("torch.compile enabled (dynamic=True)", flush=True)
        return compiled
    except Exception as e:  # noqa: BLE001 — never let compile kill a fold job
        print(f"WARNING: torch.compile failed ({e}); using eager model", flush=True)
        return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    # Group persistence: load ESMFold2 ONCE and fold every chunk in --input-dirs
    # (parallel to --output-dirs). fold() is one-complex-per-call (multiple
    # proteins would fold as ONE interacting complex), so folds stay serial —
    # the win here is amortizing the model load across the whole group, not
    # batching. Single-chunk runs pass one of each (pre-grouping behaviour).
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dirs", nargs="+", required=True)
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the resident model (opt-in; falls "
                             "back to eager on failure).")
    args = parser.parse_args()
    if len(args.input_dirs) != len(args.output_dirs):
        parser.error("--input-dirs and --output-dirs must be equal length")
    hub_cache = _enforce_offline(args.cache)
    _enable_tf32()

    model = load_model(hub_cache)            # ONE load for the whole group
    if args.compile:
        model = _maybe_compile(model)
    builder = ESMFold2InputBuilder()

    for in_dir, out_dir in zip(args.input_dirs, args.output_dirs):
        for name, sequence in parse_inputs(in_dir).items():
            print(f"Folding {name} ({len(sequence)} aa)...", flush=True)
            spi = StructurePredictionInput(
                sequences=[ProteinInput(id="A", sequence=sequence)]
            )
            # Time only the model call (CUDA-synced) for the stair benchmark's
            # infer_s column; load/setup is excluded by construction.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            result = builder.fold(
                model,
                spi,
                num_loops=args.num_loops,
                num_sampling_steps=args.num_sampling_steps,
                num_diffusion_samples=1,
                seed=args.seed,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"BENCH_INFER_S {name} {time.perf_counter() - _t0:.4f}", flush=True)
            save_outputs(out_dir, name, ESMFOLD_VARIANT, result)

            plddt_mean = float(result.plddt.mean())
            ptm = float(result.ptm)
            msg = f"  {name}: pLDDT mean {plddt_mean:.3f}, pTM {ptm:.3f}"
            if result.iptm is not None:
                msg += f", ipTM {float(result.iptm):.3f}"
            print(msg, flush=True)
