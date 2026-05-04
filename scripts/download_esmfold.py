"""Pre-download facebook/esmfold_v1 into a chosen HF cache directory.

Run on a login node (with internet) BEFORE submitting the SLURM job. The
compute node then loads the model offline by setting HF_HOME to the same path
(combine with HF_HUB_OFFLINE=1).

Usage:
    python download_esmfold.py --cache-dir /path/to/esmfold_cache
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path)
    args = ap.parse_args()

    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)

    from transformers import AutoTokenizer, EsmForProteinFolding

    model_id = "facebook/esmfold_v1"
    print(f"HF_HOME={cache_dir}", flush=True)
    print(f"Downloading tokenizer for {model_id}...", flush=True)
    AutoTokenizer.from_pretrained(model_id)
    print(
        f"Downloading model weights for {model_id} (this can take a while)...",
        flush=True,
    )
    EsmForProteinFolding.from_pretrained(model_id, low_cpu_mem_usage=True)

    model_dir = cache_dir / "hub" / "models--facebook--esmfold_v1"
    if model_dir.exists():
        total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        print(f"Cache populated: {model_dir} ({total / 1e9:.2f} GB)")
    else:
        print(
            f"Warning: expected cache subdir {model_dir} not found",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
