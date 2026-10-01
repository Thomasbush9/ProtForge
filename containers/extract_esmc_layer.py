#!/usr/bin/env python
"""Split each ESM-C `hidden_states.pt` into one `layer_{i}.pt` per saved layer.

`run_batch_esmc.py --save-hidden-states` writes optional raw states alongside
the standard final embeddings in `outputs.pt`. Each raw-state file records:

    {"hidden_states": tensor, "layers": [layer IDs], "pool": "none" or "mean",
     "name", "size", "n_layers", "n_residues", "layer_index"}

The tensor is (selected_layers, L, d_model) without pooling or
(selected_layers, d_model) with mean pooling. This tool writes one bare
(L, d_model) tensor or (d_model,) vector per selected layer next to the source:

    <root>/<name>/esmc/<size>/layer_{recorded_layer_id}.pt

Layer IDs come from the recorded `layers` list, not the stack's row numbers:
selecting layers 30 and 60 writes layer_30.pt and layer_60.pt. Layer 0 is the
embedding output, i is the input to block i, and n_layers is the final
post-LayerNorm output. Malformed or missing selection/pooling metadata fails
rather than guessing which layer a row represents.

Usage:
    python extract_esmc_layer.py /path/to/embeddings/sequences --workers 16
    python extract_esmc_layer.py /path/to/root --dtype float16 --overwrite
    python extract_esmc_layer.py /path/to/root --dry-run
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch


def load_stack(src: Path) -> tuple[torch.Tensor, list[int], str]:
    """Return the validated raw-state stack, recorded layer IDs, and pooling."""
    obj = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "hidden_states" not in obj:
        raise ValueError("expected a dict with 'hidden_states', 'layers', and 'pool'")
    stack = obj["hidden_states"]
    if not isinstance(stack, torch.Tensor):
        raise TypeError(f"expected hidden_states tensor, got {type(stack).__name__}")
    pool = obj.get("pool")
    if pool not in ("none", "mean"):
        raise ValueError("pool must be 'none' or 'mean'")
    expected_ndim = 3 if pool == "none" else 2
    if stack.ndim != expected_ndim or any(dim <= 0 for dim in stack.shape):
        raise ValueError(
            f"pool={pool!r} requires a nonempty {expected_ndim}D stack, "
            f"got {tuple(stack.shape)}"
        )
    layers = obj.get("layers")
    if not isinstance(layers, list) or len(layers) != stack.shape[0]:
        raise ValueError("layers must be a list with one ID per hidden_states row")
    if any(type(layer) is not int or layer < 0 for layer in layers):
        raise ValueError("layers must contain nonnegative integer IDs")
    if len(set(layers)) != len(layers):
        raise ValueError("layers must contain distinct IDs")
    if "n_layers" in obj:
        depth = obj["n_layers"]
        if type(depth) is not int or depth < 0 or any(layer > depth for layer in layers):
            raise ValueError("layer IDs must be within 0..n_layers")
    if "n_residues" in obj:
        length = obj["n_residues"]
        if type(length) is not int or length <= 0:
            raise ValueError("n_residues must be a positive integer")
        if pool == "none" and length != stack.shape[1]:
            raise ValueError("n_residues does not match the hidden_states residue axis")
    return stack, layers, pool


def process_dir(src: str, dtype: str | None, overwrite: bool,
                expect_layers: int | None) -> tuple[str, str]:
    """Split one hidden_states.pt. Returns (directory, status)."""
    # Each worker gets one core's worth of BLAS; the parallelism is across
    # directories, and oversubscribing here just thrashes.
    torch.set_num_threads(1)

    src = Path(src)
    d = src.parent
    try:
        stack, layers, _pool = load_stack(src)
        n_files = stack.shape[0]
        if expect_layers is not None and n_files != expect_layers:
            return str(d), (
                f"error: {n_files} layers, expected {expect_layers}"
            )

        targets = [d / f"layer_{layer}.pt" for layer in layers]
        if not overwrite and all(p.exists() for p in targets):
            return str(d), "skipped"

        for i, p in enumerate(targets):
            # .clone() is critical, and .contiguous() will NOT do: slicing dim
            # 0 yields an already-contiguous view, so .contiguous() is a no-op
            # that keeps the original storage, and torch.save serializes the
            # WHOLE storage behind a view -- every layer_i.pt would come out as
            # big as hidden_states.pt. .clone() allocates just this layer.
            layer = stack[i].clone()
            if dtype is not None:
                layer = layer.to(getattr(torch, dtype))
            tmp = p.with_suffix(".pt.tmp")  # atomic rename, safe to resume
            torch.save(layer, tmp)
            os.replace(tmp, p)

        return str(d), "ok"
    except Exception as e:  # noqa: BLE001
        return str(d), f"error: {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("root", type=Path,
                    help="directory to search for hidden_states.pt files")
    ap.add_argument("--pattern", default="**/hidden_states.pt",
                    help="glob for source files (default: %(default)s)")
    ap.add_argument("--expect-layers", type=int, default=None,
                    help="fail a directory whose stack does not contain this "
                         "many selected layers; default: accept any valid selection")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dtype", default=None,
                    help="cast each layer, e.g. float16 (default: keep as-is)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite directories that already look complete")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be done and exit")
    args = ap.parse_args()

    srcs = sorted(str(p) for p in args.root.glob(args.pattern))
    print(f"found {len(srcs)} raw-state files under {args.root}", flush=True)
    if not srcs:
        raise SystemExit(f"nothing matched {args.pattern!r} under {args.root}")

    probe, probe_layers, probe_pool = load_stack(Path(srcs[0]))
    n_files = probe.shape[0]
    print(f"probe {srcs[0]}: shape {tuple(probe.shape)} dtype {probe.dtype} "
          f"pool={probe_pool} -> layers {probe_layers} "
          f"({n_files} files/dir)", flush=True)

    if args.dry_run:
        est = len(srcs) * n_files
        print(f"dry run: would write {est} files "
              f"({est / max(len(srcs), 1):.0f} per directory)")
        return

    counts = {"ok": 0, "skipped": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(process_dir, s, args.dtype, args.overwrite,
                      args.expect_layers)
            for s in srcs
        ]
        for n, fut in enumerate(as_completed(futs), 1):
            name, status = fut.result()
            key = "error" if status.startswith("error") else status
            counts[key] += 1
            if key == "error":
                print(f"[FAIL] {name}: {status}", flush=True)
            if n % 250 == 0 or n == len(srcs):
                print(f"{n}/{len(srcs)}  {counts}", flush=True)

    (args.root / "layer_split_manifest.json").write_text(json.dumps({
        "schema": "protforge/layer_split@1",
        "source_pattern": args.pattern,
        "n_directories": len(srcs),
        "files_per_directory": n_files,
        "dtype": args.dtype or str(probe.dtype).replace("torch.", ""),
        "layer_index": "filenames use recorded layers IDs; 0 = embedding output; "
                       "n_layers = final post-norm (== last_hidden_state)",
        "counts": counts,
    }, indent=2) + "\n")

    print("done:", counts)
    if counts["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
