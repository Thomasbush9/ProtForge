"""
Command-line front door to ProtForge zero-shot saturation mutagenesis.

Wraps the same pure functions the Streamlit Saturation-Mutagenesis tab uses
(webapp/satmut.py + workflow/scripts/mutation_scan.py) so a scan can be launched,
derived, and inspected without the web app:

    # 1. launch a GPU job that produces logits.npy (one ESM-C forward pass / seq)
    python -m webapp.satmut_cli submit --config config.smoke.yaml \
        --seq MSKGEELFTG... --name gfp --size 6B --out-dir /path/scans
    python -m webapp.satmut_cli submit --config config.smoke.yaml \
        --fasta-dir /path/wts --size 600M --out-dir /path/scans
    python -m webapp.satmut_cli submit ... --dry-run     # print sbatch, don't submit

    # 2. once logits.npy lands, derive the Len x 20-AA LLR matrix (pure NumPy, CPU)
    python -m webapp.satmut_cli derive --out-dir /path/scans --size 6B --all

    # 3. read the headline + CSV path back
    python -m webapp.satmut_cli view --out-dir /path/scans --name gfp --size 6B

This is ANALYSIS, not folding: it scores every single-point substitution of a
wild-type sequence with the ESM-C language model (wt-marginal log-likelihood
ratio). Negative LLR = the model disfavours the mutation vs wild-type.

Designed to be driven by the `satmut` Claude Code skill, but usable standalone.
Only `submit` (without --dry-run) touches the cluster; derive/view are read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# satmut.py lives in webapp/ and uses flat imports (from session import ...);
# mutation_scan.py lives in workflow/scripts/. Put both on the path so this
# works as `python -m webapp.satmut_cli` or `python webapp/satmut_cli.py`, and
# mirrors the test suite's PYTHONPATH=webapp:workflow/scripts.
_WEBAPP = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_WEBAPP), "workflow", "scripts")
for _p in (_WEBAPP, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import satmut  # noqa: E402
import mutation_scan  # noqa: E402


# --- inputs ---------------------------------------------------------------


def _read_fasta_seq(path: Path) -> str:
    """Return the single sequence from a FASTA file (one record per file)."""
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
    return records[0].upper()


def _collect_targets(args) -> list[tuple[str, str]]:
    """Resolve the (name, seq) pairs to scan from --seq/--name or --fasta-dir.

    Batch-capable: a single sequence, or every .fasta/.fa in a directory.
    """
    if args.fasta_dir:
        d = Path(args.fasta_dir)
        if not d.is_dir():
            raise SystemExit(f"--fasta-dir not found: {d}")
        paths = sorted([*d.glob("*.fasta"), *d.glob("*.fa")])
        if not paths:
            raise SystemExit(f"No *.fasta/*.fa in {d}")
        out = []
        for p in paths:
            out.append((p.stem, _read_fasta_seq(p)))
        return out
    if args.seq:
        name = args.name or "query"
        return [(name, args.seq.strip().upper())]
    raise SystemExit("Provide either --seq [--name] or --fasta-dir.")


def _load_seq_for_name(out_dir: Path, name: str, override: str | None) -> str:
    """Sequence for a scan: explicit --seq, else the FASTA written at submit time."""
    if override:
        return override.strip().upper()
    fa = satmut.input_fasta(out_dir, name)
    if not fa.is_file():
        raise SystemExit(
            f"No sequence for {name!r}: pass --seq, or ensure {fa} exists "
            "(written by `submit`)."
        )
    return _read_fasta_seq(fa)


def _discover_names(out_dir: Path, size: str) -> list[str]:
    """Names that have a <name>/<size>/ dir under out_dir (i.e. were submitted)."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    names = []
    for child in sorted(out_dir.iterdir()):
        if not child.is_dir() or child.name == "input":
            continue
        if (child / size).is_dir():
            names.append(child.name)
    return names


# --- submit ---------------------------------------------------------------


def _cmd_submit(args) -> int:
    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    settings = satmut.resolve_cluster_settings(cfg)
    out_dir = Path(args.out_dir)
    targets = _collect_targets(args)

    # In dry-run we only need the command shape, so missing-cluster-config errors
    # are surfaced as a warning. A real submit refuses to proceed.
    if settings["errors"]:
        print("Cluster settings incomplete:")
        for e in settings["errors"]:
            print(f"  - {e}")
        if not args.dry_run:
            print("\nFix the config above before submitting (re-run with "
                  "--dry-run to preview the command).", file=sys.stderr)
            return 1
        print("(continuing for --dry-run; the command below would fail to "
              "submit until the above are set)\n")

    print(f"Scans: {len(targets)} sequence(s) | size {args.size} | out {out_dir}\n")
    failures = 0
    for name, seq in targets:
        if args.dry_run:
            satmut.write_query_fasta(out_dir, name, seq)
            cmd = satmut.build_sbatch_command(cfg, out_dir, name, args.size)
            print(f"[{name}] len={len(seq)}  (dry-run, not submitted)")
            print("  " + " ".join(cmd) + "\n")
        else:
            res = satmut.submit_scan(cfg, out_dir, name, seq, args.size)
            if res["error"]:
                failures += 1
                print(f"[{name}] FAILED to submit: {res['error']}")
            else:
                print(f"[{name}] len={len(seq)}  submitted job {res['job_id']}")

    if not args.dry_run:
        if failures:
            print(f"\n{failures}/{len(targets)} submissions failed.", file=sys.stderr)
            return 1
        print(f"\nSubmitted {len(targets)} scan job(s). When they finish, derive "
              f"the matrices:\n  python -m webapp.satmut_cli derive "
              f"--out-dir {out_dir} --size {args.size} --all")
    return 0


# --- derive ---------------------------------------------------------------


def _cmd_derive(args) -> int:
    out_dir = Path(args.out_dir)
    if args.all:
        names = _discover_names(out_dir, args.size)
        if not names:
            raise SystemExit(
                f"No submitted scans found under {out_dir} for size {args.size}.")
    elif args.name:
        names = list(args.name)
    else:
        raise SystemExit("Provide --name NAME [NAME ...] or --all.")

    if args.seq and len(names) > 1:
        raise SystemExit("--seq applies to a single --name; for batches let it "
                         "read each sequence from the submitted FASTA.")

    print(f"Deriving {len(names)} scan(s) | size {args.size}\n")
    derived = skipped = failed = 0
    for name in names:
        try:
            seq = _load_seq_for_name(out_dir, name, args.seq)
        except SystemExit as e:
            failed += 1
            print(f"[{name}] no sequence: {e}")
            continue
        try:
            csv_path = satmut.derive_scan(out_dir, name, args.size, seq)
        except FileNotFoundError as e:
            skipped += 1
            print(f"[{name}] not ready: {e}")
            continue
        except Exception as e:  # alignment / shape errors from mutation_scan
            failed += 1
            print(f"[{name}] FAILED: {e}")
            continue
        derived += 1
        print(f"[{name}] -> {csv_path}")

    print(f"\nDerived {derived}, skipped {skipped} (logits not ready), "
          f"failed {failed}.")
    return 1 if failed else 0


# --- view -----------------------------------------------------------------


def _cmd_view(args) -> int:
    out_dir = Path(args.out_dir)
    csv_path = satmut.matrix_csv(out_dir, args.name, args.size)
    if not csv_path.is_file():
        raise SystemExit(
            f"No mutation_scan.csv for {args.name!r}/{args.size} at {csv_path}. "
            f"Run `derive` first (needs the job's logits.npy).")

    wt_seq, aa_order, matrix = satmut.load_matrix(csv_path)
    summary = mutation_scan.summarize_scan(wt_seq, matrix, top_n=args.top_n,
                                           aa_order="".join(aa_order))

    if args.json:
        print(json.dumps({
            "name": args.name,
            "size": args.size,
            "csv": str(csv_path),
            "wt_seq": wt_seq,
            "summary": summary,
        }, indent=2, default=str))
        return 0

    print(f"Scan: {args.name}  size {args.size}")
    print(f"CSV:  {csv_path}")
    print(f"Length: {summary['length']} aa | "
          f"mean sensitivity (mean non-WT LLR): {summary['mean_sensitivity']:.3f}")
    print("\nLLR < 0 = model disfavours the substitution vs wild-type; "
          "> 0 = favours it.")

    print(f"\nLeast tolerant positions (most sensitive; lowest mean LLR):")
    for p in summary["least_tolerant_positions"]:
        print(f"  {p['wt_aa']}{p['position']:<5} sensitivity {p['sensitivity']:+.3f}")

    print(f"\nMost tolerant positions (highest mean LLR):")
    for p in summary["most_tolerant_positions"]:
        print(f"  {p['wt_aa']}{p['position']:<5} sensitivity {p['sensitivity']:+.3f}")

    print(f"\nTop favoured single substitutions (highest LLR):")
    for s in summary["top_substitutions"]:
        print(f"  {s['wt_aa']}{s['position']}{s['mut_aa']:<3} LLR {s['llr']:+.3f}")
    return 0


# --- entry ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="satmut_cli",
        description="Zero-shot saturation-mutagenesis scanning with ESM-C "
                    "(LLR of every single-point substitution vs wild-type).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser(
        "submit", help="Launch GPU scan job(s) producing logits.npy "
                       "(one sequence or a directory of FASTAs).")
    sp.add_argument("--config", required=True, help="Pipeline config YAML "
                    "(container/cache/SLURM settings).")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--seq", help="A single wild-type sequence to scan.")
    g.add_argument("--fasta-dir", help="Directory of .fasta/.fa (one seq each) "
                   "to scan in batch.")
    sp.add_argument("--name", help="Name for the single --seq scan (default: query).")
    sp.add_argument("--size", choices=satmut.ESMC_SIZES, default="6B",
                    help="ESM-C model size (default: 6B).")
    sp.add_argument("--out-dir", required=True, help="Where scan outputs land.")
    sp.add_argument("--dry-run", action="store_true",
                    help="Print the sbatch command(s) without submitting.")
    sp.set_defaults(func=_cmd_submit)

    dp = sub.add_parser(
        "derive", help="Compute the LLR matrix (mutation_scan.csv) from finished "
                       "logits.npy. Pure NumPy / CPU.")
    dp.add_argument("--out-dir", required=True, help="Scan output dir.")
    dp.add_argument("--size", choices=satmut.ESMC_SIZES, default="6B",
                    help="ESM-C model size (default: 6B).")
    dg = dp.add_mutually_exclusive_group(required=True)
    dg.add_argument("--name", nargs="+", help="Scan name(s) to derive.")
    dg.add_argument("--all", action="store_true",
                    help="Derive every submitted scan for this size.")
    dp.add_argument("--seq", help="Wild-type sequence (single --name only); "
                    "default reads the FASTA written at submit time.")
    dp.set_defaults(func=_cmd_derive)

    vp = sub.add_parser(
        "view", help="Print the headline (most/least tolerant positions, top "
                     "substitutions) for a derived scan.")
    vp.add_argument("--out-dir", required=True, help="Scan output dir.")
    vp.add_argument("--name", required=True, help="Scan name to view.")
    vp.add_argument("--size", choices=satmut.ESMC_SIZES, default="6B",
                    help="ESM-C model size (default: 6B).")
    vp.add_argument("--top-n", type=int, default=10,
                    help="How many positions/substitutions to list (default: 10).")
    vp.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of text.")
    vp.set_defaults(func=_cmd_view)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
