#!/usr/bin/env python3
"""
One-shot fix: rewrite the `msa:` line in each YAML to an absolute path.

Boltz resolves a relative `msa: msa/foo.a3m` against the YAML's directory.
When the YAML is symlinked into boltz_chunks/chunk_N/, that resolution
points inside the chunk dir (no `msa/` subdir there) and Boltz fails with
`FileNotFoundError: MSA file msa/foo.a3m not found.`

Fix: rewrite each YAML's `msa:` to point to the absolute a3m path inside
the sequence directory. Idempotent — already-absolute paths and `msa: empty`
are left untouched.

Usage:
    python scripts/fix_yaml_msa_paths.py /path/to/sequences/

Walks <sequences_dir>/seq_*/, locates the a3m under msa/, and rewrites the
YAML's `msa:` line to its absolute path.
"""

import re
import sys
from pathlib import Path


# Match `msa: <value>` (with optional surrounding whitespace and quotes)
_MSA_RE = re.compile(r'^(\s*msa:\s*)(.+?)\s*$', re.MULTILINE)


def find_a3m(seq_dir: Path) -> Path | None:
    """Locate the (single) a3m for this sequence dir, if any."""
    msa_dir = seq_dir / "msa"
    if not msa_dir.is_dir():
        return None
    a3ms = sorted(msa_dir.glob("*.a3m"))
    return a3ms[0] if a3ms else None


def rewrite_yaml_msa(yaml_path: Path, abs_a3m: Path) -> str:
    """Rewrite the YAML's `msa:` line to abs_a3m. Returns 'fixed', 'ok', or 'skip'."""
    text = yaml_path.read_text()
    m = _MSA_RE.search(text)
    if not m:
        return "skip"  # no msa line at all
    current = m.group(2).strip().strip('"').strip("'")

    # Leave `empty` alone
    if current == "empty":
        return "ok"

    # Already absolute and correct? leave alone
    new_value = str(abs_a3m)
    if current == new_value:
        return "ok"

    new_text = _MSA_RE.sub(lambda mm: f"{mm.group(1)}{new_value}", text, count=1)
    yaml_path.write_text(new_text)
    return "fixed"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    seq_root = Path(sys.argv[1])
    if not seq_root.is_dir():
        print(f"ERROR: not a directory: {seq_root}", file=sys.stderr)
        sys.exit(1)

    fixed = ok = skipped = errors = 0
    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        yamls = list(seq_dir.glob("*.yaml"))
        if not yamls:
            continue

        a3m = find_a3m(seq_dir)
        if a3m is None:
            # No a3m: only safe if YAML uses `msa: empty`. Don't touch it.
            skipped += 1
            continue
        abs_a3m = a3m.resolve()

        for y in yamls:
            try:
                status = rewrite_yaml_msa(y, abs_a3m)
            except Exception as e:
                print(f"  ERROR: {y}: {e}", file=sys.stderr)
                errors += 1
                continue
            if status == "fixed":
                print(f"  fixed: {y} -> msa: {abs_a3m}")
                fixed += 1
            elif status == "ok":
                ok += 1
            else:
                skipped += 1

    print(f"\nDone. Fixed: {fixed}, already correct: {ok}, skipped: {skipped}, errors: {errors}")


if __name__ == "__main__":
    main()
