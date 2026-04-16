#!/usr/bin/env python3
"""
One-shot fix: rewrite the first header line of each a3m to match its YAML's `id`.

Boltz uses the a3m's first header to populate chain_to_msa, then looks up
chain.id from the YAML. If they don't match, you get KeyError.

Usage:
    python scripts/fix_a3m_headers.py /path/to/sequences/

Walks <sequences_dir>/seq_*/, reads the YAML id, and rewrites the first
'>' line of the matching a3m to '>{yaml_id}'. Idempotent.
"""

import re
import sys
from pathlib import Path


def extract_yaml_id(yaml_path: Path) -> str | None:
    """Pull the protein id from a Boltz YAML without a YAML parser."""
    text = yaml_path.read_text()
    m = re.search(r'^\s*id:\s*"?([^"\n]+)"?\s*$', text, re.MULTILINE)
    return m.group(1).strip() if m else None


def rewrite_first_header(a3m_path: Path, new_id: str) -> bool:
    """Rewrite the first '>' line to '>{new_id}'. Returns True if changed."""
    lines = a3m_path.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(">"):
            new_line = f">{new_id}\n"
            if line == new_line:
                return False  # already correct
            lines[i] = new_line
            a3m_path.write_text("".join(lines))
            return True
    return False


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    seq_root = Path(sys.argv[1])
    if not seq_root.is_dir():
        print(f"ERROR: not a directory: {seq_root}", file=sys.stderr)
        sys.exit(1)

    fixed = skipped = errors = 0
    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        yamls = list(seq_dir.glob("*.yaml"))
        a3ms = list((seq_dir / "msa").glob("*.a3m")) if (seq_dir / "msa").is_dir() else []

        if not yamls or not a3ms:
            continue

        yaml_id = extract_yaml_id(yamls[0])
        if not yaml_id:
            print(f"  WARN: no id in {yamls[0]}", file=sys.stderr)
            errors += 1
            continue

        # Rewrite the first a3m (Boltz only uses the one referenced in the YAML)
        if rewrite_first_header(a3ms[0], yaml_id):
            print(f"  fixed: {a3ms[0]} -> >{yaml_id}")
            fixed += 1
        else:
            skipped += 1

    print(f"\nDone. Fixed: {fixed}, already correct: {skipped}, errors: {errors}")


if __name__ == "__main__":
    main()
