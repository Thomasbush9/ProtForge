#!/usr/bin/env python3
"""
Read one dotted key from a YAML config, print its value (for bash wrappers).

    python cfg_get.py config.yaml containers.boltz
    python cfg_get.py config.yaml containers.runtime auto   # with default

Exits non-zero (and prints nothing) if the key is missing and no default is
given, so callers can `|| die`. A literal "auto" for containers.runtime is
resolved to singularity/apptainer by the caller, not here.
"""
import sys

try:
    import yaml
except ImportError:
    sys.exit(
        "cfg_get: PyYAML not found. Activate the env you run snakemake in, "
        "or `pip install pyyaml` (it's in requirements-data.txt)."
    )


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: cfg_get.py CONFIG DOTTED.KEY [DEFAULT]", file=sys.stderr)
        return 2
    cfg_path, dotted = sys.argv[1], sys.argv[2]
    default = sys.argv[3] if len(sys.argv) > 3 else None

    with open(cfg_path) as f:
        node = yaml.safe_load(f)

    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            if default is not None:
                print(default)
                return 0
            print(f"cfg_get: missing key {dotted!r} in {cfg_path}", file=sys.stderr)
            return 1

    if node is None or node == "":
        if default is not None:
            print(default)
            return 0
        print(f"cfg_get: empty value for {dotted!r}", file=sys.stderr)
        return 1
    print(node)
    return 0


if __name__ == "__main__":
    sys.exit(main())
