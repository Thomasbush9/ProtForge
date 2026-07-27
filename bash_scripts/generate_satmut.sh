#!/bin/bash
set -euo pipefail

# Generate one FASTA per single-point mutant of a wild-type sequence.
#
# The Saturation Mutagenesis tab scores every substitution from a single ESM-C
# forward pass on the wild type, so it never embeds the mutants themselves.
# Use this when you want a real per-mutant embedding or structure: it writes the
# mutant sequences as pipeline input, ready for input.fasta_dir.
#
# Usage:
#   ./generate_satmut.sh --input wt.fasta --output-dir muts/ [OPTIONS]
#
# Required:
#   --input PATH        Wild-type sequence (.fasta/.a3m or .yaml), single entry
#   --output-dir PATH   Directory for the mutant FASTAs (created if absent)
#
# Optional:
#   --positions SPEC    1-based subset, e.g. '1-50,73'. Default: every position
#   --alphabet AAS      Substitutions to try. Default: the 20 standard AAs
#   --name-prefix STR   Prepended to file names, e.g. 'GFP_' -> GFP_M1A.fasta
#   --include-wt        Also emit the unmutated sequence as <prefix>WT.fasta
#   --dry-run           Report the file count without writing anything
#
# Size warning: a length-L sequence gives L x 19 files — 4522 for a 238-residue
# protein. Run --dry-run first if you are unsure what you are about to make.
#
# Output also includes index.csv (name, mutation, position, wt_aa, mut_aa,
# sequence) for joining predictions back to variants.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ $# -eq 0 ]]; then
  sed -n '4,29p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 1
fi

exec python "$REPO_ROOT/utils/generate_satmut.py" "$@"
