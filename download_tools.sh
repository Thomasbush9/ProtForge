#!/bin/bash

set -eu

DIR_NAME="${0:-}"
TOOLS=("$@")

echo "Downloading ${TOOLS[*]}"

for tool in "${TOOLS[@]}"; do
  case "$tool" in
  boltz)
    echo "boltz"
    ;;
  es)
    echo "es"
    git clone "es repo"
    ;;
  esm)
    echo "esm"
    ;;
  *)
    echo "tool not supported?"
    ;;
  esac
done
