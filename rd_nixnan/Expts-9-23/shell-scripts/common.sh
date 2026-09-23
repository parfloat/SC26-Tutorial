#!/usr/bin/env bash
# common.sh -- shared paths/vars for the Expts-9-23 sampling+histogram experiment.
# Sourced by every script in this directory; not meant to be run directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                 # Expts-9-23/
RD_NIXNAN_DIR="$(cd "$EXPT_DIR/.." && pwd)"              # rd_nixnan/
REPO_ROOT="$(cd "$RD_NIXNAN_DIR/.." && pwd)"             # SC26-Tutorial/

NIXNAN_SO="${NIXNAN_SO:-$REPO_ROOT/external/nixnan/nixnan.so}"
EXE="${EXE:-$RD_NIXNAN_DIR/rd_nixnan}"
LOGS_DIR="$EXPT_DIR/Logs"

if [[ ! -f "$NIXNAN_SO" ]]; then
  echo "ERROR: nixnan.so not found at $NIXNAN_SO -- run shell-scripts/00_build.sh first." >&2
  exit 1
fi
if [[ ! -x "$EXE" ]]; then
  echo "ERROR: rd_nixnan executable not found at $EXE -- run shell-scripts/00_build.sh first." >&2
  exit 1
fi

mkdir -p "$LOGS_DIR"
