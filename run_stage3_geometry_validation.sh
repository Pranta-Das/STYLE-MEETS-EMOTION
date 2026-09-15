#!/bin/bash
# Run geometry-off vs geometry-on validation for the top Stage 3 references.

set -e

WORKSPACE="/media/uvll/8d1366b3-c06e-4683-8aee-cf9759a4008c/LAM13"
cd "$WORKSPACE"

export PYTHONPATH=.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORKSPACE/.cache/matplotlib}"
mkdir -p logs "$MPLCONFIGDIR"

PYTHON="${PYTHON:-./lam_env/bin/python}"
LIMIT="${1:-3}"

"$PYTHON" tools/run_stage3_geometry_validation.py \
  --config configs/train/lam_style_stage3_geometry_aahq.yaml \
  --limit "$LIMIT"
