#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs .cache/matplotlib

export PYTHONPATH=.
export MPLCONFIGDIR="$PWD/.cache/matplotlib"

nohup lam_env/bin/python -u lam/launch.py train.lam \
  --config configs/train/lam_style_stage4_geometry_adapter_aahq_stronger.yaml \
  > logs/stage4_geometry_adapter_aahq_stronger.log 2>&1 &

echo $! > run_stage4_geometry_adapter_aahq_stronger.pid
echo "Stronger Stage 4 geometry adapter training started."
echo "PID: $(cat run_stage4_geometry_adapter_aahq_stronger.pid)"
echo "Log: logs/stage4_geometry_adapter_aahq_stronger.log"
