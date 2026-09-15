#!/bin/bash
# Three-stage LAM style pipeline.
# Stage 1: natural-face style autoencoder.
# Stage 2: AAHQ artistic appearance fine-tuning.
# Stage 3: FLAME geometry reference bank for full LAM inference.

set -e

WORKSPACE="/media/uvll/8d1366b3-c06e-4683-8aee-cf9759a4008c/LAM13"
cd "$WORKSPACE"

export PYTHONPATH=.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORKSPACE/.cache/matplotlib}"
mkdir -p logs "$MPLCONFIGDIR"

PYTHON="${PYTHON:-./lam_env/bin/python}"

STAGE1_CKPT="exps/train_lam/stage1_celeba/style_autoencoder_final.pt"
STAGE2_CKPT="exps/train_lam/stage2_aahq/style_autoencoder_final.pt"

echo "=================================================="
echo "Stage 1: Natural Face Appearance Pretraining"
echo "=================================================="
if [ -f "$STAGE1_CKPT" ]; then
    echo "Skipping Stage 1; found $STAGE1_CKPT"
else
    "$PYTHON" lam/launch.py train.lam --config configs/train/lam_style_stage1_celeba.yaml
fi

ALIGNED_COUNT=$(find ffhq-dataset/aahq-dataset/aligned -maxdepth 1 -type f -iname '*.png' 2>/dev/null | wc -l)
if [ "$ALIGNED_COUNT" -lt 1000 ]; then
    echo "Need at least 1000 aligned AAHQ images for Stage 2; found $ALIGNED_COUNT"
    echo "Run: cd ffhq-dataset/aahq-dataset && ../../lam_env/bin/python face_alignment.py --json_dir AAHQ-dataset.json --raw_dir raw --save_dir aligned --n_worker 8"
    exit 1
fi

echo "=================================================="
echo "Stage 2: AAHQ Artistic Appearance Fine-tuning"
echo "=================================================="
if [ -f "$STAGE2_CKPT" ]; then
    echo "Skipping Stage 2; found $STAGE2_CKPT"
else
    "$PYTHON" lam/launch.py train.lam --config configs/train/lam_style_stage2_aahq.yaml
fi

echo "=================================================="
echo "Stage 3: AAHQ Geometry Reference Bank"
echo "=================================================="
"$PYTHON" tools/track_stage3_geometry_refs.py --config configs/train/lam_style_stage3_geometry_aahq.yaml
"$PYTHON" tools/build_style_geometry_bank.py --config configs/train/lam_style_stage3_geometry_aahq.yaml

echo "=================================================="
echo "Three-stage pipeline complete"
echo "=================================================="
echo "Stage 1: exps/train_lam/stage1_celeba/"
echo "Stage 2: exps/train_lam/stage2_aahq/"
echo "Stage 3: exps/train_lam/stage3_geometry_aahq/"
