#!/bin/bash
# Two-stage LAM style training pipeline
# Stage 1: Train style autoencoder on CelebA-HQ (natural faces)
# Stage 2: Fine-tune on AAHQ (artistic styles)

set -e

WORKSPACE="/media/uvll/8d1366b3-c06e-4683-8aee-cf9759a4008c/LAM13"
cd "$WORKSPACE"

export PYTHONPATH=.
PYTHON="./lam_env/bin/python"

echo "=================================================="
echo "Stage 1: Training on CelebA-HQ (Natural Faces)"
echo "=================================================="
$PYTHON lam/launch.py train.lam --config configs/train/lam_style_stage1_celeba.yaml

echo ""
echo "Stage 1 completed. Model saved to: exps/train_lam/stage1_celeba/style_autoencoder_final.pt"
echo ""

# Check if aligned AAHQ images are ready
ALIGNED_COUNT=$(find ffhq-dataset/aahq-dataset/aligned -maxdepth 1 -type f -iname '*.png' 2>/dev/null | wc -l)
if [ "$ALIGNED_COUNT" -lt 100 ]; then
    echo "Warning: Only $ALIGNED_COUNT aligned AAHQ images found. Waiting for alignment to complete..."
    echo "Monitoring alignment progress..."
    while [ "$ALIGNED_COUNT" -lt 1000 ]; do
        sleep 30
        ALIGNED_COUNT=$(find ffhq-dataset/aahq-dataset/aligned -maxdepth 1 -type f -iname '*.png' 2>/dev/null | wc -l)
        echo "  Current aligned images: $ALIGNED_COUNT"
    done
    echo "Alignment completed with at least 1000 images"
fi

echo ""
echo "=================================================="
echo "Stage 2: Fine-tuning on AAHQ (Artistic Styles)"
echo "=================================================="
$PYTHON lam/launch.py train.lam --config configs/train/lam_style_stage2_aahq.yaml

echo ""
echo "=================================================="
echo "Training Pipeline Complete!"
echo "=================================================="
echo "Stage 1 output: exps/train_lam/stage1_celeba/"
echo "Stage 2 output: exps/train_lam/stage2_aahq/"
echo "=================================================="
