#!/bin/bash
# 5 content faces x 5 styles x 7 emotions = 175 renders, for one contact sheet.
#
# Content is 5 real photographs varied in age, sex and skin tone. The 5 styles
# were chosen for maximally different palettes (grey painterly / flat orange /
# neon blue / dark gothic pale / warm yellow caricature) so that a stylisation
# that failed to transfer is obvious by eye rather than needing a metric.
#
# Safe to re-run: finished renders are skipped.
set -e

OUT_DIR="${1:-exps/images/grid55}"

CONTENT=(
  "assets/sample_input/cluo.jpg"
  "assets/sample_input/james.png"
  "original-image/00028.jpg"
  "original-image/00052.jpg"
  "original-image/00062.jpg"
)
STYLES=(
  "assets/sample_input/pop.png"
  "style-image/000_256_505_4k_mark-makovey-p-029_00.png"
  "style-image/001_339_379_4k_irakli-nadar-jinxartstation_00.png"
  "style-image/002_345_899_4k_nikita-orlov-portarait-womanblack_00.png"
  "style-image/003_832_838_4k_hany-abbas-sayed-ragab4.jpg"
)
EMOTIONS="anger disgust fear happy neutral sad surprise"

for f in "${CONTENT[@]}" "${STYLES[@]}"; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

MODEL_NAME="exps/train_lam/style_appearance_e2e_v8/model_final.pt"
MOTION_SEQS_DIR="assets/sample_motion/export/Look_In_My_Eyes/"
PYTHON_BIN="${PYTHON_BIN:-./lam_env/bin/python}"
export PYTHONPATH="$PYTHONPATH:."

N=0; TOTAL=$(( ${#CONTENT[@]} * ${#STYLES[@]} * 7 ))
for ci in "${!CONTENT[@]}"; do
  for si in "${!STYLES[@]}"; do
    for EMOTION in $EMOTIONS; do
      N=$((N+1))
      IMG="${CONTENT[$ci]}"; CID=$(basename "$IMG"); CID="${CID%.*}"
      DUMP="${OUT_DIR}/c${ci}_s${si}/${EMOTION}"
      if [ -f "${DUMP}/${CID}/stylized_preview.png" ]; then
          echo "[$N/$TOTAL] skip c${ci} s${si} ${EMOTION}"; continue
      fi
      echo "[$N/$TOTAL] c${ci}($CID) s${si} ${EMOTION}"
      CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m lam.launch infer.lam \
          --config configs/inference/lam-20k-8gpu.yaml \
          model_name="$MODEL_NAME" \
          image_input="$IMG" \
          motion_seqs_dir="$MOTION_SEQS_DIR" \
          export_video=false export_mesh=true test_sample=true \
          style_image_path="${STYLES[$si]}" \
          style_optimize_color=true style_optimize_steps=100 style_optimize_num_views=4 \
          style_geometry_strength=1.0 style_track_geometry=true \
          emotion_class="$EMOTION" emotion_optimize=true \
          emotion_safety=true emotion_final_symmetry=0.5 \
          seed=42 save_img=false \
          image_dump="$DUMP" video_dump="${DUMP}_v" \
          > /dev/null 2>&1 || echo "    FAILED"
    done
  done
done

echo
echo "=== ${OUT_DIR}: $(find "$OUT_DIR" -name stylized_preview.png | wc -l)/${TOTAL} renders ==="
