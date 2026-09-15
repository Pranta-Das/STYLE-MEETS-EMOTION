#!/bin/bash
# Experiment 1: does style geometry strength trade off against emotion legibility?
#
# The 5x5x7 grid showed the two styles with the most non-human FACE GEOMETRY
# (gothic doll, caricature) scoring 25.7% / 20.0% against 40-43% for the three
# structurally-normal styles. Those two are re-rendered here at half geometry
# strength, everything else identical, so the comparison is paired per cell.
set -e
OUT_DIR="${1:-exps/images/grid55_geo05}"
GEO="${2:-0.5}"

CONTENT=("assets/sample_input/cluo.jpg" "assets/sample_input/james.png"
         "original-image/00028.jpg" "original-image/00052.jpg" "original-image/00062.jpg")
# indices kept as in the 5x5 grid (s3, s4) so cells line up for the paired test
declare -A STYLES=(
  [3]="style-image/002_345_899_4k_nikita-orlov-portarait-womanblack_00.png"
  [4]="style-image/003_832_838_4k_hany-abbas-sayed-ragab4.jpg"
)
EMOTIONS="anger disgust fear happy neutral sad surprise"
PYTHON_BIN="${PYTHON_BIN:-./lam_env/bin/python}"
export PYTHONPATH="$PYTHONPATH:."

N=0; TOTAL=$(( ${#CONTENT[@]} * ${#STYLES[@]} * 7 ))
for ci in "${!CONTENT[@]}"; do
  for si in 3 4; do
    for EMOTION in $EMOTIONS; do
      N=$((N+1))
      IMG="${CONTENT[$ci]}"; CID=$(basename "$IMG"); CID="${CID%.*}"
      DUMP="${OUT_DIR}/c${ci}_s${si}/${EMOTION}"
      [ -f "${DUMP}/${CID}/stylized_preview.png" ] && { echo "[$N/$TOTAL] skip"; continue; }
      echo "[$N/$TOTAL] c${ci}($CID) s${si} ${EMOTION} geo=${GEO}"
      CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m lam.launch infer.lam \
          --config configs/inference/lam-20k-8gpu.yaml \
          model_name="exps/train_lam/style_appearance_e2e_v8/model_final.pt" \
          image_input="$IMG" \
          motion_seqs_dir="assets/sample_motion/export/Look_In_My_Eyes/" \
          export_video=false export_mesh=true test_sample=true \
          style_image_path="${STYLES[$si]}" \
          style_optimize_color=true style_optimize_steps=100 style_optimize_num_views=4 \
          style_geometry_strength="$GEO" style_track_geometry=true \
          emotion_class="$EMOTION" emotion_optimize=true \
          emotion_safety=true emotion_final_symmetry=0.5 \
          seed=42 save_img=false \
          image_dump="$DUMP" video_dump="${DUMP}_v" > /dev/null 2>&1 || echo "    FAILED"
    done
  done
done
echo "=== ${OUT_DIR}: $(find "$OUT_DIR" -name stylized_preview.png | wc -l)/${TOTAL} ==="
