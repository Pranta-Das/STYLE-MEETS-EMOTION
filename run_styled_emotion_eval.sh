#!/bin/bash
# Render the emotion evaluation set in one condition (styled or unstyled).
#
# The 14 identities are the same photographs used for the original unstyled
# table: sorted(original-image/*.jpg)[:14], recovered by FaceNet matching after
# the renamed copies were deleted (original_001 -> 00028.jpg, ... in order).
#
# BOTH arms must be re-rendered together. The unstyled numbers currently in
# EVALUATION_TABLES.md were produced on 2026-09-01, before the emotion adapter
# checkpoint, sharpening defaults and safety clamp changed on 2026-09-03, so
# they cannot be differenced against anything rendered on today's code.
#
# Usage:
#   ./run_styled_emotion_eval.sh <out_dir> [style_image]
#     style_image omitted  -> unstyled arm
#     style_image given    -> styled arm
set -e

OUT_DIR="${1:?usage: ./run_styled_emotion_eval.sh <out_dir> [style_image]}"
STYLE_IMAGE="${2:-}"

MODEL_NAME="exps/train_lam/style_appearance_e2e_v8/model_final.pt"
# Ablation switch. Look_In_My_Eyes has the 2nd-highest speech jaw variance
# (sd 0.0511) and 2nd-highest lateral jaw yaw (0.0182) of the 12 clips on
# disk -- both larger than the entire emotion jaw budget (open 0.04,
# lateral 0.01). Speeding_Scandal is the quietest (0.0116 / 0.0066).
MOTION_SEQS_DIR="${MOTION_SEQS_DIR:-assets/sample_motion/export/Look_In_My_Eyes/}"
EMOTIONS="anger disgust fear happy neutral sad surprise"
PYTHON_BIN="${PYTHON_BIN:-./lam_env/bin/python}"
# Ablation switch: EMOTION_SAFETY=false isolates the 09-03 clamp.
EMOTION_SAFETY="${EMOTION_SAFETY:-true}"
# Ablation switch: EMOTION_OPTIMIZE=false removes the ViT-driven test-time
# optimiser, whose objective is blind to anger/disgust (0% on real photos).
EMOTION_OPTIMIZE="${EMOTION_OPTIMIZE:-true}"
# Max jaw OPENING the emotion delta may add, in radians. Default 0.04.
# With projected gradient on, this is also the bound the optimiser searches
# inside, so raising it lets the optimiser buy classifier confidence with a
# wider mouth -- which is exactly what this sweep is testing.
MAX_JAW_OPEN="${MAX_JAW_OPEN:-0.04}"
export PYTHONPATH="$PYTHONPATH:."

IDENTITIES=$(ls original-image/*.jpg | sort | head -14)

STYLE_ARGS=()
if [ -n "$STYLE_IMAGE" ]; then
    STYLE_ARGS=(
        style_image_path="$STYLE_IMAGE"
        style_optimize_color=true style_optimize_steps=100 style_optimize_num_views=4
        style_geometry_strength=1.0 style_track_geometry=true
    )
fi

N=0
for IMG in $IDENTITIES; do
    for EMOTION in $EMOTIONS; do
        N=$((N+1))
        DUMP_DIR="${OUT_DIR}/${EMOTION}"
        CID=$(basename "$IMG"); CID="${CID%.*}"
        # skip work already done, so an interrupted batch can simply be re-run
        if [ -f "${DUMP_DIR}/${CID}/stylized_preview.png" ]; then
            echo "[$N/98] skip ${CID}/${EMOTION} (already rendered)"
            continue
        fi
        echo "[$N/98] ${CID} / ${EMOTION}"
        CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m lam.launch infer.lam \
            --config configs/inference/lam-20k-8gpu.yaml \
            model_name="$MODEL_NAME" \
            image_input="$IMG" \
            motion_seqs_dir="$MOTION_SEQS_DIR" \
            export_video=false export_mesh=true test_sample=true \
            "${STYLE_ARGS[@]}" \
            emotion_class="$EMOTION" emotion_optimize="$EMOTION_OPTIMIZE" \
            emotion_safety="$EMOTION_SAFETY" emotion_final_symmetry=0.5 \
            emotion_max_jaw_open_delta="$MAX_JAW_OPEN" \
            seed=42 save_img=false \
            image_dump="$DUMP_DIR" video_dump="${DUMP_DIR}_v" \
            > /dev/null 2>&1 || echo "    FAILED"
    done
done

echo
echo "=== ${OUT_DIR}: produced $(find "$OUT_DIR" -name 'stylized_preview.png' | wc -l)/98 renders ==="
