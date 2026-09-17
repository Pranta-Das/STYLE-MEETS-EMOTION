
set -e

IMAGE_INPUT="${1:?usage: ./run_7emotions.sh <input_image> <style_image> [output_dir] [gs_mouth_symmetry]}"
STYLE_IMAGE="${2:?usage: ./run_7emotions.sh <input_image> <style_image> [output_dir] [gs_mouth_symmetry]}"
OUT_DIR="${3:-check_7emotions}"
# 4th arg: damps the lateral asymmetry the Gaussian decoder invents in its own
# per-point xyz offsets, in the mouth region only (0.0 = off, 1.0 = fully
# symmetric). Off by default until the 14-identity validation clears it.
GS_SYM="${4:-0.0}"

MODEL_NAME="exps/train_lam/style_appearance_e2e_v8/model_final.pt"
MOTION_SEQS_DIR="assets/sample_motion/export/Look_In_My_Eyes/"
EMOTIONS="anger disgust fear happy neutral sad surprise"

PYTHON_BIN="${PYTHON_BIN:-./lam_env/bin/python}"
EMOTION_OPTIMIZE="${EMOTION_OPTIMIZE:-true}"
SEED="${SEED:-42}"
export PYTHONPATH="$PYTHONPATH:."

for EMOTION in $EMOTIONS; do
    DUMP_DIR="${OUT_DIR}/${EMOTION}"
    echo "=== ${EMOTION} ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m lam.launch infer.lam \
        --config configs/inference/lam-20k-8gpu.yaml \
        model_name="$MODEL_NAME" \
        image_input="$IMAGE_INPUT" \
        motion_seqs_dir="$MOTION_SEQS_DIR" \
        export_video=false export_mesh=true test_sample=true \
        style_image_path="$STYLE_IMAGE" \
        style_optimize_color=true style_optimize_steps=100 style_optimize_num_views=4 \
        style_geometry_strength=1.0 style_track_geometry=true \
        emotion_class="$EMOTION" emotion_optimize="$EMOTION_OPTIMIZE" \
        emotion_safety=true emotion_final_symmetry=0.5 \
        gs_mouth_symmetry="$GS_SYM" \
        seed="$SEED" \
        save_img=true \
        image_dump="$DUMP_DIR" video_dump="${DUMP_DIR}_v"
done

echo
echo "Done. Checking for actual output files:"
CONTENT_ID=$(basename "$IMAGE_INPUT")
CONTENT_ID="${CONTENT_ID%.*}"
for EMOTION in $EMOTIONS; do
    RESULT="${OUT_DIR}/${EMOTION}/${CONTENT_ID}/stylized_preview.png"
    if [ -f "$RESULT" ]; then
        echo "  OK   $RESULT"
    else
        echo "  MISSING  $RESULT"
    fi
done
