
set -e

INPUT_IMAGE="${1:?usage: ./run_emotion_style_check.sh <input_image> <style_image> [out_dir]}"
STYLE_IMAGE="${2:?usage: ./run_emotion_style_check.sh <input_image> <style_image> [out_dir]}"
OUT_DIR="${3:-my_emotion_test}"

# Fail here rather than 40 minutes of silent no-ops later. A mistyped path used
# to glob to zero images and exit 0 with nothing rendered.
[ -f "$INPUT_IMAGE" ] || { echo "no such input image: $INPUT_IMAGE" >&2; exit 1; }
[ -f "$STYLE_IMAGE" ] || { echo "no such style image: $STYLE_IMAGE" >&2; exit 1; }

MODEL_NAME="exps/train_lam/style_appearance_e2e_v8/model_final.pt"
MOTION_SEQS_DIR="assets/sample_motion/export/Look_In_My_Eyes/"
EMOTIONS="${EMOTIONS:-anger disgust fear happy neutral sad surprise}"
STEPS="${STEPS:-100}"
FRAMES="${FRAMES:-false}"
FULL="${FULL:-false}"
if [ "$FULL" = "true" ]; then TEST_SAMPLE=false; else TEST_SAMPLE=true; fi
PYTHON_BIN="${PYTHON_BIN:-./lam_env/bin/python}"
export PYTHONPATH="$PYTHONPATH:."

CID=$(basename "$INPUT_IMAGE"); CID="${CID%.*}"
mkdir -p "$OUT_DIR"

for EMOTION in $EMOTIONS; do
    echo "=== generating: $EMOTION ==="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m lam.launch infer.lam \
        --config configs/inference/lam-20k-8gpu.yaml \
        model_name="$MODEL_NAME" \
        image_input="$INPUT_IMAGE" \
        motion_seqs_dir="$MOTION_SEQS_DIR" \
        export_video=true export_mesh=false test_sample="$TEST_SAMPLE" \
        style_image_path="$STYLE_IMAGE" \
        style_optimize_color=true style_optimize_steps=100 style_optimize_num_views=4 \
        style_geometry_strength=1.0 style_track_geometry=true \
        emotion_class="$EMOTION" \
        emotion_optimize=true emotion_optimize_steps="$STEPS" \
        emotion_safety=true emotion_final_symmetry=0.5 \
        seed=42 save_img="$FRAMES" \
        image_dump="$OUT_DIR/$EMOTION" \
        video_dump="$OUT_DIR/${EMOTION}_v"
done

echo
echo "=== results ==="
for EMOTION in $EMOTIONS; do
    for F in "$OUT_DIR/$EMOTION/$CID/stylized_preview.png" \
             "$OUT_DIR/${EMOTION}_v/${CID}_audio.mp4"; do
        if [ -f "$F" ]; then echo "  OK       $F"; else echo "  MISSING  $F"; fi
    done
done
