#!/bin/bash
# 18 inputs x 20 styles x 7 emotions (2520 styled) + 18 x 7 unstyled baseline.
#
# ORDER MATTERS on a ~20h job: the unstyled baseline runs FIRST (both tables
# need it), then styles are completed one at a time across ALL contents and
# emotions. So at any point you can stop and still have a balanced, analysable
# subset -- N complete styles over every content -- rather than a few contents
# done and the rest missing. Re-running skips finished renders.
set -e
IN_DIR="new_input_image"
ST_DIR="new-style-image"
OUT="exps/images/bigrun"
EMOTIONS="anger disgust fear happy neutral sad surprise"
export PYTHONPATH="$PYTHONPATH:."

mapfile -t CONTENT < <(ls "$IN_DIR"/* | sort)
mapfile -t ALL_STYLES < <(ls "$ST_DIR"/* | sort)
# Only the first NUM_STYLES are used. 7 styles x 18 contents x 7 emotions = 882
# styled renders is already far more statistical power than anything else in
# this project; the remaining 13 styles would buy a wider spread of styles for
# the adoption table, not tighter error bars, at ~12 more GPU hours.
NUM_STYLES="${NUM_STYLES:-7}"
STYLES=("${ALL_STYLES[@]:0:$NUM_STYLES}")
mkdir -p "$OUT"
: > "$OUT/manifest.txt"
{ echo "# content images"; for i in "${!CONTENT[@]}"; do echo "c$i ${CONTENT[$i]}"; done
  echo "# style images (first $NUM_STYLES of ${#ALL_STYLES[@]})"
  for i in "${!STYLES[@]}";  do echo "s$i ${STYLES[$i]}";  done
  echo "# emotion_optimize_project (PGD) = default true"; } >> "$OUT/manifest.txt"

run () {  # $1=image $2=emotion $3=dump $4=style(optional)
    local args=()
    [ -n "$4" ] && args=(style_image_path="$4" style_optimize_color=true
        style_optimize_steps=100 style_optimize_num_views=4
        style_geometry_strength=1.0 style_track_geometry=true)
    CUDA_VISIBLE_DEVICES=0 ./lam_env/bin/python -m lam.launch infer.lam \
        --config configs/inference/lam-20k-8gpu.yaml \
        model_name="exps/train_lam/style_appearance_e2e_v8/model_final.pt" \
        image_input="$1" motion_seqs_dir="assets/sample_motion/export/Look_In_My_Eyes/" \
        export_video=false export_mesh=true test_sample=true \
        "${args[@]}" \
        emotion_class="$2" emotion_optimize=true \
        emotion_safety=true emotion_final_symmetry=0.5 \
        seed=42 save_img=false \
        image_dump="$3" video_dump="${3}_v" > /dev/null 2>&1 || echo "    FAILED $1 $2 $4"
}

echo "### PHASE 1/2: unstyled baseline (${#CONTENT[@]} x 7 = $(( ${#CONTENT[@]} * 7 )))"
N=0
for IMG in "${CONTENT[@]}"; do
    CID=$(basename "$IMG"); CID="${CID%.*}"
    for E in $EMOTIONS; do
        N=$((N+1)); D="$OUT/unstyled/$E"
        [ -f "$D/$CID/stylized_preview.png" ] && continue
        echo "[unstyled $N/$(( ${#CONTENT[@]} * 7 ))] $CID $E"
        run "$IMG" "$E" "$D"
    done
done

TOT=$(( ${#CONTENT[@]} * ${#STYLES[@]} * 7 ))
echo "### PHASE 2/2: styled (${#CONTENT[@]} x ${#STYLES[@]} x 7 = $TOT)"
N=0
for si in "${!STYLES[@]}"; do
    echo "--- style s$si : $(basename "${STYLES[$si]}") ---"
    for IMG in "${CONTENT[@]}"; do
        CID=$(basename "$IMG"); CID="${CID%.*}"
        for E in $EMOTIONS; do
            N=$((N+1)); D="$OUT/styled/s$si/$E"
            [ -f "$D/$CID/stylized_preview.png" ] && continue
            echo "[styled $N/$TOT] s$si $CID $E"
            run "$IMG" "$E" "$D" "${STYLES[$si]}"
        done
    done
    echo "=== style s$si complete: $(find "$OUT/styled/s$si" -name stylized_preview.png|wc -l)/126 ==="
done
echo "### DONE  unstyled $(find "$OUT/unstyled" -name stylized_preview.png|wc -l)  styled $(find "$OUT/styled" -name stylized_preview.png|wc -l)"
