#!/bin/bash
# Out-of-distribution style test. The seven styles used so far are all AAHQ,
# the same distribution the style adapter was trained on, so "does it generalise
# to an unseen style" had no evidence. These four are caricatures with
# deliberately non-human proportions (inflated cheeks, widened smile, exaggerated
# brow) -- which also directly stresses the finding that FLAME tracking projects
# a style onto the nearest plausible HUMAN head before the blend sees it.
set -e
OUT="${1:-exps/images/oodstyle}"
STYLES=(style_image.png style_image1.png style_image2.png style_image3.png)
# same 5 identities as the unstyled baseline already on disk, so every cell is paired
CONTENT=($(ls new_input_image/* | sort | head -5))
EMOTIONS="anger disgust fear happy neutral sad surprise"
export PYTHONPATH="$PYTHONPATH:."
N=0; TOT=$(( ${#STYLES[@]} * ${#CONTENT[@]} * 7 ))
for si in "${!STYLES[@]}"; do
  echo "--- ood style o$si : ${STYLES[$si]} ---"
  for IMG in "${CONTENT[@]}"; do
    CID=$(basename "$IMG"); CID="${CID%.*}"
    for E in $EMOTIONS; do
      N=$((N+1)); D="$OUT/o$si/$E"
      [ -f "$D/$CID/stylized_preview.png" ] && { echo "[$N/$TOT] skip"; continue; }
      echo "[$N/$TOT] o$si $CID $E"
      CUDA_VISIBLE_DEVICES=0 ./lam_env/bin/python -m lam.launch infer.lam \
        --config configs/inference/lam-20k-8gpu.yaml \
        model_name="exps/train_lam/style_appearance_e2e_v8/model_final.pt" \
        image_input="$IMG" motion_seqs_dir="assets/sample_motion/export/Look_In_My_Eyes/" \
        export_video=false export_mesh=true test_sample=true \
        style_image_path="${STYLES[$si]}" \
        style_optimize_color=true style_optimize_steps=100 style_optimize_num_views=4 \
        style_geometry_strength=1.0 style_track_geometry=true \
        emotion_class="$E" emotion_optimize=true \
        emotion_safety=true emotion_final_symmetry=0.5 \
        seed=42 save_img=false image_dump="$D" video_dump="${D}_v" >/dev/null 2>&1 || echo "  FAILED"
    done
  done
  echo "=== o$si done: $(find "$OUT/o$si" -name stylized_preview.png|wc -l)/35 ==="
done
echo "### OOD DONE $(find "$OUT" -name stylized_preview.png|wc -l)/$TOT"
