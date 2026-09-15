#!/bin/bash
# Unstyled baseline for the 3 robotic heads, needed as the reference for colour
# adoption (a render is a Gaussian reconstruction on white, so the style-adoption
# metric must be measured against the pipeline's own unstyled output).
set -e
export PYTHONPATH="$PYTHONPATH:."
for ID in 113 115 116; do
  for E in anger disgust fear happy neutral sad surprise; do
    D="exps/images/robot_unstyled/$E"
    [ -f "$D/$ID/stylized_preview.png" ] && continue
    echo "[$ID $E]"
    CUDA_VISIBLE_DEVICES=0 ./lam_env/bin/python -m lam.launch infer.lam \
      --config configs/inference/lam-20k-8gpu.yaml \
      model_name="exps/train_lam/style_appearance_e2e_v8/model_final.pt" \
      image_input="figure/$ID.png" \
      motion_seqs_dir="assets/sample_motion/export/Look_In_My_Eyes/" \
      export_video=false export_mesh=true test_sample=true \
      emotion_class="$E" emotion_optimize=true \
      emotion_safety=true emotion_final_symmetry=0.5 \
      seed=42 save_img=false image_dump="$D" video_dump="${D}_v" >/dev/null 2>&1 || echo "  FAILED $ID $E"
  done
done
echo "### ROBOT UNSTYLED DONE $(find exps/images/robot_unstyled -name stylized_preview.png|wc -l)/21"
