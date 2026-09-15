#!/bin/bash
# Re-render every "Ours" cell in the three paper figures, front-facing and at
# 1024x1024.  Baseline cells (EDTalk/EAT/StyleCLIP/JoJoGAN/StyTR-2/AdaAttN) are
# other systems' outputs and are left untouched.
cd "$(dirname "$0")"
OUT=exps/images/frontal_hi
MODEL="exps/train_lam/style_appearance_e2e_v8/model_final.pt"
EMO=(anger disgust fear happy neutral sad surprise)
COMP=004_618_397_4k_masayoshi-shinohara-1701sn-1920
NEDKO=new_input_image/001_162_059_4k_nedko-ivanov-digital-miranda-kerr-by-vannenov-d75s0z2.jpg
BIGNON=new-style-image/000_383_224_4k_thomas-bignon-20.jpg

run () {  # $1=image $2=emotion $3=dump $4=style(optional)
    local args=()
    [ -n "$4" ] && args=(style_image_path="$4" style_optimize_color=true
        style_optimize_steps=100 style_optimize_num_views=4
        style_geometry_strength=1.0 style_track_geometry=true)
    CUDA_VISIBLE_DEVICES=0 ./lam_env/bin/python -m lam.launch infer.lam \
        --config configs/inference/lam-20k-8gpu.yaml \
        model_name="$MODEL" \
        image_input="$1" motion_seqs_dir="assets/sample_motion/export/Look_In_My_Eyes/" \
        export_video=false export_mesh=true test_sample=true \
        render_scale=2.0 frontalize=true frontalize_pitch=0.14 \
        "${args[@]}" \
        emotion_class="$2" emotion_optimize=true \
        emotion_safety=true emotion_final_symmetry=0.5 \
        seed=42 save_img=false \
        image_dump="$3" video_dump="${3}_v" > /dev/null 2>&1 \
        || echo "    FAILED $1 $2 $4"
}

N=0; T=29
echo "### composition figure: 3 rows x 7 emotions"
for E in "${EMO[@]}"; do
    run "new_input_image/${COMP}.jpg" "$E" "$OUT/unstyled/$E"
    N=$((N+1)); echo "  [$N/$T] unstyled $E"
done
for E in "${EMO[@]}"; do
    run "new_input_image/${COMP}.jpg" "$E" "$OUT/nedko/$E" "$NEDKO"
    N=$((N+1)); echo "  [$N/$T] nedko $E"
done
for E in "${EMO[@]}"; do
    run "new_input_image/${COMP}.jpg" "$E" "$OUT/bignon/$E" "$BIGNON"
    N=$((N+1)); echo "  [$N/$T] bignon $E"
done

echo "### stylisation figure: 1 cell"
run "$NEDKO" neutral "$OUT/stylecomp/neutral" "$BIGNON"
N=$((N+1)); echo "  [$N/$T] stylecomp neutral"

echo "### emotion figure: identity 00052, unstyled x 7"
for E in "${EMO[@]}"; do
    run "original-image/00052.jpg" "$E" "$OUT/emo00052/$E"
    N=$((N+1)); echo "  [$N/$T] emo00052 $E"
done
echo "### ALL DONE"
