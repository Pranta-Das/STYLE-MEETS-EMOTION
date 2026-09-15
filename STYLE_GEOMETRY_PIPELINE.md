# LAM Face Stylization Pipeline

This document describes the pipeline as it actually works today. It supersedes the original
Stage 1-4 plan below in every way that matters for output quality: Stage 1/2
(`TinyStyleAutoEncoder`) trained a model disconnected from the real style path and is no longer
used; Stage 4 as originally written only trained geometry and left appearance dead. Both are kept
in this repo for reference but should not be used for new work.

## What actually carries style, end to end

`ModelLAM.style_adapter` (`ReferenceStyleAdapter` in `lam/models/modeling_lam.py`) has two parts:

- **`geometry_mlp`** — learned residual on top of an algebraic FLAME-shape blend
  (`_blend_style_shape_params`). Needs a tracked FLAME shape for the style image
  (`style_track_geometry=true`), no training required for the algebraic part.
- **`appearance_mlp`** — outputs a small global RGB gain+shift, applied to decoded Gaussian color
  only, *after* scale/opacity/rotation/xyz have already been computed from the (style-untouched)
  shared hidden features. This is deliberate: an earlier version modulated the shared feature
  vector before decode, which let "appearance" changes leak into Gaussian scale and blur every
  styled render. Isolating style to color, post-decode, removed that leak at the source.

Both are trained by rendering real (content, style) image pairs through the frozen pretrained
backbone (encoder/transformer/renderer never change) and backpropagating straight into these two
small heads — no disconnected side-model in between.

**Known-bad checkpoints, do not use:** `exps/train_lam/style_appearance_e2e_v2/model_final.pt` and
everything chained from it (`v4`, `v5`, `v6`, `v7`) — see "Cross-attention corruption bug" and
"appearance_mlp learns the wrong color direction" below.

Recommended checkpoint: `exps/train_lam/style_appearance_e2e_v8/model_final.pt` — first version
verified to shift color in the *correct* direction (cosine similarity 0.97 between the actual
RGB shift and the style image's target shift, vs. v2 which was nearly opposite). Still only a
bounded 1-epoch/300-step run and undertrained (~23% of the target shift magnitude, some uneven
shading) — a good base to continue training longer, not a finished result.

## Pipeline stages (current)

### Stage 1: FLAME-track a content subset

```bash
PYTHONPATH=. lam_env/bin/python tools/track_content_refs.py \
  --config configs/train/track_content_refs_celeba.yaml
```

Tracks a CelebA-HQ subset (default 300 images), producing
`exps/train_lam/content_refs/tracked_content_refs.json` — each entry's `canonical_flame_param.npz`
supplies the FLAME pose/shape and render camera needed to render that image through the model
during training.

### Stage 2: train `appearance_mlp`

```bash
PYTHONPATH=. lam_env/bin/python lam/launch.py train.lam \
  --config configs/train/lam_style_appearance_e2e_v2.yaml
```

Renders each (content, style) pair via `render_view_with_grad` in `lam/runners/train/lam.py` (a
differentiable equivalent of `ModelLAM.infer_single_view`, which is `@torch.no_grad()`-decorated
and therefore unusable for training) and backprops a Gatys-style content+style+AdaIN(+identity)
loss into `appearance_mlp` only. `configs/train/lam_style_appearance_e2e.yaml` (v1) is kept for
reference but produced a blurry appearance path under the old feature-space modulation — use `_v2`.

### Stage 3: geometry

Build/expand the geometry reference bank (tracks AAHQ style images, selects ones with a
meaningfully different FLAME shape from the source identity):

```bash
PYTHONPATH=. lam_env/bin/python tools/track_stage3_geometry_refs.py \
  --config configs/train/lam_style_stage3_geometry_aahq.yaml
PYTHONPATH=. lam_env/bin/python tools/build_style_geometry_bank.py \
  --config configs/train/lam_style_stage3_geometry_aahq.yaml
```

Train `geometry_mlp` (chain off the Stage 2 checkpoint so appearance carries over):

```bash
PYTHONPATH=. lam_env/bin/python lam/launch.py train.lam \
  --config configs/train/lam_style_geometry_adapter_v2.yaml
```

### Stage 4 (attempted, not recommended): per-point cross-attention appearance

`configs/train/lam_style_appearance_e2e_v3.yaml` / `_v3_continue.yaml` and the
`style_cross_attn` module in `ReferenceStyleAdapter` replace the single global RGB
transform with one derived per query-point via cross-attention against the style
image's spatial DINOv2 feature grid, aiming for spatially-varying style (different
tint on hair vs. skin vs. background) instead of one uniform tint.

Result: measurably *worse* than v2, not better. Comparing hair-vs-skin color
differentiation (L2 distance between average hair-region and skin-region RGB) on the
same test case: v2 = 19.5, v3 (900 steps) = 13.1, v3-continue (+600 steps, 1.5x style
loss weight) = 14.0 — i.e. less spatial variation than the simpler global version,
not more, and 600 additional steps with a stronger style-loss pull barely moved it.
Root cause hypothesized at the time: the training loss (content/style/AdaIN) is
computed on the whole rendered image with no spatial/regional structure, so gradient
descent has no region-specific signal to push per-point attention to specialize.
That's still true, but it turned out to be only half the story — see the next
section for what was actually broken underneath this comparison.

### Bug found later: cross-attention corruption affected every checkpoint, including "v2"

The "fall back to v2" recommendation above was never actually implemented in code —
`ReferenceStyleAdapter.forward()` kept routing every checkpoint's style features
through `style_cross_attn` unconditionally, including v2, which predates that module
and has no trained `style_cross_attn` weights in its checkpoint. `CrossAttnBlock` has
no residual connection (`forward()` returns the raw `nn.MultiheadAttention` output),
so loading v2 — for training OR plain inference — silently reinitialized
`style_cross_attn` to fresh, random PyTorch-default weights every single process
launch. `appearance_mlp` was therefore always fed a signal scrambled by an untrained,
non-deterministic attention layer, never the clean mean-pooled style vector it was
actually trained on. This means the v2-vs-v3 comparison above, and every appearance
training run chained from v2 (`v4`, `v5`, `v6`, `v7`), were all confounded by this
noise to some degree — none of those results are fully trustworthy as measurements
of the underlying learned weights.

Fixed by removing `style_cross_attn` entirely and restoring the direct broadcast of
`style_code = style_feats.mean(dim=1)` to every query point (`modeling_lam.py`,
`ReferenceStyleAdapter.forward()`) — exactly what v2 was actually trained on, so its
checkpoint still loads and now behaves deterministically.

### appearance_mlp learns the wrong color direction

Re-measuring v2 with the bug fixed (deterministic now) exposed a second, real
problem: v2's appearance_mlp shifts color the WRONG way. On the `status.png` test
case with `style_image1.png` (a warm/tan, dark-haired style): unstyled mean RGB is
`(195, 189, 180)`; the style image's mean is `(138, 124, 122)` (darker, warmer); v2's
styled output moves to `(228, 219, 216)` — brighter, the opposite direction. This
matches the user-reported symptom exactly ("not even taking the colour texture from
the style image"), now confirmed quantitatively rather than just visually.

Cause: `style_style_loss_weight` (Gram-matrix style loss) is calibrated against a raw
style-loss value that stayed ~0.0001–0.0008 across every training run's logs to
date — small enough to be dominated by content/pixel/identity loss regardless of the
weight multiplier. Gradient descent settled on "get brighter" as a shortcut that
reduces content/pixel loss on this training set rather than learning real per-style
color transfer. `style_adain_loss_weight` (direct channel mean/std matching) is
structurally what a single global affine color transform can actually satisfy, and
is a much more promising primary loss term — being tested in `v8` now that the
cross-attention bug no longer corrupts the conditioning signal it depends on.

**Verification lesson**: weight-magnitude growth (the diagnostic used for v4/v5/v6)
is necessary but not sufficient — it doesn't tell you the direction is correct. Check
`(styled_mean_rgb − unstyled_mean_rgb)` against `(style_mean_rgb − unstyled_mean_rgb)`
and confirm the sign/direction agrees, not just that the magnitude grew.

## Per-instance color optimization (recommended for actual use)

`appearance_mlp` (above) tries to learn one small network that generalizes a color
transform across many (content, style) pairs from limited training data -- in
practice (v2 through v8) this converges slowly and unreliably; see the "wrong color
direction" section above. For the real use case -- one content image, one style
image, one output -- `lam/stylization/color_optimize.py` skips generalization
entirely: it directly optimizes *this* reconstruction's per-point Gaussian color
against *this* style image's actual Gram/AdaIN statistics (classic Gatys-style
optimization, applied to 3D Gaussian color instead of 2D pixels), for a few hundred
steps, at inference time. Verified result (`musk.jpg` + a grayscale painting style):
saturation moved from 0.256 (input) to 0.056, matching the style's near-zero 0.001,
vs. 0.176 (barely moved) for the trained-network path alone. Geometry (jaw/face
shape) is untouched by this -- it still comes from the existing algebraic FLAME
blend + `geometry_mlp`, computed once and frozen before the color optimization runs.

Costs more per request (a few hundred optimization steps, ~2-3 min total including
tracking/video export, vs. a single network forward pass) but actually converges to
the requested style instead of hoping a network generalized. Supervising against
only one camera view let per-point colors overfit to that view and showed up as
speckled/rainbow noise once the animated head turned in later frames -- fixed by
sampling a handful of views spread across the motion sequence each step
(`style_optimize_num_views`, default 6) so the optimized color has to stay
consistent across angles.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 lam_env/bin/python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=exps/train_lam/style_appearance_e2e_v8/model_final.pt \
  image_input=<YOUR_INPUT_IMAGE> \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  export_video=true export_mesh=false \
  style_image_path=<YOUR_STYLE_IMAGE> \
  style_optimize_color=true \
  style_optimize_steps=500 \
  style_optimize_num_views=6 \
  style_optimize_lr=0.04 \
  style_internal_appearance_strength=0.0 \
  style_geometry_strength=0.85 \
  style_opacity_strength=0.0 \
  style_strength=0.0 \
  style_track_geometry=true \
  save_img=true
```

**Multi-color style images** (e.g. a style image with distinctly different hair/skin/lip/background
colors, not one overall tone) need two more mechanisms, on by default:
- A per-point **anatomical correspondence** loss (`lambda_anatomy`, default 60) -- see below,
  this replaced an earlier coarse-grid approach that kept mismatching regions.
- A 3D k-nearest-neighbor smoothness term (`lambda_color_smoothness`, default 150), now
  **region-aware** -- see below.

### Anatomical correspondence (replaced `RegionCorrespondence`)

The original approach (`RegionCorrespondence`, now removed) matched a coarse 16x16 grid of the
render to whichever region of the style image looked structurally similar, via VGG feature cosine
similarity plus a hand-tuned position-distance penalty (`region_position_weight`, ended up at 6.0
after two rounds of retuning). It worked well enough for the single style image it was tuned
against, but on real user photos it kept mismatching: hair color bled into skin, skin bled into
other places, because a texture-similarity proxy has no real notion of "this is hair, that is
skin" -- it's a coarse grid cell average, and a cell straddling a hair/skin boundary just gets one
blended target.

The actual fix: query points aren't arbitrary -- every one is tied to a specific canonical FLAME
vertex (that's the whole point of the `e2e_flame` query-point scheme), and FLAME ships its own
real per-vertex part labels (`FLAME_masks.pkl`: face/scalp/lips/nose/ears/eyes/neck/...), already
loaded into `flame_model.mask.v.<region>` for other purposes but unused for styling. Two things
follow from that:

1. **Exact per-point style targets, no 2D matching at all.** Content and a from-scratch
   reconstruction of the *style image itself* (run through the model's own normal, unstyled path)
   share the same canonical point ordering -- point `i` is "the same anatomical vertex" in both,
   regardless of whose face shape it is. So point `i`'s optimization target is simply
   `style_reconstruction.colors[i]`: the network's own predicted color for the style image at that
   exact vertex. `_encode_own_point_colors()` in `color_optimize.py` computes this once
   (no_grad, before the optimization loop); `lambda_anatomy * F.mse_loss(content_base_color *
   (1+gamma) + beta, style_point_colors)` is added to every step, computed purely in 3D --
   no rendering involved, so every point gets a clean, region-correct gradient every step,
   including points poorly covered by the sampled render views.
2. **Region-aware smoothness.** The KNN smoothness term now only averages a point with same-part
   neighbors (`_build_region_labels()` propagates FLAME's part labels from the ~5k-vertex template
   onto the ~20k subdivided query points by nearest neighbor, once, since both are static
   shape-independent templates). Verified zero cross-region neighbor pairs in the built graph.
   Before this, a scalp point 2mm from a face-skin point got smoothed toward it regardless of what
   either one was -- exactly the mechanism most directly responsible for hairline colors bleeding
   into skin.

Because `lambda_anatomy` now gives a *real* per-point target, `lambda_color_uniformity` (which used
to fight isolated points going extreme, back when whole-image AdaIN/Gram loss was the only signal
and gave no reason for any two points to differ) was lowered 30.0 -> 3.0 -- a high uniformity
weight actively fights legitimate region differentiation now. `lambda_adain` was also lowered
20.0 -> 5.0: AdaIN matches whole-image feature mean/std with no regional awareness at all, so on a
style photo that's mostly warm hair by pixel area, it pulled skin toward that same warmth,
fighting the anatomy target. Verified on `musk.jpg` + an AAHQ portrait (`aligned/11.png`, warm
gold-orange hair, pale skin, blue eyes): anatomy loss converges cleanly (0.098 -> 0.012 over 150
steps), eyes and lips pick up distinctly different tones from skin/hair (previously indistinguishable
under the grid approach), hair reads as a cohesive warm copper tone.

**Residual limitation, not fixed by this change:** even a *plain, unstyled* reconstruction of a
painted/stylized style image through this model's frozen encoder already reads noticeably warmer
than the original photo (confirmed by feeding the style image through the normal, non-styling
inference path with no reference at all) -- the pretrained DINOv2+transformer backbone doesn't
perfectly disentangle a painting's warm color grading from the subject's actual local color when
reconstructing an out-of-photographic-domain image on its own. Since the anatomy target is *this
network's own prediction*, it inherits that bias; no loss reweighting fixes it, since the target
itself is what's warm. Only a backbone retrained on stylized/painted images would remove this
residual, not a change to the color-optimization loss.

**Cloth/clothing is not modeled at all.** FLAME is a bald head+neck model with no torso, shoulder,
or clothing geometry or vertex group -- there is no "cloth" region to give an anatomical target to.
Whatever color appears on a shirt/jacket in a render is emergent from whichever nearby vertex
group (usually `neck`/`boundary`) and the encoder's general image features, not a real,
independently-controllable region. This is an architectural ceiling of the current point
representation, not something the color-optimization loss can address -- modeling real
torso/clothing geometry would be a separate, much larger project (a new point set, or a learned
offset region, extending beyond FLAME's head-only topology).

`style_internal_appearance_strength=0.0` disables the (weak, unreliable)
network-predicted color so it doesn't fight the optimized override.
`style_strength` (LAB post-process) can stay at `0.0` since the optimized color
already targets the real style statistics directly; raise it only for extra
contrast/mood polish on top. Tunable knobs: `style_optimize_steps` (more = closer
convergence, diminishing returns past ~500-800), `style_optimize_lr`,
`style_optimize_num_views` (more = better view-consistency, slower),
`style_optimize_anatomy_weight` (default 60 -- the main region-correctness knob now).

### Geometry: `style_geometry_strength=1.0` now actually reaches the style's shape

`_blend_style_shape_params()` used to give the first ~24 (most visually dominant) PCA shape
dimensions a boosted per-dimension weight, up to 1.6x, on the assumption that this would make
`style_geometry_strength` feel more responsive. It was never checked against the thing it was
supposed to approach -- the style's own tracked shape -- and when it finally was (by exporting the
style image's own FLAME mesh and measuring 3D vertex distance from the *blended* mesh to it, not
just distance from the content), it turned out to move AWAY from the style as strength increased
past ~0.5:

| `style_geometry_strength` | distance to true style shape (old, boosted-weight formula) |
|---|---|
| 0.0 (no blend) | 4.06% of face bbox diagonal |
| 0.5 | 1.17% (closest the old formula ever got) |
| 1.0 | 2.60% |
| 1.3 | 4.53% (worse than not blending at all) |
| 2.0 | 9.05% |

Root cause: at `strength=1.0` the ~276 unboosted dims matched the style exactly, but the boosted
early dims overshot past it -- and because PCA concentrates most of the visible shape variation in
those early dims, the overshoot dominated the net result. "Turn the strength up" was moving the
face away from the style image, the opposite of what the parameter promised.

Fixed by dropping the per-dimension boost entirely: `betas = content + strength * (style -
content)`, uniform across all dims. This is a plain linear interpolation/extrapolation, so
`strength=1.0` is now *mathematically guaranteed* to equal the style's own tracked shape exactly --
verified directly: 0.0 residual displacement from the true target (down from 2.60% under the old
formula), while still moving 4.53% away from the original content shape (98% of vertices displaced
&gt;1mm), so it's a real, substantial shape change, not a no-op. `strength` above 1.0 still
extrapolates past the style shape, now in a predictable straight line (`strength=1.5` lands at
`style + 0.5*(style-content)`) instead of a direction skewed by an unvalidated weighting scheme.
FLAME's shape space is still a real-human-head subspace -- it cannot represent genuinely
cartoon-exaggerated proportions (huge eyes, tiny nose) no matter how far strength is pushed, and if
the style image itself is stylized/painted, its own FLAME tracking is already a lossy
realistic-human approximation of it, which is a ceiling on how "similar to the style image" any
FLAME-space blend can get, independent of this fix.

## Inference

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 lam_env/bin/python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=exps/train_lam/style_appearance_e2e_v2/model_final.pt \
  image_input=assets/sample_input/status.png \
  motion_seqs_dir=assets/sample_motion/export/Look_In_My_Eyes/ \
  export_video=true export_mesh=false \
  style_image_path=<STYLE_IMAGE> \
  style_internal_appearance_strength=0.75 \
  style_geometry_strength=0.75 \
  style_opacity_strength=0.0 \
  style_strength=0.4 \
  style_track_geometry=true \
  save_img=true
```

`style_strength` is the LAB color-statistics post-process (`lam/stylization/color_transfer.py`).
It's still useful as a contrast/mood refinement layered on top of the network's color (matches
things a single global RGB affine can't, like shadow contrast) — it's no longer doing all the
work by itself the way it was before `appearance_mlp` was trained, so keep it moderate (~0.3-0.4)
rather than relying on it fully.

**Update — `style_strength`/`style_internal_appearance_strength` defaults raised 0.75 -> 1.0**
(`lam/runners/infer/lam.py:parse_configs`). Measured on 20 AAHQ style images against
`assets/sample_input/status.png`, using `tools/check_style_color.py`'s directional-magnitude
metric (cosine x magnitude -- penalizes a big move in the wrong direction rather than counting it
as "more style"): mean 41.9% -> 53.2%, median 42.4% -> 59.0%, 18/20 styles improved (+7 to +25
points each). Only 2 styles regressed, and both were already wrong-direction at 0.75 too (s06:
cosine 0.40 -> -0.40; s07: cosine -0.29 -> -0.27) -- raising strength amplifies whichever
direction the color-shift estimate already landed on, so it helps a reliable estimate and hurts an
unreliable one, but doesn't flip which side of zero a style lands on. `style_geometry_strength`
was already validated separately at 1.0 (see the geometry section above) and unaffected by this
change. `style_opacity_strength` (0.45 default) was not tested here and left alone.

**s07's wrong-direction color, investigated, not fixed**: separately, s07's own FLAME tracking was
found to silently fail -- `FaceBoxesDetector` (trained on real photos) scored a correctly-placed
face box at only 0.22-0.35 confidence on this cel-shaded illustration, well under the hardcoded
0.8 threshold, so it fell back to appearance-only (no shape blend at all) with no error, just a
swallowed exception. Fixed with a low-confidence fallback retry in
`tools/flame_tracking_single_image.py` (only fires when the strict pass finds nothing, logs loudly
when it does) -- verified end-to-end (preprocess/optimize/export all succeed, exported crop is
correctly aligned by eye). This likely affects other flat/cel-shaded AAHQ images beyond s07, not
just this one. It did NOT fix s07's color direction, though (cosine -0.29 before the tracking fix,
-0.27 after) -- the wrong-direction color estimate turned out to be a separate, still-open problem
from the tracking failure, not caused by it.

## Validating changes

Isolate appearance from geometry by comparing `style_internal_appearance_strength`=0 vs 1 with
`style_geometry_strength=0` fixed, and vice versa, both with `style_strength=0` (post-process off)
so you're only looking at network output. Compare against
`model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/` (unstyled pretrained release)
as a sharpness/identity baseline — any blur or identity drift relative to that baseline at
`style_internal_appearance_strength=0` means something leaked into the frozen backbone and is a
bug, not a style effect.

---

## Original Stage 1-4 plan (superseded, kept for reference)

This pipeline separates appearance/style reconstruction from face geometry. That separation is important: Stage 1 and Stage 2 train the lightweight style autoencoder, Stage 3 prepares and validates FLAME geometry references, and Stage 4 fine-tunes the full LAM geometry adapter.

### Stage 1: Natural-Face Appearance Pretraining

Goal: learn stable face appearance reconstruction before artistic fine-tuning.

Config:
```text
configs/train/lam_style_stage1_celeba.yaml
```

Input:
```text
celeba_hq_256/*.jpg
```

Run:
```bash
PYTHONPATH=. lam_env/bin/python lam/launch.py train.lam \
  --config configs/train/lam_style_stage1_celeba.yaml
```

Output:
```text
exps/train_lam/stage1_celeba/style_autoencoder_final.pt
exps/train_lam/stage1_celeba/style_train_summary.json
exps/train_lam/stage1_celeba/samples/
```

### Stage 2: AAHQ Artistic Appearance Fine-Tuning

Goal: adapt the appearance/style autoencoder from natural faces to aligned artistic AAHQ faces.

Config:
```text
configs/train/lam_style_stage2_aahq.yaml
```

Input:
```text
ffhq-dataset/aahq-dataset/aligned/*.png
```

Run:
```bash
PYTHONPATH=. lam_env/bin/python lam/launch.py train.lam \
  --config configs/train/lam_style_stage2_aahq.yaml
```

Output:
```text
exps/train_lam/stage2_aahq/style_autoencoder_final.pt
exps/train_lam/stage2_aahq/style_train_summary.json
exps/train_lam/stage2_aahq/samples/
```

**Superseded**: this checkpoint was never wired into anything but a pixel-level pre-filter on the
style image (`_apply_style_autoencoder` in `lam/runners/infer/lam.py`) — it never touched
`ReferenceStyleAdapter`, the module that actually carries style into the model. Do not use for new
work; see "What actually carries style" above.

### Stage 3: AAHQ Geometry Reference Bank

Still current — see "Stage 3: geometry" above.

### Stage 4: Full-LAM Geometry Adapter Fine-Tuning (original)

Goal: train `ModelLAM.style_adapter.geometry_mlp` so a selected AAHQ style reference can produce
stronger learned query-point deformation.

**Superseded by `configs/train/lam_style_geometry_adapter_v2.yaml`**, which does the same thing but
chains off a checkpoint with a working `appearance_mlp` instead of the raw pretrained release, so
appearance and geometry styling compose correctly together.

## Emotion control

Separate feature from stylization, deliberately kept architecturally decoupled from it -- see
`lam/models/emotion_adapter.py`'s module docstring for why (short version: it edits plain
`flame_params["expr"]`/`["jaw_pose"]` tensors in `lam/runners/infer/lam.py`, before the frozen
model is even called -- zero changes to `modeling_lam.py`/`gs_renderer.py`, so it can't interfere
with anything above). `EmotionAdapter` is trained from scratch (random init, not any pretrained
emotion model) via `tools/train_emotion_adapter.py`, pure FLAME-parameter regression against
FLAME-tracked reference photos, no rendering involved in training.

**Reference data**: `Piro17/affectnethq` (AffectNet-HQ, real photos, HuggingFace, ungated) tracked
via `tools/track_emotion_refs.py`, 683/700 usable. An independent pretrained expression classifier
(`trpakov/vit-face-expression`, PyTorch/`transformers`, chosen over `deepface` after `deepface`'s
TensorFlow dependency silently upgraded `numpy`/`opencv` and broke the whole pipeline's pinned
versions -- restored via `pip install --force-reinstall --no-deps` to the exact `requirements.txt`
pins) is used two ways: as a **judge**, scoring rendered outputs independently of whatever produced
them, and as a **training-data filter**, keeping AffectNet-HQ references only where the judge
agreed with the human label or disagreed without much confidence (< 0.6) -- this filtered set is
the "hybrid" checkpoint (`exps/train_lam/emotion_adapter_hybrid/`), the default for
sad/happy/neutral/surprise. `disgust` is exempt from this filter (kept 100% human-labeled) since
the judge has ~0% agreement on it in the source data too -- filtering by its opinion would delete
the class, not clean it.

**Per-class defaults** (`PER_CLASS_DEFAULT_STRENGTH`, `PER_CLASS_ADAPTER_PATH` in
`emotion_adapter.py`) exist because no single strength or checkpoint was best for every class,
measured, not assumed -- see those tables' inline comments for the numbers. `sad` needed a strength
boost (3.0/1.0 vs. global 1.0/0.5) because its class-mean delta was being cancelled by internally
inconsistent AffectNet-HQ references, fixed by `trimmed_inlier_indices`/`discriminative_trim_indices`
selecting only the most prototypical/distinctive subset before averaging. `fear` needed a different
checkpoint entirely (`emotion_adapter_discriminative/`, trained with `discriminative_trim_indices`,
which selects references by margin -- more similar to their own class mean than to the *closest
other* class's mean, not just internally self-consistent) plus **component-sharpening**
(`distinctive_component_mask`, amplify only the ~8 expr components distinctively fear's, not the
whole 100-dim vector uniformly) -- together: fear-recognition 20% -> ~40%.

### Anger and disgust: investigated thoroughly, not fixed -- here's the real diagnosis

Both were tried extensively -- 7+ checkpoint variants, multiple strength/sharpening levels up to
deliberately extreme settings -- and neither crossed into being correctly classified. The two
failures are diagnostically different, which matters for anyone picking this back up:

- **Anger** has a real, strong distinctive direction in the trained data (top component magnitude
  2.71, gap over every other class 0.99 -- comfortably above noise). Amplifying it, even 5x
  (`emotion_sharpen_top_k=15 emotion_sharpen_boost=5.0`, expanding the delta's own norm from 8.5 to
  26.7) barely changed the render and never moved the classifier's anger-score off ~0.5-1.7%. This
  was concluded to be a rendering/representation ceiling: the direction exists but doesn't translate
  into a visible facial change in this pipeline, for reasons scaling can't reach.
  **Correction, superseded below**: this conclusion was wrong, or at least incomplete. It wasn't a
  ceiling scaling couldn't reach -- it was that every fix tried here was still one *fixed* global
  direction, and the actual problem (see "Per-instance emotion optimization" below) is that a fixed
  direction reads correctly on some faces and not others. Per-instance test-time correction moved
  anger from 19.0% to 84.4% mean confidence across 14 identities, zero regressions -- not a ceiling.
- **Disgust** originally had no strong distinctive direction at all (best gap only 0.30 vs.
  anger's 0.99), so the working hypothesis was that it needed more/better source data -- AffectNet-HQ's
  candid photos might just be too subtle. Tested this directly: added 168 references from CK+
  (`AlirezaF138/ckplus-dataset` on HuggingFace), a *posed*, peak-apex-expression dataset, visibly
  strong/exaggerated disgust even at native 48x48 (confirmed by eye before committing any tracking
  time). Two real engineering hurdles solved along the way: (1) FLAME tracking's face detector
  failed on CK+'s tight, borderless crops ("No face detected") until padding was added around each
  image before tracking (`tools/track_ckplus_disgust.py`); (2) merged with the existing hybrid
  disgust set (99 + 168 = 267 total, more than any other class) and retrained with
  `discriminative_trim_indices`. Result: the hypothesis was **wrong** -- the distinctiveness gap
  barely moved (0.25 vs 0.30). What *did* change: the render finally showed real, visible movement
  (unlike anger, which never moved) -- but that movement's direction still isn't separable from
  other classes, now landing on "sad" instead of "fear"/"neutral"/"anger" depending on which
  checkpoint. Conclusion: disgust's problem was never data volume or clarity -- CK+ proved
  real, trackable, strongly-expressed disgust movement exists and is capturable. The ceiling is
  that FLAME's expression PCA basis does not carve out disgust's specific muscle pattern (nose
  wrinkle / upper-lip raise, AU9+AU10) as a direction separable from other negative-valence
  expressions, regardless of source data quality. Not promoted to a default checkpoint since it
  didn't fix the class it was built for.

**What would actually be needed for disgust specifically** (anger is addressed below -- this
paragraph's "beyond what's reachable" conclusion held for disgust, not for anger): a genuinely
different expression representation with explicit brow/nose degrees of freedom (e.g. a real
blendshape rig like ARKit's 52 blendshapes) rather than FLAME's general-purpose PCA expression
space, which was very likely built with more weight on broad/speech-relevant expressions than on
these specific, localized action units.

### Per-instance emotion optimization: the identity-blindness bug, and its fix

Found by chance while generating a 23-original x 26-style x 7-emotion batch: one "sad" render (a
real photo, no style involved) showed a clear smile -- teeth bared, mouth corners up -- not sad at
all, despite `emotion_class=sad` and the classifier disagreeing loudly (top1 "happy", 82.7%).

**Diagnosis.** `EmotionAdapter` (see its module docstring above) predicts one
`(expr_delta, jaw_delta)` per class conditioned only on `(class, driving-clip's own expr)` -- it
has zero information about the target person's face shape (`betas`). FLAME's expression components
are linear vertex offsets; the same offset can read as a frown on one face's resting geometry and a
smile on another's. Confirmed directly: a full strength sweep on the broken case
(`emotion_strength` from -3.0 to 3.0, i.e. both directions and multiple magnitudes) **never once
produced a render that read as sad** -- it only ever moved between neutral (weak) and happy
(strong). That rules out a strength or sign fix: the trained direction itself is wrong for that
face, at every magnitude, not just mis-scaled.

**Fix**: `lam/stylization/emotion_optimize.py`, `optimize_emotion_expr()`, wired into
`lam/runners/infer/lam.py`'s emotion block behind `emotion_optimize` (now default `true`). Mirrors
`optimize_style_colors`'s already-proven per-instance pattern exactly, applied to expression instead
of color: the expensive frozen backbone (image encoding, `latent_points`, `forward_gs`) is computed
once under `no_grad`; a short loop (default 30 steps, Adam, `lr=0.05`) then re-runs only the cheap
per-view `forward_animate_gs` + a differentiable pass through the same judge classifier
(`trpakov/vit-face-expression`, loaded directly rather than through the `pipeline()` wrapper so
gradients flow), starting from the adapter's own prediction and pulling `expr_delta`/`jaw_delta`
toward higher judge confidence for the requested class *on this specific face's actual rendered
output*. An L2 penalty on drift from the initial prediction (`emotion_optimize_reg`, default 0.02)
keeps the search close to the trained prediction rather than free to wander into a
classifier-fooling artifact -- checked by eye on every validation case below, not just trusted.

**Cost**: measured directly, +3.7s/request at the default 30 steps (23.0s -> 26.7s) -- cheap
because only the per-view render re-runs each step, not the backbone.

**Validation (196 runs: 14 real-photo identities x 7 emotion classes, base vs. optimized, no
style)**:

| class | base confidence | optimized | base match | optimized match |
|---|---|---|---|---|
| anger | 19.0% | 84.4% | 21.4% | 92.9% |
| disgust | 0.2% | 5.7% | 0.0% | 7.1% |
| fear | 53.9% | 71.7% | 78.6% | 85.7% |
| happy | 75.2% | 86.3% | 78.6% | 85.7% |
| neutral | 66.9% | 78.7% | 71.4% | 78.6% |
| sad | 5.0% | 49.2% | 7.1% | 50.0% |
| surprise | 19.7% | 58.6% | 21.4% | 71.4% |
| **overall** | **34.3%** | **62.1%** | **39.8%** | **67.3%** |

Every class improved, zero regressions across all 98 base/optimized pairs -- promoted to the
default on that basis. Disgust is the one class that barely moved, consistent with (not
contradicting) the diagnosis above: it has no strong distinctive direction to converge toward in
the first place, so there's little for a per-instance search to find regardless of identity.

**Follow-up: a counterexample the 92.9% match rate hid.** On `musk.jpg` -- an identity outside
the 14 above -- `anger` on the default (hybrid) checkpoint hit 97-98% classifier confidence but
rendered with the mouth held open, nearly identically to disgust/fear/sad/surprise's *raw,
unoptimized* predictions on the same checkpoint (confirmed by testing `emotion_optimize=false`
directly: the open-mouth look is already present before any optimization runs, and separately
that it survives `emotion_jaw_influence=0`, ruling out jaw_pose specifically -- it comes from
`expr_delta`). This matches the 0.77 anger/disgust cosine similarity noted above: the hybrid
checkpoint's anger direction is not actually distinct from a generic "mouth open" direction shared
across several classes, and the optimizer, hunting for anything that raises the classifier's
anger logit, found that shared shortcut instead of a genuine angry expression. A high mean
confidence and match rate across 14 identities did not rule this out, because neither metric
checks whether the render looks visibly *wrong* the way this does -- the same "verify by eye"
gap already flagged for the reg default.

Switching `anger` to the discriminative checkpoint (`PER_CLASS_ADAPTER_PATH` in
`lam/models/emotion_adapter.py`, previously only applied to `fear`) fixed it on this identity:
raw (unoptimized) prediction is already clean (closed mouth, tightened lips), and with
`emotion_optimize` on top it reaches 98.4% confidence while barely moving from that starting
point. Disgust was tested the same way on the discriminative checkpoint and got *worse*
(confidence collapsed to 0.8%, mouth rendered stretched) -- left off disgust's override on that
basis, consistent with disgust having no good direction on either checkpoint.

**Re-validated across the original 14-identity set -- reverted, this was a net regression.**
Re-ran the discriminative checkpoint for `anger` across the same 14 identities/protocol as the
196-run table above (no style, `emotion_optimize=true`). Result: mean confidence 84.4% -> 69.8%,
match rate 13/14 -> 10/14. Three identities that worked fine on the hybrid checkpoint
(`original_008`, `original_009`, `original_013`) came out top1='neutral' on discriminative, and
one (`original_001`) flipped from a correct match to top1='fear'. musk.jpg's open-mouth artifact
is real, but it is not representative of how hybrid's anger behaves on most identities --
switching the *global* default traded one visible failure for several new ones. **Reverted**:
`PER_CLASS_ADAPTER_PATH` no longer overrides `anger` (only `fear` still does). musk.jpg's specific
case remains an open, unfixed instance of the same shared-direction problem described above; if it
resurfaces on a particular identity, override per-request with `emotion_adapter_path` rather than
changing the global default without this same broad-sample check.

### Root-caused the mouth-open artifact and fixed it with a data retrain, not another knob

The checkpoint-swap experiment above pointed at *something* checkpoint-shaped being wrong for
anger, without explaining *why*. Traced it properly this round, on `musk.jpg`: turned
`emotion_optimize` off entirely (raw adapter, zero per-instance correction) -- mouth still open.
Zeroed `emotion_jaw_influence` -- mouth still open. That rules out both the optimizer and
`jaw_pose`; whatever's driving it lives in the trained adapter's own `expr_delta` prediction.

Measured it directly with the real FLAME head model (not a `jaw_pose` proxy -- confirmed above
that jaw_pose alone doesn't capture it): inner-lip landmark distance (`measure_mouth_opening.py`)
across all 683 tracked training references. Result: `anger` (mean 0.0101) and `disgust` (0.0099)
sit at roughly double `neutral`'s (0.0046) and close to `fear`'s (0.0122) -- meaning most of the
*source training photos* for anger/disgust are open-mouth shouting/snarling shots (confirmed by
eye: of 6 sampled anger references, only 1 -- a Clint Eastwood closed-mouth glare -- showed the
"quiet fury" look; the other 5 were teeth-baring/shouting). `fear`/`surprise`'s own high
mouth-openness is correctly part of THEIR signature and wasn't touched. This is a genuine data
problem, not an inference-time one -- no amount of regularization, strength, or checkpoint
selection was ever going to out-vote what the training targets themselves say "anger" looks like.

**Fix**: filtered out every anger/disgust reference with mouth_open >= 0.006 (close to neutral's
own median) -- keeps 54/92 anger, 36/99 disgust, all other classes untouched -- and retrained
`EmotionAdapter` from scratch on the filtered manifest
(`exps/train_lam/emotion_refs/tracked_emotion_refs_closedmouth.json`) with the same
discriminative trim used for every other checkpoint. Effect on the trained network: anger/disgust
cosine similarity dropped from 0.77 to 0.31 -- real separation, not just each class agreeing with
itself.

**Re-validated across the same 14-identity x 7-class protocol** (no style, `emotion_optimize=true`,
this checkpoint used for every class, since a full joint retrain can shift classes whose own data
didn't change):

| class | hybrid (before) | closed-mouth retrain (after) |
|---|---|---|
| anger | 84.4% / 13/14 | 77.2% / 12/14 (roughly flat) |
| disgust | 5.7% / 1/14 | 21.0% / 3/14 (better, still weak) |
| fear | 71.7% / 12/14 | 86.0% / 13/14 |
| happy | 86.3% / 12/14 | 93.6% / 14/14 |
| neutral | 78.7% / 11/14 | 83.2% / 12/14 |
| sad | 49.2% / 7/14 | 86.1% / 13/14 |
| surprise | 58.6% / 10/14 | 74.1% / 12/14 |
| **overall** | **62.1% / 67/98** | **74.5% / 79/98** |

A broad win, not just an anger fix -- `sad` nearly doubled its match rate (the single biggest
jump), `happy` hit 14/14, and this checkpoint alone beats the old fear-specific
discriminative-checkpoint-plus-sharpening combo (~40% in earlier testing) without needing that
override -- `CLASSES_WITH_DISTINCTIVE_SHARPENING` sharpening still applies automatically on top
and was active during this validation. **Promoted to `GLOBAL_DEFAULT_ADAPTER_PATH`** for every
class; the old fear-specific `PER_CLASS_ADAPTER_PATH` override was removed since this checkpoint
already outperforms it.

**Disgust is still the one real holdout** -- meaningfully better (7.1% -> 21.4% match) but not
fixed. Consistent with, not contradicting, the standing diagnosis: the mouth-open training bias
was part of disgust's problem, not all of it. FLAME's expression PCA basis likely still doesn't
carve out disgust's specific muscle pattern (nose wrinkle / upper-lip raise) as a direction
separable from other negative-valence expressions, regardless of how clean the training data is
-- see "What would actually be needed for disgust specifically" above.

**Styled-batch validation (140 runs: 4 identities x 5 styles x 7 emotions) -- found a real
regression, partially fixed, one gap still open.** The flagged limitation above was real: scored
against the identical base/optimized pairs already in the unstyled table, styled results were mixed
-- anger/happy/sad still improved sharply (e.g. anger 3.6% -> 63.9%), but **neutral got worse on
average (66.7% -> 51.1% confidence, 75.0% -> 55.0% match)**, and 30/140 combos (21%) regressed by
more than 5 points, some drastically (one case: 96.8% -> 0.9%).

Root cause: `optimize_emotion_expr` was correcting `expr_delta`/`jaw_delta` against the content
image's own unstyled face shape, while the real final render uses the style-blended shape
(`style_geometry_strength` swaps in the style image's tracked shape) -- the exact same
identity-blindness problem this whole feature exists to fix, just one level removed. **Fixed** by
threading `style_shape_params`/`style_geometry_strength`/`style_image` into the optimizer
(`prepare_style_reference`'s call moved earlier in `infer_single`, ahead of the emotion block, so
this is available in time -- see the reordering comment there). Verified directly on the worst
regression case: the in-loop proxy now reports 98.1% confidence (was silently wrong before).

**Second, separate problem surfaced while verifying that fix, still open**: the optimizer's proxy
render zeroes style color (`style_appearance_strength=0.0`, matching how `optimize_style_colors`
isolates geometry) -- but the real final image has the fully per-instance-optimized style COLOR
applied afterward, which the proxy never sees. Re-tested the same worst-case regression with the
shape fix in place: the in-loop proxy reported 98.1% neutral, but the actual final rendered/scored
output was 0.3% neutral (top1 'happy' at 99.4%) -- the optimizer converged against a face that
doesn't look like what actually ships, because color changes after optimization finishes.

**Color mismatch: fixed.** `prepare_style_reference` and the style-color-optimize block (which
produces `color_style_override`) were both moved ahead of the emotion block in `infer_single`, and
`color_style_override` is now threaded into `optimize_emotion_expr` the same way shape was --
`forward_gs`'s `color_style` is swapped for the real per-instance-optimized color, matching what the
final render actually uses. Verified on the worst regression case from the shape-only fix (96.8% ->
0.9% -> [shape fix only] 0.3%): with both fixes, in-loop proxy 98.1% and the real final render 95.7%
-- proxy and final now closely agree, unlike before. Re-validated on a partial rerun of the same 140
styled-batch combos (84/140 scored so far): every class beats *both* the base and the first broken
optimizer, including neutral (81.7%, was 66.7% base / 52.8% broken) -- no regressions found in this
partial sample.

**A second, different artifact -- found by the user looking at output, not by any of the above
metrics.** A styled 'happy' render scored 99.2% confidence and matched top-1, but the mouth was
visibly wide, asymmetric, and smeared -- wrong to the eye despite a clean score on every automated
check run so far. Isolated by generating the same content+style pair three ways: style alone (clean
mouth), emotion alone/no style (mild, plausible smile), style+emotion together (the distorted
mouth). Neither ingredient alone caused it -- only the combination did. Diagnosis: the optimizer,
now correctly targeting the style-blended face shape, found a mouth configuration on that
unfamiliar shape that satisfies the classifier without being anatomically sound -- exactly the
"classifier-fooling artifact" risk `lambda_reg` exists to bound, and at the old default (0.02) it
wasn't actually bounding anything (see `emotion_optimize.py`'s updated docstring and
`lam.py`'s `emotion_optimize_reg` default comment for the full sweep: 10x and 100x that default
barely changed the render; only ~20+ was a real constraint). Raised the default to 20.0 on that
basis -- confirmed by eye to clean up this specific case (confidence only dropping 99.2% -> 96.4%),
though some residual asymmetry remained even at 100, so this raises the floor rather than
guaranteeing every render is artifact-free.

**Net decision**: `emotion_optimize` defaults to `true` only when no style image is requested
(`_user_requested_style` in `parse_configs`, checked *before* `style_image_path`'s own fallback
default fills in -- that fallback happens to point at a real leftover file in this repo's root, so
checking file-existence after defaulting can't tell "no style" from "style"). Still defaults to
`false` whenever style is active: the color mismatch is fixed and partial re-validation looks
strong, but the full 140-combo rerun hasn't finished, and the mouth-artifact case above is a fresh
reminder that a good aggregate score doesn't rule out a bad individual render -- pass
`emotion_optimize=true` explicitly to use it with style, and check results by eye, the same
discipline this whole feature has needed from the start.

### RAVDESS augmentation: tried, measured worse, not deployed

User asked to try RAVDESS (Livingstone & Russo, 2018 -- video-only, no audio needed) as
additional training data, specifically to see if disgust's remaining weakness (21.4% match on
the closed-mouth checkpoint above) was a data-quality problem rather than the standing
FLAME-representation-ceiling hypothesis. RAVDESS actors perform each emotion as a controlled,
professionally-acted, intensity-labeled clip -- a genuinely different and cleaner source than the
candid AffectNet-HQ/CK+ photos used so far.

**Practical execution**: downloaded 6 of RAVDESS's 24 actors' `Video_Speech_Actor_XX.zip` files
(bandwidth-limited connection, ~490KB/s -- 24 actors would have taken many hours; picked a
male/female-balanced subset: 1, 2, 5, 6, 9, 10). Extracted one apex frame per clip at 80% of
duration (actors speak a fixed sentence then briefly hold the expression) via OpenCV, FLAME-tracked
all 622 resulting frames (0 failures), applied the same mouth_open >= 0.006 filter to the
anger/disgust frames only (kept 32/96 anger, 15/96 disgust; other classes added unfiltered, their
mouth state being correct for what they represent), merged into the existing closed-mouth-filtered
manifest (1,059 references total, roughly double), and retrained from scratch with the same
discriminative trim.

**Result: worse, not better.** Re-validated on the same 14-identity x 7-class protocol:

| class | closed-mouth (v1) | + RAVDESS (v2) |
|---|---|---|
| anger | 77.2% / 12/14 | 70.5% / 10/14 |
| disgust | 21.0% / 3/14 | 19.4% / 4/14 |
| fear | 86.0% / 13/14 | 80.0% / 13/14 |
| happy | 93.6% / 14/14 | 92.1% / 13/14 |
| neutral | 83.2% / 12/14 | 85.4% / 12/14 |
| sad | 86.1% / 13/14 | 71.1% / 11/14 |
| surprise | 74.1% / 12/14 | 77.6% / 13/14 |
| **overall** | **74.5% / 79/98** | **70.9% / 76/98** |

`sad` lost 2 matches, `anger` lost 2, `happy` lost 1 -- `disgust`'s +1 match gain didn't come
close to offsetting it. This was predictable before running the validation: the retrained
network's fear/sad expr-delta cosine similarity came out at +0.78 (every other checkpoint in this
document sits between -0.02 and +0.31 on its most-confused pair) -- the two classes stopped being
separable. Likely cause, not confirmed: only 6 of 24 actors were used, so fear/sad's directions in
the merged data may be dominated by a couple of actors whose acting style happens to overlap on
those two classes specifically, rather than reflecting genuine RAVDESS-wide overlap. A larger
actor sample, or apex-frame selection smarter than a fixed 80%-of-duration heuristic (e.g.
classifier-scored peak-frame selection), might avoid this -- but that is unverified, not assumed.

**Not deployed.** `GLOBAL_DEFAULT_ADAPTER_PATH` stays on `emotion_adapter_closedmouth`; the RAVDESS
checkpoint is kept at `exps/train_lam/emotion_adapter_v2/` for reference only, in case someone
wants to pick this back up with the full 24-actor set. This also produced two reusable, unrelated
artifacts worth keeping regardless of the outcome: `measure_mouth_opening.py`'s approach
generalizes to any new reference source (real FLAME landmarks, not a jaw_pose proxy), and a bug
in the download retry script (a flaky HEAD request returning a tiny/wrong content-length silently
marked one actor's download "complete" at 14MB instead of 523MB) was caught and fixed by sanity-
checking the expected size before trusting it -- worth remembering for any future bulk download
over an unreliable connection.

### The lateral lower-lip shift: measured decomposition (dominant cause NOT yet fixed)

User-reported across three separate runs (two content images, two style images, two driving
clips): "the lower lip goes more to the right side of the face and it looks bad", on most
emotions. It is a real, reproducible artifact, and it is NOT any of the things this document has
been tuning.

**Pose-robust metric.** Measuring the mouth against the nose-bridge midline is confounded by head
yaw (a turned head displaces the mouth in projection even on a perfectly symmetric face -- the
naive metric reported +40px on renders vs -14px on the source photo, mostly pose). The honest
measure for this specific artifact is the LOWER lip's lateral offset relative to the UPPER lip:
both sit at nearly the same depth on the same head, so head rotation shifts them together and
cancels. All figures below are in px on the landmark detector's 1024x1024 crop.

| condition | lower-vs-upper lip |
|---|---|
| source photo, musk.jpg | **-3.84** |
| source photo, status.png | **-4.03** |
| render, no style, no emotion | **+11.28** |
| render, styled + emotion (neutral) | +11.24 |
| render, styled + emotion (sad) | **+23.49** |
| render, styled + emotion (happy) | **+23.49** |

So the source faces are very slightly LEFT-biased, and the pipeline introduces a ~15px RIGHTWARD
swing with everything disabled, which style+emotion then roughly doubles.

**What it is not** (each ruled out by measurement, not assumption):
- *Not the EmotionAdapter or the per-instance optimizer* -- the artifact is fully present with
  `emotion_class` unset.
- *Not the style pipeline* -- fully present with style strengths all at 0.0.
- *Not the subject's own face* -- source photos measure -3.8/-4.0, the opposite direction.
- *Not the tracked shapes* -- every tracked identity AND style shape (musk, status, pop, aahq 10,
  style_image), evaluated at zero expression, measures between -0.00025 and +0.00006
  lower-vs-upper, i.e. symmetric. `style_geometry_strength=1.0` is not importing a crooked mouth.

**What it partly is: the driving clips.** Every avatar replays a clip's tracked `expr`/`jaw_pose`
verbatim, and those clips carry a real one-directional lateral bias. Mean lower-lip offset in
FLAME space, and the share of frames pulled the same way: `Look_In_My_Eyes` +0.00230 (95% of
frames, jaw-yaw mean +0.0156), `GEM` +0.00180 (90%), `Michael_Wayne_Rosen` +0.00137 (95%),
`I_Am_Iron_Man` +0.00104 (95%), `Joe_Biden` +0.00075 (80%); the rest are near-neutral.
`Look_In_My_Eyes` -- the default in `run_7emotions.sh` -- is the worst of the twelve.

`lam/models/motion_symmetry.py` fixes that component properly: FLAME's expression basis is linear,
so "the laterally symmetric part of an expression" is a fixed 100x100 matrix (mirror each
expression direction's vertex offsets in the canonical pose, average, project back), cached once
and applied per request, plus proportional damping of jaw yaw/roll (pitch untouched). Exposed as
`motion_symmetry` (0.0 off, 1.0 fully symmetric). In FLAME space it works exactly as intended --
mean lower-lip offset over `Look_In_My_Eyes` goes +0.00211 -> +0.00005 at strength 0.75, and at
1.0 the residual (-0.00064) is precisely the identity's own natural asymmetry, which is correct to
keep.

**But it only moves the rendered artifact 11.28px -> 10.12px (about 10%), so it is NOT the fix.**
Defaulted to **0.0 (off)** on that basis -- it is a real correction for a real bias, but enabling
it by default would imply a fix it does not deliver.

**The dominant terms are downstream of FLAME parameters**, roughly ~10px from the base
reconstruction/Gaussian-decoder path and a further ~12px from the style+emotion interaction.
Neither is expressible in expr/jaw/betas -- the FLAME mesh for these inputs is near-symmetric while
the render is not, which localizes it to the learned per-point Gaussian prediction (`gs_net`
predicts xyz offsets on the FLAME surface) or the rasterization. Note `gs_renderer.py` already has
a `vtx_sym_idxs` symmetry hook, but it is dead code (hardcoded `None` at both call sites) and
covers only `shs/scaling/opacity/rotation` -- not `xyz`, so enabling it as-is could not fix a
positional shift. Fixing this properly means symmetrizing the decoder's positional output, which
touches model internals rather than the isolated inference-time edits everything above uses --
not attempted here, and it should be validated on the 14-identity protocol like every other
change in this document before being defaulted on.

**Follow-up: the direction flips with the content image, and the emotion deltas are half
asymmetric.** A third content image (`cluo.jpg`, a frontal symmetric headshot) reproduced the
artifact pulling the OTHER way, which rules out a single fixed directional bias. Measuring the
mouth's deviation from its own anatomical midline (nose-bridge 27 -> chin 8, normalised by
inter-ocular distance, so head yaw largely cancels) against each render's OWN source photo:

| emotion | cluo (delta vs its photo) | musk (delta vs its photo) |
|---|---|---|
| neutral | -2.2 | -0.4 |
| happy | -2.3 | -3.4 |
| sad | +3.3 | -1.3 |
| anger | +1.4 | -- |
| disgust | -2.3 | -- |
| fear | -1.3 | -- |
| surprise | +4.2 | -- |
| no style, no emotion | -- | -2.9 |

The swing is -3.4 to +4.2 %IOD depending on which emotion is selected -- i.e. a large part of the
artifact is emotion-class-dependent, not a constant offset.

Root cause for that part, measured on the default checkpoint at the mean driving context: the
asymmetric residual `d - M@d` (M = the expression-symmetry matrix from motion_symmetry.py) is
**35-72% of every class's delta norm** -- anger 59.5%, happy 52.6%, surprise 51.6%, disgust 44.6%,
fear 39.1%, sad 35.5%, neutral 72.5% (of a tiny 0.31 norm). Each class also carries its own jaw
yaw (surprise -0.0123, happy -0.0064, anger -0.0037, fear +0.0037). A prototypical
anger/happy/surprise is anatomically symmetric, so that residual is mostly the idiosyncratic
asymmetry of whichever reference photos survived trimming -- and it is why the sideways pull
changes direction with the emotion class.

`emotion_symmetry` (in `parse_configs`, applied to the adapter's prior before the per-instance
optimizer) damps it with the same matrix. **Results are mixed, so it is defaulted 0.0 (off):**
on cluo, `surprise` improved (lower-lip deviation 3.91 -> 0.35) and `happy` roughly held
(-1.68 -> 0.12 mouth), but `sad` overshot and flipped sign at similar magnitude (+5.83 -> -6.35).
The likely reason for sad specifically is that it runs the loosest regularization of any class
(`emotion_optimize_reg` 1.0 vs 20.0 for happy), so the per-instance optimizer has the most freedom
to wander away from the symmetrized prior it was given. Both knobs are real, measured corrections
for real components of the artifact; neither is a fix on its own, and neither should be defaulted
on without the 14-identity validation.

**Correction: the FLAME-space clip bias does NOT predict rendered mouth quality.** The per-clip
lateral-bias table above led to an obvious-looking recommendation -- "prefer a low-bias driving
clip" -- which measurement then contradicted. 4 clips x 7 emotions on cluo.jpg + pop.png, full
stylization, symmetric checkpoint, scoring each render's lower-lip deviation from its own source
photo's baseline (%IOD) and the judge classifier's top-1 match:

| clip | FLAME-space bias | mean abs deviation | max | emotion match |
|---|---|---|---|---|
| `Look_In_My_Eyes` | +0.00230 (worst) | **3.54** | 9.5 | 71% |
| `Joe_Biden` | +0.00075 | 6.63 | 18.0 | **86%** |
| `Speeding_Scandal` | +0.00020 | 7.36 | 14.8 | **86%** |
| `Anti_Drugs` | -0.00054 (lowest) | **15.61** | 32.9 | 57% |

The ranking is essentially inverted: the clip with the WORST parameter-space bias renders the most
stable mouth, and the one with the LOWEST bias renders by far the worst (confirmed by eye --
`Anti_Drugs`/anger shows a badly twisted mouth, `Look_In_My_Eyes`/happy is clean). So whatever
dominates rendered mouth quality is a different property of the clip (plausibly head-pose range,
or how far its frame-0 pose sits from the content photo's) and not its lateral expression bias.
This also undercuts the premise behind `motion_symmetry`: it corrects a quantity that does not
drive the artifact, which is consistent with it only moving the rendered result ~10%.

Practical guidance from this sweep: keep `Look_In_My_Eyes` when mouth stability matters most,
`Joe_Biden` for the best emotion-recognition/stability balance, and avoid `Anti_Drugs`. Disgust
failed on all four clips, consistent with its standing ceiling.

### SUPERSEDED -- do not act on this section (see "Both changes reverted" below)

> This section recorded a promotion that a later, better measurement reversed. It is kept
> because the reasoning is still useful, but the conclusion is wrong and it should not be
> used to set defaults. It caused a real incident: a later agent read this section, found
> the code disagreed with it, and "fixed" the code to match -- re-applying two changes that
> had been reverted on evidence.

### Resolution: symmetric-target retrain promoted to default

Two changes, both now default:

1. **`CLASSES_WITH_DISTINCTIVE_SHARPENING` emptied.** Sharpening did not merely reweight fear's
   delta, it nearly doubled its norm (4.08 -> 7.54) and more than doubled its lateral asymmetry,
   because the components it boosts are themselves largely asymmetric. Stacked on
   `emotion_strength=1.5` that is ~2.8x the raw delta -- the direct cause of the visibly twisted,
   diagonally-skewed fear mouth. Fear no longer needed the rescue it was added for.

2. **`emotion_adapter_symmetric` is the new `GLOBAL_DEFAULT_ADAPTER_PATH`** -- the closed-mouth
   manifest retrained with `--symmetrize_targets 1.0` (new flag in tools/train_emotion_adapter.py),
   making each reference's target delta laterally symmetric before trimming and regression, so the
   learned delta is symmetric by construction. This was chosen over the inference-time
   `emotion_symmetry` knob because that lands after the per-instance optimizer and interacts with
   it unpredictably (it fixed surprise but made sad overshoot and flip sign).

Delta-level effect: asymmetric share ~4x lower (anger 59.5% -> 11.5%, happy 52.6% -> 19.6%,
surprise 51.6% -> 12.1%, sad 35.5% -> 8.7%), jaw yaw/roll exactly 0.0 for every class, magnitudes
preserved (4.4-5.0).

Rendered effect (cluo.jpg + pop.png, lower-lip deviation from the source photo's baseline, %IOD):

| emotion | before | after |
|---|---|---|
| surprise | 6.63 | **0.76** |
| fear | 2.57 / 3.69 | **0.52** |
| neutral | 1.03 | **0.04** |
| sad | 5.96 | 4.68 |
| disgust | 2.47 | 2.67 |
| anger | 2.19 | 4.96 |
| happy | 0.05 | 2.37 |
| **cross-emotion spread** | **6.59** | **4.92** |

Recognition cost on the standard 14-identity x 7-class protocol: a wash -- 74.5% -> 74.0% mean
confidence, 79/98 -> 77/98 match, with surprise gaining (12/14 -> 14/14) and fear losing
(13/14 -> 11/14, confounded with the sharpening removal in the same run). Promoted on the basis
that recognition is unchanged within noise while cross-emotion mouth consistency measurably
improves.

**Still not fully fixed, and should not be described as such:** anger and happy regressed on
deviation, `sad` remains the least stable class, `disgust` is untouched by any of this, and the
largest single contributor to the artifact is still the ~10px introduced by the base
reconstruction with style and emotion both disabled -- i.e. the Gaussian decoder, which no
FLAME-parameter-space change can reach.

### Both changes reverted, and the render pipeline is nondeterministic

`emotion_adapter_symmetric` and the emptied `CLASSES_WITH_DISTINCTIVE_SHARPENING` were both
reverted after a lip SHAPE-asymmetry measurement (the centroid measure used to promote them is
blind to a lip that is smeared while its average point sits centred). At n=98 per checkpoint the
two checkpoints are indistinguishable (6.22 vs 6.24), recognition slightly favours closed-mouth
(79/98 vs 77/98), and removing fear's sharpening was worse on both measures. See the constants in
`lam/models/emotion_adapter.py` for the numbers.

**The pipeline is not reproducible run to run.** Three renders with identical config and
identical `seed=42` produce three different images (verified by md5): lip asymmetry 5.08 / 6.54 /
5.25, sd 0.80. Seeding is already applied at init, so this is not a missing-seed bug -- the
Gaussian rasterizer's blending order is not deterministic on GPU. Consequences:

* A single-render A/B comparison cannot distinguish a real effect from noise, which is how four
  separate "fixes" in this document appeared to work and then dissolved under n=98 validation.
* The practical lever is selection, not correction: `run_emotion_best_of_n.py` renders each
  emotion N times and keeps the best by an automatic criterion. Measured on cluo.jpg + pop.png at
  N=4: mean lip asymmetry 6.2 -> 2.62 against a 1.80 source-photo floor.
* Note the separate variance scales: run-to-run sd is 0.80, but sd ACROSS identity x emotion
  combinations is 4.36 -- some combinations are reliably worse, and best-of-N does not help there.

### The judge classifier cannot recognise anger or disgust, even in real photographs

Benchmarked `trpakov/vit-face-expression` -- the classifier used as the emotion metric throughout
this document, and inside `optimize_emotion_expr`'s objective -- on real RAVDESS photographs of
professional actors performing each emotion at strong intensity (n=12 per class):

| emotion | mean confidence | top-1 accuracy |
|---|---|---|
| happy | 97.8% | 100% |
| neutral | 77.0% | 92% |
| surprise | 63.9% | 67% |
| sad | 38.0% | 50% |
| fear | 20.2% | 33% |
| anger | 7.7% | **0%** |
| disgust | 0.6% | **0%** |
| overall | 43.6% | 49% |

**This invalidates a meaningful part of this document.** Every conclusion of the form "anger/
disgust/fear/sad is broken" that rests on this classifier is measuring the judge's blindness at
least as much as the render. In particular the repeated finding that "disgust has no separable
direction / is at a representational ceiling" was built on a metric that scores 0.6% on real
photographs of disgust -- it is not evidence about FLAME's expression basis at all. The renders
for these classes may have been substantially better than reported, for a long time.

The classifier is only trustworthy here for happy, neutral and surprise. For the other four,
human judgement is the better instrument, and automatic scores should be reported against the
ceiling row above rather than read as absolute recognisability. Styling is NOT a factor: styled
and unstyled renders of the same expression score almost identically (3/7 match each), so this is
the classifier's own class bias, not a domain-shift artifact of stylisation.
