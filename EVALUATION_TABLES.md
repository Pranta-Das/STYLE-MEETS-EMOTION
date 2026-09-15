# Evaluation tables

Paper-ready numbers with their measurement protocols. Every figure below was
measured on this pipeline; literature values are quoted as context only and are
**not** directly comparable (different datasets, different task setups).

---

## 1. Emotion control

All emotion numbers below were re-measured on 2026-09-04 on the **same 14
identities** (`sorted(original-image/*.jpg)[:14]`), the **same current code**,
`seed=42`, `emotion_optimize=true`, driving clip `Look_In_My_Eyes`. The unstyled
and styled arms are **paired**: same identity, same emotion, same seed, only
`style_image_path` differs. Reproduce with `run_styled_emotion_eval.sh` and
score with `score_emotion_arms.py`.

### 1a. Judge calibration, and its robustness to stylisation

Both candidate judges were benchmarked on **real photographs** — RAVDESS,
professional actors, strong intensity, n=12 per class — and then on those same
photographs after the pipeline's own stylisation pass. A judge that cannot
recognise an emotion in a real human face cannot tell you anything about a
render of it; and a judge that loses accuracy merely because an image is
stylised cannot be used to compare a styled arm against an unstyled one.

| emotion | ViT, real | ViT, stylised | py-feat, real | py-feat, stylised |
|---|---|---|---|---|
| happy | **100%** | 100% | 67% | 83% |
| neutral | 92% | 75% | **100%** | 92% |
| surprise | 67% | 17% | 50% | 50% |
| sad | **50%** | 33% | 25% | 67% |
| fear | 33% | 33% | **83%** | 67% |
| anger | 0% | 0% | **33%** | 17% |
| disgust | 0% | 0% | **67%** | 67% |
| **overall** | **49%** | **37%** | **61%** | **63%** |

Three consequences, and they decide the whole evaluation design:

* The ViT is **blind to anger and disgust** (0% on real photographs of both).
  Any "our anger/disgust is broken" conclusion drawn from it is measuring the
  judge, not the render.
* The ViT loses **12 points** to stylised appearance alone; **py-feat loses
  nothing** (+2, within noise). py-feat reads facial geometry and action units,
  so a palette shift barely moves it. This is what makes the styled-vs-unstyled
  comparison in 1b interpretable at all: under py-feat, a drop cannot be blamed
  on the judge.
* Scope caveat for that control: only the 2D colour transfer can be applied to
  a flat photograph, so it bounds the judge's loss to *appearance* change. Loss
  from geometry blending is correctly left attributed to the method.

### 1b. EmoAcc — unstyled vs styled (14 identities x 7 emotions, n=98 per arm)

| emotion | py-feat unstyled | py-feat styled | real-photo ceiling |
|---|---|---|---|
| anger | 28.6% | 28.6% | 33% |
| disgust | 42.9% | 21.4% | 67% |
| fear | 14.3% | 35.7% | 83% |
| happy | 78.6% | 78.6% | 67% |
| neutral | 71.4% | 78.6% | 100% |
| sad | 57.1% | 14.3% | 25% |
| surprise | 100.0% | 85.7% | 50% |
| **overall** | **56.1%** | **49.0%** | **61%** |

**Stylisation does not measurably cost emotion accuracy.** The 7.1-point gap is
not significant: McNemar's exact test on the 98 paired renders gives
**p = 0.296** (20 lost, 13 gained, 33 discordant). No individual class reaches
significance either (smallest p = 0.070, sad). This is the reportable claim —
and it is a stronger one than the accuracy number itself, because 1a already
established that py-feat is not the thing degrading.

For reference, the same judge scores 61% on real acted photographs, so both arms
sit near the level this judge reaches on real human faces.

**The ViT column must not be reported as an evaluation result.**
`optimize_emotion_expr` (lam/stylization/emotion_optimize.py) backpropagates
through that exact classifier as its optimisation objective, so the renders are
fitted to satisfy it. For completeness it reads 54.1% unstyled / 55.1% styled
(p = 1.000 paired), against 49% on real photographs.

### 1c. Regression introduced by the 2026-09-03 emotion changes

The same 14 identities, unstyled, scored before and after the 09-03 changes
(safety clamp added, `optimize_reg_jaw` raised 5 -> 50, symmetric adapter
checkpoint promoted, fear sharpening disabled):

| judge | 09-01 code | 09-04 code | lost / gained | p (McNemar) |
|---|---|---|---|---|
| py-feat (independent) | 65.3% | 56.1% | 17 / 8 | 0.108 |
| ViT (circular) | 80.6% | 54.1% | 28 / 2 | **<0.001** |

The ViT collapse is expected by construction — the safety clamp bounds exactly
the quantity the optimiser moves, so it must reduce the optimiser's own
objective. The py-feat drop is the one that matters: 9 points, 2:1 loss ratio,
**not significant at n=98 (p = 0.11) but not null either**. Worst-hit classes
are anger (57.1 -> 28.6) and fear (42.9 -> 14.3).

Confound to state honestly: the 09-01 run's exact flags are unrecoverable (its
script was deleted), so this compares two *configurations*, not one flag. A
clean isolation is one 35-minute run: re-render this arm with
`emotion_safety=false` on current code. Until that is done, do not attribute the
drop to the clamp specifically in writing.

### 1f. Ablations (2026-09-05, all paired, same 14 identities)

**Safety clamp (`emotion_safety`), unstyled, n=98 paired renders:**

| judge | clamp on | clamp off | lost / gained | p (McNemar) |
|---|---|---|---|---|
| py-feat (independent) | 56.1% | 56.1% | 5 / 5 | **1.0000** |
| ViT (circular) | 54.1% | 79.6% | 0 / 25 | **<0.0001** |

This is the cleanest demonstration of the circularity in this project, and it
settles the clamp question. Removing the clamp lets `optimize_emotion_expr`
reach its ViT objective on 25 of 98 renders and it never once loses one --
a perfectly monotone +25.5 points. The independent judge sees **exactly zero**
net change (5 lost, 5 gained, a coin flip).

Two conclusions:

* **Keep the clamp on.** It fixes the visible mouth artifact at *no measurable
  cost in real emotion recognisability*. The 09-03 "regression" in 1c is
  entirely a ViT artifact: the clamp accounts for 100% of that judge's 26-point
  drop and 0% of py-feat's 9-point drop, so whatever moved py-feat was one of
  the *other* 09-03 changes, or noise.
* Any future tuning scored on the ViT will chase this same phantom. A +25-point
  ViT gain here bought nothing real.

**Test-time optimiser (`emotion_optimize`), unstyled, n=98 paired renders:**

| judge | optimiser on | optimiser off | lost / gained | p (McNemar) |
|---|---|---|---|---|
| py-feat (independent) | 56.1% | 41.8% | 15 / 1 | **0.0005** |
| ViT (circular) | 54.1% | 28.6% | 25 / 0 | **<0.00001** |

**The optimiser earns its keep, and this refutes the "blind teacher" reading of
1a.** The prediction was that anger and disgust -- the two classes the ViT
scores 0% on in real photographs -- would *improve* once that judge stopped
steering them. They got worse (anger 2 lost / 0 gained, disgust 4 lost /
0 gained), as did every other class. Removing the optimiser costs 14.3 points
on a judge that was never in its loop.

The lesson is that **top-1 accuracy and gradient utility are different
properties of a classifier.** The ViT can have a decision boundary so
miscalibrated it never *outputs* "anger" on a real photograph, while its
internal features still respond to anger-like configurations -- so ascending
its anger logit moves the face somewhere an independent judge also reads as
angrier. Do not infer "useless as an optimisation target" from "inaccurate as a
judge"; they must be measured separately, and here they point opposite ways.

**Read together with the clamp result, these two ablations locate the exact
boundary between useful optimisation and overfitting:**

| optimiser excursion | py-feat | ViT |
|---|---|---|
| none (optimiser off) | 41.8% | 28.6% |
| within the clamp (shipping config) | **56.1%** | 54.1% |
| beyond the clamp (clamp off) | 56.1% | 79.6% |

The first step is real (+14.3 independent points). The second is pure
judge-overfitting (+25.5 ViT points, +0.0 independent). The safety clamp sits
precisely where the genuine gains stop -- which is the strongest available
justification for shipping it, and worth stating in the paper as a method
choice rather than a safety hack.

**Driving clip, unstyled, n=98 paired renders.** `Look_In_My_Eyes` (the clip
every number in this document uses) has the 2nd-highest speech jaw variance
(sd 0.0511) and 2nd-highest lateral jaw yaw (0.0182) of the 12 clips on disk;
`Speeding_Scandal` is the quietest (0.0116 / 0.0066). The hypothesis was that
the noisy clip was both burying the emotion signal and causing the sideways-lip
artifact.

| | talking clip | quiet clip |
|---|---|---|
| EmoAcc (py-feat) | **56.1%** | 42.9% (lost 23, gained 10, **p = 0.035**) |
| lip shape asymmetry, mean | **2.81** | 3.99 |
| lip asymmetry, cross-emotion sd | **0.57** | 0.68 |

**Wrong on both counts, and the quiet clip is significantly worse.** Despite
carrying nearly 3x the lateral jaw yaw, the talking clip produces *less* lip
asymmetry and better recognition. Keep `Look_In_My_Eyes`.

The recognition loss is almost entirely two classes: surprise 100% -> 28.6%
(10 lost, 0 gained, p = 0.002) and neutral 71.4% -> 35.7% (p = 0.062); disgust
moved the other way (42.9% -> 78.6%, 6 gained). The mechanism is worth
recording, because it is a real limitation of the clamp design: **the jaw clamp
is expressed as an absolute delta (+0.04) but expressiveness depends on the
resulting absolute pose.** From the quiet clip's near-closed resting jaw
(mean 0.0312) the most-open reachable mouth is ~0.071; from the talking clip's
0.0813 it is ~0.121. Surprise is simply unreachable from a closed-mouth clip.
A clamp expressed relative to the clip's own jaw range would not have this
failure mode -- untested, but it follows directly from the measurement.

**Projected gradient in the test-time optimiser, unstyled, n=98 paired.**
("clamp" here means clamping a NUMBER to a range, e.g. jaw delta to +-0.04 --
unrelated to "driving clip", which is the 519-frame video whose tracked FLAME
parameters drive the avatar. No audio is used anywhere in the model.)
`optimize_emotion_expr` solved an unconstrained problem whose answer
`_apply_emotion_safety` then truncated -- measured `jaw_drift` 0.101 against a
clamp of 0.04, so 60% of the solution was discarded and the optimiser had spent
its budget on a lever it was never allowed to pull. `emotion_optimize_project`
applies the hard clamps inside the loop instead (clamps only: the symmetry
blend is a contraction, not a projection, and iterating it drives the delta into
the fully symmetric subspace). The jaw shortcut is thereby closed --
verified in the logs, `jaw_drift` now stays 0.015-0.039 for all 100 steps while
`expr_drift` rises to 3.33.

| | clamp-after | PGD | test |
|---|---|---|---|
| py-feat (independent) | 56.1% | 54.1% | 5 lost / 3 gained, **p = 0.73** |
| ViT (circular) | 54.1% | **67.3%** | 1 lost / 14 gained, **p = 0.0010** |
| lip shape asymmetry | 2.809 | 2.709 | -3.5%, **p = 0.20** (Wilcoxon), better on 53/98 |
| lip asymmetry, cross-emotion sd | 0.57 | 0.48 | -16% |

**A null on everything that matters.** The reasoning was that closing the jaw
shortcut would force the optimiser to find real expression signal. Instead it
found an equivalent shortcut in *expression* space: +13.2 points on its own
objective, zero transfer to the independent judge, and a mouth improvement
indistinguishable from noise.

The formulation is kept as the default because it is the correct one -- the
constrained optimum rather than a truncated unconstrained one -- and it costs
nothing. **But it must not be presented as an improvement, and the +13.2 ViT
jump is a trap**: it is the same judge-overfitting the clamp ablation exposed,
reached by a different route.

**Read across all five ablations, the emotion accuracy looks like a plateau,
not a tuning problem.** py-feat sits at 54-56% across every intervention tried
(clamp on/off, optimiser on/off, projection on/off, two driving clips, two
geometry strengths), against 61% for that judge on real acted photographs. Only
removing the optimiser entirely moved it, and downward. The remaining headroom
to the judge's own ceiling on real human faces is about 5 points.

**Jaw opening bound (`emotion_max_jaw_open_delta`), 0.04 vs 0.20, n=98 paired
per condition.** Tested because a wider mouth raises classifier confidence, and
because the mouth artifact was historically blamed on jaw opening.

| | py-feat (independent) | ViT (circular) | lip asymmetry | mouth opening |
|---|---|---|---|---|
| unstyled 0.04 | 54.1% | 67.3% | 2.71 | 13.27 |
| unstyled 0.20 | 56.1% (p = 0.688) | 72.4% (p = 0.062) | 2.67 (**p = 0.836**) | 15.74 |
| styled 0.04 | 45.9% | 66.3% | 2.64 | 13.73 |
| styled 0.20 | 43.9% (p = 0.727) | 67.3% (p = 1.000) | 2.74 (**p = 0.126**) | 15.34 |

Two conclusions, and the first is the useful one:

* **The mouth artifact is NOT controlled by the opening bound.** Opening the
  mouth 5x wider changes lip symmetry by -0.04 (unstyled) and +0.10 (styled),
  neither significant, across 196 renders in both conditions. The opening really
  did increase (+2.46 / +1.61), so this is not a no-op. What controls the
  artifact is the *lateral* jaw clamp (0.01 rad) and the one-shot
  symmetrisation, which are independent of how far the mouth opens. The earlier
  destruction came from an unconstrained optimiser reaching jaw drift 0.101 with
  no lateral bound at all.
* **Opening wider buys no recognisability.** +2.0 points unstyled, -2.0 styled,
  both p > 0.68. The unstyled ViT column gains 5 with 0 lost (p = 0.062) while
  py-feat does not move -- the same optimise-against-the-judge pattern seen in
  the clamp and projection ablations.

**Decision: keep 0.04.** Not because 0.20 is unsafe -- it demonstrably is not --
but because it changes nothing measurable and the tighter bound needs no
defending. The finding worth reporting is that the bound is safe to expose as a
user parameter, since mouth stability does not depend on it.

**Style geometry strength, styled, n=70 paired cells:**

| style | geo=1.0 | geo=0.5 | lost / gained | p |
|---|---|---|---|---|
| gothic pale | 25.7% | 31.4% | 1 / 3 | 0.625 |
| yellow caricature | 20.0% | 17.1% | 1 / 0 | 1.000 |
| **both** | **22.9%** | **24.3%** | 2 / 3 | **1.000** |

**Negative result.** The hypothesis was that these two styles score low because
their non-human face geometry is blended in at full strength and flattens the
expression. Halving the blend changes nothing (5 discordant cells out of 70).
The geometry blend is not the cause; do not write it up as a trade-off curve.

The leading remaining explanation is judge degradation specific to *these
appearances*: the stylisation-robustness control in 1a was measured with
`pop.png` only, and py-feat's robustness may not hold for every style. That
control should be repeated per style before any per-style claim is made.


## 1g. EDTalk baseline — same identities, same driving performance

EDTalk (ECCV 2024 oral) accepts a single source image plus a discrete emotion
label, the closest match to our control modality. We ran the official code on
the **same 14 CelebA-HQ identities**, driven by the **same performance** (pose
and audio from our own `Look_In_My_Eyes` clip, resampled to their required
25 fps / 16 kHz). All seven of our emotions map 1:1 onto their predefined
weights. Four compatibility patches were needed (librosa keyword-only `mel()`,
deprecated `np.float`/`np.int` aliases, an fps cast, and replacing torchvision's
`write_video`, which is broken against the installed PyAV) — all pure API
renames that do not touch the method.

| emotion | ours | EDTalk |
|---|---|---|
| anger | 35.7% | **92.9%** |
| disgust | **50.0%** | 28.6% |
| fear | **35.7%** | 0.0% |
| happy | 78.6% | **100.0%** |
| neutral | **71.4%** | 14.3% |
| sad | 42.9% | **85.7%** |
| surprise | **92.9%** | 78.6% |
| **overall** | **58.2%** | **57.1%** |

n = 98 each. **Essentially tied overall, with opposite per-class strengths:**
EDTalk is far stronger on anger, happy and sad; we are stronger on neutral,
fear, disgust and surprise.

### Two measurement traps this exposed, both caught by looking at the images

**Frame 0 is not scoreable for EDTalk.** Its expression ramps in over the
sequence; frame 0 is byte-identical across all seven requested emotions. EDTalk
is therefore scored at the mid frame. Ours is scored on its frame-0 preview as
everywhere else in this document.

**py-feat is not domain-invariant, and this nearly produced a badly wrong
number.** Scoring EDTalk's native 256x256 warped crops (which carry large black
borders from the alignment warp) returned *"anger"* for almost every requested
emotion — an apparent 14.3% overall — while the frames visibly showed the
correct expression. Both systems are therefore scored on identical
MTCNN-detected 320x320 face crops with the same margin rule. That single change
moves EDTalk from 14.3% to 57.1%, and our own figure from 54.1% to 58.2%.

**Any cross-system emotion comparison must normalise the image domain before
judging.** This is the same lesson as the judge-calibration table in 1a, and it
means a raw EmoAcc quoted from one paper cannot be compared to one computed on
another paper's output format.

### Scope
EDTalk emits 256x256 2D video driven by audio; ours emits an animatable 3D
Gaussian avatar from a discrete label with no audio subsystem. This compares
emotion recognisability of the output only, not equivalent systems.


## 1h. SOTA-style metric tables (re-run baselines, our validation set)

Both tables are computed by us, on our own data, with one judge — not quoted
from other papers. See the semantics warning below before reading columns 1-4
of Table A.

### Table A — Emotion (14 CelebA-HQ identities, n=98 each)

| Method | PSNR | LPIPS | SSIM | LMD | AUE-L$\downarrow$ | AUE-U$\downarrow$ | EmoAcc$\uparrow$ |
|---|---|---|---|---|---|---|---|
| **Ours** | 38.43 | **0.038** | **0.842** | 4.03 | 0.244 | 0.234 | **58.2%** |
| EDTalk (ECCV'24) | 34.38 | 0.118 | 0.697 | 4.66 | **0.241** | **0.203** | 57.1% |
| EAT (ICCV'23) | 37.25 | 0.074 | 0.769 | **3.86** | 0.247 | 0.224 | 35.7% |

All three re-run by us on the same identities and scored by the same judge on
identical MTCNN 320x320 face crops.

**Driving-signal parity differs by row, and this matters.** Ours and EDTalk are
driven by the *same* performance (our `Look_In_My_Eyes` clip: pose + audio).
EAT cannot ingest a custom driving video without its TensorFlow-1.15
preprocessing path, so it uses one of its own bundled pre-processed clips
(`M003_neu_1_001`). Emotion comes from the discrete label in all three, but
EAT's row is "same identities, same labels, different driving performance" --
weaker parity than the EDTalk row.

**Columns 1-4 do NOT mean what they mean in prior-work tables.** EDTalk's own
Tab. 1 computes PSNR/SSIM/LMD as *fidelity against a held-out real MEAD video*.
MEAD is unavailable here, so there is no paired ground truth. These are measured
against **each system's own neutral output** = **expression displacement**: how
far the face moved when the emotion was applied. A system that barely moves the
face scores "well" on PSNR and SSIM, so no arrow is given for those four.

Read that way: we reach the highest EmoAcc with the *least* displacement from
neutral, EDTalk matches our accuracy with roughly 3x the displacement, and EAT
displaces about as little as we do but converts less of it into recognisable
emotion.

**AUE and EmoAcc keep their original meaning.** AUE is distance from the mean
action-unit profile of *real* RAVDESS photographs of that emotion. All three are
within 0.006 on the lower face; EDTalk is closest on the upper face (0.203),
consistent with its stronger brow and eye movement.

EAT's errors are the classic confusions rather than a collapse -- fear read as
surprise 12/14, disgust as anger 10/14 -- which is why its 35.7% is reported as
a real measurement and not discarded as an artifact.

### Table B — Stylisation (18 identities x 7 styles, n=126 each)

| Method | LPIPS$\downarrow$ | SSIM$\uparrow$ | PSNR$\uparrow$ | CLIP out$\leftrightarrow$style$\uparrow$ | Rank-1 pref. |
|---|---|---|---|---|---|
| **Ours** | **0.282** | **0.585** | **13.44** | 0.744 | not run |
| JoJoGAN (ECCV'22) | 0.363 | 0.504 | 12.40 | **0.754** | not run |
| *SigStyle (AAAI'25)* | *0.5191* | *--* | *--* | *--* | *9.1%* |

Ours and JoJoGAN are measured identically, each against its own un-stylised
output (our unstyled render; JoJoGAN's e4e reconstruction). The SigStyle row is
**quoted from its paper on different data** and is context only -- it could not
be re-run here (section 2e).

**Rank-1 preference requires a human user study and was not run.** It is left
blank rather than substituted with an automatic proxy.

Colour adoption on face pixels, the metric that most directly measures
stylisation strength, is in section 2d: **50.9% ours vs 19.6% JoJoGAN**.


## 1i. Final re-run baseline tables (all executed by us on our data)

Five prior methods were obtained, patched for modern PyTorch, and run on our own
validation set. No number below is quoted from a paper.

### Emotion — 14 CelebA-HQ identities, n=98 each

| Method | Venue | Driving | PSNR | LPIPS | SSIM | LMD | AUE-L$\downarrow$ | AUE-U$\downarrow$ | EmoAcc$\uparrow$ |
|---|---|---|---|---|---|---|---|---|---|
| **Ours** | -- | our clip | 38.43 | 0.038 | 0.842 | 4.03 | 0.244 | 0.234 | **58.2%** |
| EDTalk | ECCV 2024 | our clip | 34.38 | 0.118 | 0.697 | 4.66 | **0.241** | 0.203 | 57.1% |
| EAT | ICCV 2023 | own clip | 37.25 | 0.074 | 0.769 | 3.86 | 0.247 | 0.224 | 35.7% |
| StyleCLIP | ICCV 2021 | none (static) | 38.73 | 0.051 | 0.813 | -- | 0.266 | **0.188** | 31.6% |
| PD-FGC* | CVPR 2023 | own clip | 62.16 | 0.001 | 0.998 | 0.15 | 0.322 | 0.243 | 14.3%* |

Judge: py-feat (ResMaskNet); all systems scored on identical MTCNN 320x320 face
crops. Real-photograph ceiling for this judge is 61%. We lead EmoAcc, and are
within 3 points of that ceiling.

**Parity differs by row and is stated in the Driving column.** Ours and EDTalk
share our `Look_In_My_Eyes` performance -- the tightest comparison. EAT uses its
own bundled clip (it cannot ingest a custom driving video without its
TensorFlow-1.15 preprocessing). StyleCLIP edits a static image with no
animation at all, so it is the loosest parity here.

**Correction:** StyleCLIP saves its result as `[original | edited]` side by side;
the first scoring pass let the face detector pick either half, so its row was
wrong (PSNR 59.30, EmoAcc 26.5%). Re-scored on the edited half only it is
PSNR 38.73 / EmoAcc 31.6%, and it no longer leads the displacement columns.

**PSNR/LPIPS/SSIM/LMD are expression displacement, not fidelity** -- measured
against each system's own neutral output, because MEAD is unavailable and there
is no paired ground truth. High PSNR means the face barely moved. This is why
PD-FGC and StyleCLIP sit at the top of those columns while scoring lowest on
EmoAcc: they change the face least. No arrows are given for those four columns.

***PD-FGC's 14.3% measures our input, not the method.** It takes emotion from a
temporally coherent expression video of one identity; we could only supply 40
RAVDESS stills from six different actors, which its expression encoder averages
toward neutral (outputs differ from its own neutral by 3-6/255, and 14.3% is
chance for 7 classes). Reported for completeness and struck from any ranking.

### Stylisation — 18 content x 7 styles, n=126 each

| Method | Venue | LPIPS$\downarrow$ | SSIM$\uparrow$ | PSNR$\uparrow$ | CLIP$\uparrow$ | colour adoption$\uparrow$ |
|---|---|---|---|---|---|---|
| **Ours** | -- | **0.282** | 0.585 | 13.44 | 0.744 | **50.1%** |
| JoJoGAN | ECCV 2022 | 0.363 | 0.504 | 12.40 | **0.754** | 13.6% |
| StyTR-2 | CVPR 2022 | 0.372 | 0.427 | 12.31 | 0.635 | 33.0% |
| AdaAttN | ICCV 2021 | 0.312 | **0.583** | **15.28** | 0.631 | 32.1% |

Fidelity is against each system's own un-stylised output (our unstyled render;
JoJoGAN's e4e reconstruction; the content photograph for the two pure 2D
transfers). Colour adoption is on MTCNN face crops for all four.

**PSNR here rewards doing nothing.** AdaAttN leads it (15.28) because it changes
the image least; colour adoption is the column that measures stylisation
strength, and we lead it by 17 points over the next method.

### PD-FGC (CVPR 2023) — ran, but not validly drivable here

PD-FGC executed successfully after patching (`lws` does not build on modern
Python; only its `.stft()` is used, which is a standard STFT, so librosa was
substituted -- LWS differs in phase reconstruction, which this path never uses).

It takes emotion from a **temporally coherent expression video of one person**,
not a label. We supplied 40 real RAVDESS photographs per emotion; that is a bag
of stills from six different actors, which is not the signal its expression
encoder expects. The result was a near-neutral expression code: outputs differ
from its own neutral by only 3-6/255 and EmoAcc lands at 14.3%, i.e. chance for
7 classes.

**That number measures our input, not the method, and is not reported as a
baseline result.** Driving PD-FGC properly needs per-emotion video of a single
identity -- MEAD-shaped data we do not have.

### Reference list for these baselines

| method | citation | code | weights |
|---|---|---|---|
| JoJoGAN | Chong & Forsyth, ECCV 2022 | github.com/mchong6/JoJoGAN | StyleGAN2-FFHQ + e4e (Drive) |
| StyTR-2 | Deng et al., CVPR 2022 | github.com/diyiiyiii/StyTR-2 | authors' Drive links |
| AdaAttN | Liu et al., ICCV 2021 | github.com/Huage001/AdaAttN | authors' Drive link |
| EDTalk | Tan et al., ECCV 2024 | github.com/tanshuai0219/EDTalk | HuggingFace tanhshuai0219/EDTalk |
| EAT | Gan et al., ICCV 2023 | github.com/yuangan/EAT_code | authors' Drive links |
| PD-FGC | Wang et al., CVPR 2023 | github.com/Dorniwang/PD-FGC-inference | authors' Drive link |
| StyleCLIP | Patashnik et al., ICCV 2021 | github.com/orpatashnik/StyleCLIP | StyleGAN2-FFHQ + OpenAI CLIP |

**Datasets used in these comparisons:** CelebA-HQ (content identities, emotion
table), AAHQ-derived portraits (content identities and style references,
stylisation table), RAVDESS (real-photograph AU reference, judge calibration,
and the PD-FGC expression attempt). MEAD and HDTF were **not** available and no
result here uses them.

### 1d. Self-reconstruction fidelity (RAVDESS protocol, n=3 actors)

Each actor's own neutral photograph is the pipeline input; the output is
compared against that same actor's real photograph for the target emotion.

| metric | neutral set (n=3) | emotional set (n=18) |
|---|---|---|
| PSNR | 10.79 dB | 11.00 dB |
| LPIPS | 0.318 | 0.305 |
| SSIM | 0.599 | 0.598 |
| LMD (px @1024) | 75.08 | 67.99 |
| AUE-U | 0.172 | 0.213 |
| AUE-L | 0.171 | 0.242 |

Protocol note, needed for honest comparison: EmoTaG reconstructs from a full
driving **video** of the same identity (dense per-frame supervision); this
generates from a **single photograph** with no footage of that person in that
emotion. The ~30x gap in LMD reflects that task difference, not method quality,
and these should not be presented as a ranking against it.

### 1e. Render stability

| quantity | value |
|---|---|
| lip shape asymmetry, source photographs | 1.80 |
| lip shape asymmetry, single render, **current config** (n=98) | **2.81** |
| lip shape asymmetry, single render, pre-09-03 config | ~6.2 |
| lip shape asymmetry, best-of-4 selection | 2.62 |
| run-to-run sd (identical config **and** seed) | 0.80 |
| sd across identity x emotion combinations | 4.36 |

The pipeline is **not reproducible**: three renders with identical config and
`seed=42` produce three different images (verified by md5). The Gaussian
rasteriser's blending order is nondeterministic on GPU. Qualitative figures are
therefore selected best-of-N by an automatic lip-symmetry criterion
(`run_emotion_best_of_n.py`); this is worth one sentence in implementation
details. Note this also sets the floor for how small a real effect the n=98
tables above can resolve.

---

## 2. Stylisation

### 2a. Our results (n=150 content x style combinations)

| metric | value |
|---|---|
| LPIPS vs unstyled reconstruction | 0.255 |
| SSIM vs unstyled reconstruction | 0.607 |
| PSNR vs unstyled reconstruction | 13.49 dB |
| Style loss (VGG19 Gram, vs style reference) | 0.0001 |
| CLIP similarity, output <-> style reference | 0.713 |
| CLIP similarity, output <-> unstyled reconstruction | 0.758 |

Protocol notes:

* LPIPS/SSIM/PSNR are measured against this pipeline's **own unstyled
  reconstruction**, not the raw source photograph. The raw photograph has a
  different background, lighting and compression, which dominates pixel metrics
  (measuring against it gives ~5 dB PSNR driven almost entirely by background
  mismatch).
* "CLIP similarity" is image-image cosine similarity, **not** CLIP *direction*
  similarity, which requires paired text prompts this dataset does not have.
* Style-loss magnitude is implementation-dependent (VGG layer choice and
  normalisation), so it is not comparable to another paper's number under the
  same metric name.


### 2c. Out-of-distribution styles

Every style in 2a/2b is drawn from AAHQ, the corpus the style adapter was
trained on, so "does this generalise to an unseen style" had no evidence. Four
caricature references with deliberately non-human proportions -- inflated cheeks
and jaw, a widened smile, an exaggerated brow -- were run on 5 identities x 7
emotions = 140 renders, paired against the same unstyled baseline.

| | OOD (n=140) | in-distribution (n=882) |
|---|---|---|
| colour adoption | **65.6%** (per style 42.0-78.2%) | 69.9% (48.8-88.1%) |
| content->style shape gap | **2.14 mm** (1.95-2.56) | 2.05 mm |
| as % of head diagonal | 0.48% | 0.46% |
| EmoAcc | **38.6%** | 33.2% |
| EmoAcc, unstyled baseline (same 5 ids) | 37.1% | -- |

**Colour adoption generalises.** 65.6% vs 69.9%; three of the four sit inside
the in-distribution per-style range. The one below it (42.0%) is the most
photorealistic of the four, i.e. the one with least colour distance to close.

**Emotion survives an unseen style**, at 38.6% against a 37.1% unstyled
baseline on the same identities.

**The geometry result is the substantive one.** These are caricatures whose
proportions are visibly non-human, and their tracked shape gap (2.14 mm) is
indistinguishable from ordinary portraits (2.05 mm). This converts the
tracking-prior limitation from an inference into a measurement: FLAME tracking
projects *any* reference onto the nearest plausible human head, and a
caricature's exaggeration is discarded before the blend sees it. Report this
alongside the 2.05 mm figure -- it is the strongest available evidence for why
geometry stylisation is complete yet small.


### 2d. JoJoGAN baseline — same content, same styles, same metrics

JoJoGAN (ECCV 2022) is the only prior method found whose input shape matches
ours exactly: one content photograph plus one style reference. We ran the
official code (github.com/mchong6/JoJoGAN, StyleGAN2-FFHQ + e4e inversion,
300 fine-tuning steps per style, `preserve_color=False`) on the **same 18
content photographs and same 7 style references** used for our own evaluation.

Every metric is computed by the same code for both systems, and always against
**that system's own un-stylised output** — our unstyled render, JoJoGAN's e4e
reconstruction. Each therefore measures "how far did this system move from its
own starting point toward the style", not an absolute distance between two
different renderers.

| metric | ours | JoJoGAN |
|---|---|---|
| colour adoption, face pixels | **50.9%** (sd 27.7) | 19.6% (sd 54.2) |
| PSNR vs own unstyled | 13.44 dB | 12.40 dB |
| SSIM vs own unstyled | **0.585** | 0.504 |
| LPIPS vs own unstyled | 0.282 | 0.363 |
| CLIP similarity to style | 0.744 | **0.754** |
| FaceNet identity vs own unstyled | 0.445 | **0.482** |
| n (content x style) | 126 | 126 |

**Scope — state this whenever the table is used.** JoJoGAN produces a single 2D
image. It has no 3D representation, no animation, and no emotion control, so
this compares **stylisation quality on one frontal frame only**. The 3D
consistency, drivability and emotion axes of our system have no baseline at all;
that is a statement about the literature, not a favourable comparison.

**Reading the numbers.** We transfer more colour onto the face (50.9% vs 19.6%)
and JoJoGAN's per-case variance is very large (sd 54.2 — on some pairs it moves
*away* from the reference). JoJoGAN scores marginally higher on CLIP similarity
to the style and on identity retention, which is consistent with it restyling
the whole frame including background and hair while changing the face itself
less.

**Measurement note, because the first attempt was wrong.** Measuring colour
adoption against our FLAME-tracked style crop (matted onto white) gave JoJoGAN
2.4%, which contradicted its visibly strong stylisation and its higher CLIP
score. The bias was the background: our renders share the white matte, JoJoGAN's
do not. The table above detects the face box with MTCNN in every image and
measures inside it only, excluding background for both systems by the same rule.
Note this also moves our own figure from the 69.9% of section 2b (whole matted
head, background dropped) to 50.9% here (tight face box) — the two measure
different regions and should not be quoted interchangeably.


### 2e. Baselines attempted and not completed

**SigStyle (AAAI 2025) — attempted, could not be run faithfully.** Setup got as
far as a running training loop; three blockers stopped a faithful run:

1. *Framework version conflict.* `shapeinv/` imports
   `diffusers.models.cross_attention`, removed after diffusers 0.14, while the
   repo ships diffusers 0.27.2. With diffusers 0.14 the code loads, but on
   torch 2.x diffusers selects `AttnProcessor2_0`, which does not accept their
   custom `appearance_feature_dict` argument. Forcing their own
   `CrossAttnProcessor` fixes it but disables scaled-dot-product attention.
2. *Memory.* With their own processor the teacher-student architecture (two SD
   UNets plus CLIP vision) exceeds 24 GB even at resolution 256, batch 1,
   gradient accumulation 1. Their `SlicedAttnProcessor` would reduce memory but
   does not accept `appearance_feature_dict`, so fitting it would require
   writing a sliced version of their custom processor.
3. *Repo bug.* `utils_wy.backup_files` copies the source tree into the output
   directory, which by default lives inside the source tree, producing
   unbounded path nesting (`OSError: File name too long`). Worked around by
   placing the output directory outside the repo.

Blocker 2 is the one that matters: a faithful run needs a larger GPU (40 GB+),
or a reimplementation of their attention processor. Any number obtained from a
modified processor at reduced resolution would be a measurement of our
reimplementation, not of SigStyle, so none is reported.

### 2b. Prior work (context only — different datasets and protocols)

| metric | paper | venue | value |
|---|---|---|---|
| LPIPS | SigStyle | AAAI 2025 | 0.5191 |
| Style loss | SigStyle | AAAI 2025 | 0.7641 |
| Rank-1 preference | SigStyle | AAAI 2025 | 9.1% |
| RMSE (short-term, LLFF) | G-Style | CGF 2024 | 0.039 (best baseline) |
| LPIPS (short-term, LLFF) | G-Style | CGF 2024 | 0.019 (best baseline) |

---

## 3. Reproducing these numbers

```bash
# render both emotion arms (98 renders each, ~20-30s per render)
./run_styled_emotion_eval.sh exps/images/emoeval_unstyled
./run_styled_emotion_eval.sh exps/images/emoeval_styled_pop assets/sample_input/pop.png

# score both arms with both judges -> table 1b
lam_env/bin/python score_emotion_arms.py \
  unstyled=exps/images/emoeval_unstyled styled=exps/images/emoeval_styled_pop

# judge calibration + stylisation robustness -> table 1a
lam_env/bin/python score_stylized_photo_ceiling.py assets/sample_input/pop.png

# qualitative figures: best-of-N selection (nondeterminism makes this necessary)
lam_env/bin/python run_emotion_best_of_n.py \
  assets/sample_input/cluo.jpg assets/sample_input/pop.png fig_out -n 8
```

The 14 identities are `sorted(original-image/*.jpg)[:14]`. The renamed copies
used for the 09-01 run were deleted; the mapping was recovered by FaceNet
matching each unstyled *neutral* render to its source photograph (14/14 unique,
cosine margin 0.22-0.58, and the recovered order was exactly sorted order).

Re-run the judge calibration (1a) first if any judge or checkpoint changes,
since every other emotion number depends on it.

---

## 4. Protocol comparison with prior work

**These are protocol descriptions, not a leaderboard.** None of the numbers
below can be placed in a common column with ours, for reasons stated in the
last column. Bibliographic details in this table came from a literature search
and have **not been verified against the papers' PDFs** -- verify before citing.

| method | venue | driving signal | one-shot? | benchmark | why not directly comparable to ours |
|---|---|---|---|---|---|
| EmoTaG | CVPR 2026 | audio + 6 AUs | no | MEAD | reconstructs from a full driving video of the same identity; dense per-frame supervision |
| EmoHead | - | audio | - | MEAD | audio-driven |
| EDTalk | - | audio | - | MEAD (M003, M030, W009, W015) | audio-driven |
| EmoSpeaker | IEEE Trans. 2026 | audio | **yes** | MEAD (train 10M+10F, test 2M+2F), CREMA-D, HDTF | closest input format to ours, but lip motion is generated from audio, not replayed from a driving video |
| PortraitTalk | - | audio | - | MEAD (M003, M009, M030, W015) | audio-driven |
| GemTalk | 2026 | audio | - | - | cited for architecture, not benchmark: emotion-agnostic backbone + emotion module on top, structurally the same frozen-backbone-plus-adapter design as ours |
| InstructAvatar | AAAI 2025 | text instruction | - | not checked | different control modality |
| **ours** | - | **video (tracked FLAME expr/jaw)** + discrete emotion class | **yes** | RAVDESS / AffectNet-HQ / in-the-wild portraits | **no audio subsystem anywhere in the pipeline** |

### 4a. Why the MEAD protocol was not adopted

Two independent blockers:

1. **MEAD is not available locally.** Neither are CREMA-D or HDTF. MEAD is
   access-gated (registration required), so it cannot simply be fetched.
2. **Even with MEAD in hand the comparison would not be fair.** Every method
   above generates lip motion *from audio*. This pipeline has no audio
   subsystem at all -- it replays a driving video's tracked FLAME parameters
   and adds a discrete emotion offset. Running their split would produce a
   number that *looks* comparable and is not, which is worse than not
   reporting one. The size of that trap is already measured: against EmoTaG the
   self-reconstruction LMD gap is ~30x, driven entirely by single-image vs
   full-video supervision (section 1d).

### 4b. The MEAD split discrepancy

Two different "standard" MEAD test splits appear in the sources consulted:

* EDTalk: M003, M030, **W009**, W015
* PortraitTalk: M003, **M009**, M030, W015

These differ in one identity and cannot both be the same convention. This was
not resolved -- it needs the actual PDFs, not search snippets. **Do not present
either as the universal standard.** If a MEAD evaluation is ever run, phrase it
as "we follow the held-out-identity convention used in prior work [cite both],
using split X", naming the split actually verified and used.

### 4c. What this pipeline can claim instead: identity-disjoint evaluation

The shipped emotion adapter checkpoint
(`exps/train_lam/emotion_adapter_symmetric`) is trained from
`tracked_emotion_refs_closedmouth.json`: **582 references, all AffectNet-HQ
derived, containing zero RAVDESS images** (verified by inspecting every path in
the manifest). The RAVDESS frames in `data/emotion_refs/images/` were never
used for this checkpoint.

So all **6 RAVDESS actors (a01, a02, a05, a06, a09, a10)** used in the judge
calibration and self-reconstruction tables are **held out by construction** --
no identity, and no dataset, overlap between training and those evaluations.
That is the same property a MEAD held-out-identity split is designed to
guarantee, and here it is stronger: the training and test sets are different
corpora entirely. This is worth stating explicitly in the paper; it costs
nothing and it is verifiable from the manifest.

---

## 5. Stylisation: comparison with prior work

**Protocol/capability comparison, not a leaderboard.** Bibliographic details
came from a literature search and are **not verified against the papers' PDFs**.

| method | venue | dim | input | geometry? | emotion? | code public | runnable here |
|---|---|---|---|---|---|---|---|
| JoJoGAN | ECCV 2022 | 2D | 1 content face + 1 style ref | no | no | **yes** (StyleGAN2) | **yes, with setup** |
| StyO | AAAI 2025 | 2D | one-shot face stylisation | claims geometry variation | no | not checked | unknown |
| DINO deformable one-shot | CVPR 2024 | 2D | one-shot | deformation | no | not checked | unknown |
| SigStyle | AAAI 2025 | 2D | sig. style transfer | no | no | not checked | unknown |
| StyleID | arXiv 2026 | - | *dataset + metric* for identity under stylisation | - | - | not checked | would replace our identity metric |
| G-Style / StyleGaussian | CGF 2024 / - | 3D scene | multi-view capture | n/a | no | - | **no** -- needs multi-view scenes, not a single face |
| **ours** | - | **3D Gaussian avatar** | 1 content face + 1 style ref | **FLAME betas blend** | **7-class** | - | - |

**JoJoGAN is the only genuinely fair, runnable baseline in this list.** It takes
exactly our input shape (one content face + one style reference) and its code
and weights are public. It is 2D-only with no geometry and no emotion, so it
cannot be compared against the full pipeline -- but on the narrow question
"given a face photo and a style image, how good is the stylisation", it is an
honest apples-to-apples test, and the first one found in this project.

### 5a. What is on disk for such a comparison

| asset | status |
|---|---|
| AAHQ (style pool) | **9,598 images, 11 GB** -- present |
| CelebA-HQ 256 (content pool) | **7,236 images** -- present |
| FFHQ (JoJoGAN's training set) | **absent** (2 files); `ffhq-dataset/download_ffhq.py` + 256 MB metadata present, so fetchable |
| JoJoGAN code/weights | **absent** |
| StyleID dataset/metric | **absent**, availability unchecked |

Content and style pools are already here, so a JoJoGAN comparison needs only
JoJoGAN itself -- FFHQ is its *training* set and is not required to run
inference with the released weights.

Realistic cost: JoJoGAN depends on StyleGAN2's custom CUDA ops, which are
notoriously fragile to compile against a specific torch/CUDA pair (this env is
torch 2.3.0+cu121). Budget setup risk, not just setup time.

### 5b. Our measured stylisation numbers (for whatever comparison is run)

From section 2a and the 882-render bigrun:

| quantity | value |
|---|---|
| colour adoption (gap closure vs own unstyled render) | **69.9%** (per-style 48.8-88.1%, n=882) |
| geometry adoption | **100%** by construction; the gap closed is **2.05 mm** = 0.46% of head diagonal |
| LPIPS vs unstyled reconstruction | 0.255 |
| SSIM / PSNR vs unstyled reconstruction | 0.607 / 13.49 dB |
| CLIP similarity, output <-> style ref | 0.713 |
| emotion accuracy cost of stylisation | **-0.9 points** (n=882 vs 126, not significant) |

The identity-preservation metric currently used is CLIP/ArcFace cosine, which
was not designed for stylised faces. **StyleID is purpose-built for exactly
that question** and would be a genuine rigour upgrade if its dataset and code
turn out to be released -- worth checking before committing to the current
metric.

