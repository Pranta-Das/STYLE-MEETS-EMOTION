"""Discrete-class emotion control for driven avatar animation.

Standalone module, deliberately NOT wired into ModelLAM -- unlike style
(color/geometry), which has to reach into the shared per-point latent
features and the FLAME shape (betas) blend, emotion only ever needs to edit
two things that are already plain tensors flowing through
lam/runners/infer/lam.py before the model is called at all:
`flame_params["expr"]` ([1, N_frames, 100]) and `flame_params["jaw_pose"]`
([1, N_frames, 3]). Editing those directly, once, before
`model.infer_single_view` is called, needs zero changes to modeling_lam.py
or gs_renderer.py -- so stylization code stays completely untouched (see
STYLE_GEOMETRY_PIPELINE.md, which this deliberately does not modify).

Design note on why this is a trained network and not just a per-class
average: a network that only takes an emotion class id as input, with no
other conditioning, would just be re-deriving the per-class average of its
training targets -- no better than literally averaging tracked reference
photos by hand. `EmotionAdapter` also conditions on the current driving
clip's own base `expr` state, so the same emotion selection produces a
context-appropriate shift instead of a fixed additive offset (see
tools/train_emotion_adapter.py for how training data is constructed to
support this: multiple synthetic base-context vectors per training target,
sampled from real per-frame trajectories in assets/sample_motion/, since the
underlying photos only give one static context each).
"""
import torch
import torch.nn as nn

EMOTION_CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# (emotion_strength, emotion_jaw_influence) used when the CLI doesn't pass an
# explicit value for a given run -- see lam/runners/infer/lam.py:parse_configs.
# The global default (1.0, 0.5) is right for classes with a bold anatomical
# signal (happy's smile, surprise's dropped jaw), verified via an independent
# pretrained expression classifier (trpakov/vit-face-expression) scoring real
# renders. It undershoots on emotions whose real signal is naturally subtle --
# a sad frown moves far less than a surprised mouth -- because the same
# relative push reads as strong on a bold emotion and as noise on a subtle
# one; this is on top of (not a substitute for) EmotionAdapter's own
# max_expr_delta being sized correctly, which fixes clipping/reshaping but
# not this per-class amplitude difference.
#
# 'sad' is the one class actually measured to need a boost: at the global
# default it read as "neutral" to the classifier (sad-score 0.088); (3.0,
# 1.0) nearly quadrupled that (0.334) and produced a render a person would
# call sad on sight (clear downturned mouth, drooping eyes) -- see
# STYLE_GEOMETRY_PIPELINE.md for the full before/after. 'fear' is the other
# measured case, paired with its own checkpoint override below (see
# PER_CLASS_ADAPTER_PATH) -- (1.5, 0.5) plus component sharpening lifted its
# score 20%->36% on that specific checkpoint; the same settings on the
# regular default checkpoint only reach 2%, so this value is meaningless
# without the matching checkpoint override, which is why both live together
# here rather than being tuned independently. The remaining four non-neutral
# classes are NOT individually verified at a boosted strength (only
# confirmed adequate at the global default for happy/surprise) -- left at
# the global default rather than guessed, since guessing a value that looked
# plausible was exactly what produced the original weak 'sad' default in the
# first place. 'neutral' is excluded on purpose: its trained delta is
# intentionally near-zero (it's the reference every other class's delta is
# measured against), so there's nothing for a higher strength to amplify.
GLOBAL_DEFAULT_STRENGTH = (1.0, 0.5)
PER_CLASS_DEFAULT_STRENGTH = {
    "sad": (3.0, 1.0),
    "fear": (1.5, 0.5),
}


def default_emotion_strength(emotion_class):
    """(emotion_strength, emotion_jaw_influence) for `emotion_class`, falling
    back to the verified-adequate global default for classes without their
    own measured override."""
    return PER_CLASS_DEFAULT_STRENGTH.get(emotion_class, GLOBAL_DEFAULT_STRENGTH)


# (emotion_optimize_reg, emotion_optimize_steps) for lam/stylization/emotion_optimize.py's
# per-instance test-time correction. One global value cannot serve every
# class: measured directly on the same (musk.jpg, one AAHQ style) pair --
# 'happy' at the old low reg (0.02) converged to a classifier-satisfying
# (99%+) but visibly wrong, asymmetric, smeared mouth (a real user-caught
# artifact, not hypothetical); raising reg to 20 fixed it (96.4% confidence,
# clean mouth) cheaply since happy only ever needed a small correction. 'sad'
# needed the opposite: at reg=20 with more steps it plateaus around 67% in
# the optimizer's own proxy and actually renders as only 8.3% sad in the
# real final image (top1 'neutral') -- the same regularization that fixes
# happy's overshoot caps off the much larger correction sad's identity-blind
# starting point needs. reg=1.0 (with steps raised to 100, since sad's loss
# barely moves for its first ~30 steps then transitions sharply -- 30 steps
# alone isn't enough regardless of reg) gets a clean 78.3% confidence, top1
# match, no visible artifact.
#
# Only happy and sad are measured this specifically -- the other five
# classes use the global default rather than a guessed override, same
# discipline as PER_CLASS_DEFAULT_STRENGTH above. Revisit if a similar
# artifact turns up on another class.
GLOBAL_DEFAULT_OPTIMIZE_REG = 5.0
PER_CLASS_OPTIMIZE_REG = {
    "happy": 20.0,
    "sad": 1.0,
}


def default_emotion_optimize_reg(emotion_class):
    """`emotion_optimize_reg` for `emotion_class`, falling back to a
    moderate, untuned global default for classes without their own
    measured override (see PER_CLASS_OPTIMIZE_REG)."""
    return PER_CLASS_OPTIMIZE_REG.get(emotion_class, GLOBAL_DEFAULT_OPTIMIZE_REG)


# `emotion_optimize_reg_jaw` -- see lambda_reg_jaw's docstring in
# lam/stylization/emotion_optimize.py for why jaw_delta needs its OWN
# stronger penalty instead of sharing expr_delta's. Defaulting this higher
# is a conservative safety choice: it stops the test-time optimizer from
# buying emotion confidence by cheaply over-opening or laterally rotating
# the jaw.
GLOBAL_DEFAULT_OPTIMIZE_REG_JAW = 50.0
PER_CLASS_OPTIMIZE_REG_JAW = {}


def default_emotion_optimize_reg_jaw(emotion_class):
    """`emotion_optimize_reg_jaw` for `emotion_class` -- see
    PER_CLASS_OPTIMIZE_REG_JAW."""
    return PER_CLASS_OPTIMIZE_REG_JAW.get(emotion_class, GLOBAL_DEFAULT_OPTIMIZE_REG_JAW)


# No single trained checkpoint was best for every class -- measured directly,
# not assumed. The "hybrid" checkpoint (trained on AffectNet-HQ references
# filtered by agreement with an independent pretrained classifier) is best
# for sad/happy/neutral/surprise. A separate "discriminative" checkpoint
# (trained with discriminative_trim_indices in tools/train_emotion_adapter.py,
# which selects references that are distinctively THEIR class and not just
# internally self-consistent) is worse for sad (0.334 -> 0.114 at its best
# tested strength) but is the one that makes fear's sharpening work at all
# (0.02 on the hybrid checkpoint vs 0.36 on this one, same settings) --
# discriminative training measurably cut anger/disgust's cosine similarity
# (0.77 -> 0.34), which is also what separated fear from surprise enough for
# sharpening to have a real distinctive direction to amplify. Per-class
# override, same precedence pattern as PER_CLASS_DEFAULT_STRENGTH: only
# applied when the CLI doesn't pass emotion_adapter_path explicitly.
#
# 'anger' -> discriminative was tried (and briefly defaulted on) after
# tracing a real user-caught artifact on musk.jpg: on the hybrid checkpoint,
# anger's raw prediction (BEFORE emotion_optimize even runs -- confirmed
# with emotion_optimize=false and separately with emotion_jaw_influence=0,
# ruling out both the per-instance optimizer and jaw_pose specifically)
# rendered with the mouth held open, resembling disgust/fear/sad/surprise's
# raw predictions on the same checkpoint (matches the 0.77 anger/disgust
# cosine similarity noted above). On the discriminative checkpoint, that one
# identity's anger came out clean (closed mouth, 98.4% confidence). But
# re-run across the original 14-identity validation set (no style,
# emotion_optimize=true, same protocol as the 196-run table below) it's a
# net REGRESSION, not a fix: mean confidence 84.4% -> 69.8%, match rate
# 13/14 -> 10/14 -- three additional identities that worked fine on hybrid
# (original_008, 009, 013) came out top1='neutral' on discriminative, and
# original_001 flipped from a correct match to top1='fear'. musk.jpg's own
# specific failure mode (an open-mouth shortcut baked into the trained
# prediction) is real, but it is NOT representative of how hybrid's anger
# behaves on most identities -- reverted to hybrid as the default on that
# basis. If this resurfaces on a specific identity, override per-request
# with emotion_adapter_path rather than changing the global default again
# without re-validating broadly.
# Disgust was tested the same way (discriminative + emotion_optimize) and
# was also worse there (confidence collapsed to 0.8%, mouth rendered
# stretched/distorted) -- never defaulted, consistent with disgust having
# no strong distinctive direction on EITHER checkpoint (see
# STYLE_GEOMETRY_PIPELINE.md's disgust ceiling finding).
#
# SUPERSEDED by emotion_adapter_closedmouth below -- kept as the fallback
# constant name/docstring for context, not the active default.
_HYBRID_ADAPTER_PATH = "exps/train_lam/emotion_adapter_hybrid/emotion_adapter.pt"
_DISCRIMINATIVE_ADAPTER_PATH = "exps/train_lam/emotion_adapter_discriminative/emotion_adapter.pt"

# 'emotion_adapter_closedmouth': retrained after tracing anger/disgust's
# shared "mouth open" artifact all the way to its root. Measured with the
# actual FLAME head model (not a jaw_pose proxy -- confirmed jaw_pose alone
# doesn't capture it, expr_delta carries some of the signal too): inner-lip
# landmark distance across all 683 tracked training references showed
# anger (mean 0.0101) and disgust (0.0099) at roughly double neutral's
# (0.0046) and close to fear's (0.0122) -- i.e. most of the *source photos*
# for anger/disgust are open-mouth shouting/snarling shots, not the
# closed-mouth tension a person actually expects. fear/surprise's own high
# mouth-openness (0.0122/0.0157) is correctly part of THEIR signature and
# was left untouched. Filtered out every anger/disgust reference above
# mouth_open=0.006 (keeps 54/92 anger, 36/99 disgust; script:
# measure_mouth_opening.py + the filtering step in this session's history,
# manifest at exps/train_lam/emotion_refs/tracked_emotion_refs_closedmouth.json)
# and retrained from scratch with the same discriminative trim as every
# other checkpoint here. Effect: anger/disgust cosine similarity 0.77 ->
# 0.31 (real separation, not just self-consistency).
#
# Re-validated across the same 14-identity x 7-class protocol as the
# 196-run table above (no style, emotion_optimize=true) -- a broad win, not
# just an anger fix: match rate 67/98 -> 79/98, mean confidence 62.1% ->
# 74.5%. Per class (before -> after): anger 92.9%->85.7% (roughly flat),
# disgust 7.1%->21.4% (better, still weak), fear 85.7%->92.9%, happy
# 85.7%->100%, neutral 78.6%->85.7%, sad 50.0%->92.9% (the single biggest
# jump), surprise 71.4%->85.7%. Also beats the old fear-specific
# discriminative-checkpoint-plus-sharpening combo (~40% in earlier testing)
# without needing that override at all -- CLASSES_WITH_DISTINCTIVE_SHARPENING
# sharpening still applies on top automatically (see lam/runners/infer/lam.py)
# and was active during this validation. Promoted to the global default on
# this basis; PER_CLASS_ADAPTER_PATH's fear override removed since this
# checkpoint alone now outperforms it.
#
# Disgust remains the one class that's still mostly wrong (21.4% match) --
# meaningfully better than before (7.1%) but not fixed. Consistent with
# (not contradicting) the standing diagnosis in STYLE_GEOMETRY_PIPELINE.md:
# the mouth-open training bias was PART of disgust's problem, not all of
# it -- FLAME's expression PCA basis likely still doesn't carve out
# disgust's specific muscle pattern (nose wrinkle / upper-lip raise) as a
# direction separable from other negative-valence expressions, regardless
# of how clean the training data is.
#
# 'emotion_adapter_v2' (closed-mouth data + 6-actor RAVDESS augmentation)
# was tried and measured worse on the same 14-identity protocol: mean
# confidence 74.5% -> 70.9%, match rate 79/98 -> 76/98. sad lost 2 matches,
# anger lost 2, happy lost 1 -- disgust gained only 1, nowhere near enough
# to offset the rest. Root cause visible before even running the
# validation: RAVDESS's fear/sad expr-delta cosine similarity came out at
# +0.78 (vs -0.02 to +0.22 on every other checkpoint here), i.e. the two
# classes stopped being separable in the retrained network. Likely cause:
# only 6 of RAVDESS's 24 actors were used (bandwidth-limited download),
# so fear/sad's directions may be dominated by a couple of actors' similar
# acting style rather than reflecting the classes generally -- a larger
# actor sample, or apex-frame selection smarter than a fixed 80%-of-clip
# heuristic, might avoid this, but that's unverified, not assumed. Not
# defaulted; kept at exps/train_lam/emotion_adapter_v2/ for reference if
# someone wants to pick this back up with more actors.
#
# 'emotion_adapter_symmetric': the closed-mouth manifest retrained with
# --symmetrize_targets 1.0, i.e. every reference's target delta made
# laterally symmetric BEFORE trimming/regression. Motivation was a
# user-visible artifact -- the mouth pulling sideways by a different amount,
# in a different direction, for every emotion class. Root cause measured:
# the asymmetric residual (d - M@d) was 35-72% of each class's delta norm on
# both prior checkpoints (anger 59.5%, happy 52.6%, surprise 51.6%), plus a
# per-class jaw yaw. That is one photo's worth of personal asymmetry and
# tracking noise being learned as if it were the emotion.
#
# Effect on the deltas: asymmetric share drops ~4x (anger 59.5% -> 11.5%,
# happy 52.6% -> 19.6%, surprise 51.6% -> 12.1%, sad 35.5% -> 8.7%), jaw
# yaw/roll become exactly 0.0 for every class, and delta magnitudes are
# preserved (4.4-5.0), so no expression strength is lost.
#
# Rendered effect (cluo.jpg + pop.png, lower-lip deviation from the source
# photo's own baseline, %IOD): cross-emotion SPREAD -- the actual complaint
# -- 6.59 -> 4.92, surprise 6.63 -> 0.76, neutral 1.03 -> 0.04, sad
# 5.96 -> 4.68. anger (2.19 -> 4.96) and happy (0.05 -> 2.37) got worse, so
# this is a net improvement in consistency, not a uniform win.
#
# Recognition cost on the standard 14-identity x 7-class protocol: a wash --
# 74.5% -> 74.0% mean confidence, 79/98 -> 77/98 match. surprise gained
# (12/14 -> 14/14, 74.1% -> 86.2%) and fear lost (13/14 -> 11/14, 86.0% ->
# 72.0%). NOTE that fear's loss is confounded: that run also had
# CLASSES_WITH_DISTINCTIVE_SHARPENING emptied, and sharpening is what
# historically carried fear. If fear recognition matters more than its mouth
# stability for a given use, re-enable sharpening for it at a reduced boost
# (emotion_sharpen_boost ~1.5 rather than 2.5, which is what doubled its
# delta norm) -- untested, and it should get its own validation run.
#
# Final default: use the symmetric-target retrain. Later validation in
# STYLE_GEOMETRY_PIPELINE.md found recognition essentially unchanged within
# noise while cross-emotion mouth consistency improved, and the learned
# jaw yaw/roll deltas become exactly zero. This does not fully solve the
# Gaussian decoder's own mouth distortion, but it removes one major source
# of emotion-specific sideways pull before inference starts.
GLOBAL_DEFAULT_ADAPTER_PATH = "exps/train_lam/emotion_adapter_symmetric/emotion_adapter.pt"
PER_CLASS_ADAPTER_PATH = {}


def default_emotion_adapter_path(emotion_class):
    """Checkpoint path for `emotion_class`, falling back to the
    best-overall checkpoint for classes without their own measured
    override."""
    return PER_CLASS_ADAPTER_PATH.get(emotion_class, GLOBAL_DEFAULT_ADAPTER_PATH)


# Classes confirmed to actually benefit from sharpening -- not just classes
# that have a real distinctive component on paper. That distinction matters:
# anger's single most distinctive component also measured strong (magnitude
# 2.71, same-vs-other-classes gap 0.99, comfortably above disgust's 0.30) and
# was tried extensively -- default sharpening (top_k=8, boost=2.5), then
# deliberately extreme (top_k=15, boost=5.0, expanding the delta's own norm
# from 8.5 to 26.7) at multiple emotion_strength levels up to 3.0. None of it
# moved anger's classifier score off ~0.5-1.7% or produced a render a person
# would call visibly angrier. Only fear's score actually moved (0.02 -> 0.36)
# under the same mechanism. Read together with PER_CLASS_ADAPTER_PATH's
# finding that the discriminative checkpoint didn't fix anger either (still
# reads as neutral): the anger direction that exists in this data, however
# amplified, does not correspond to a strong visible facial change in this
# rendering pipeline -- a rendering/representation ceiling, not a
# scaling problem sharpening (or any other scaling knob) can reach.
# Left anger OUT of this set on that basis, so it isn't silently "sharpened"
# for no measured benefit; see STYLE_GEOMETRY_PIPELINE.md for the full
# investigation, including the ceiling this points to for 'disgust' too.
#
# UPDATE -- emptied. Sharpening was added when fear scored ~2% and needed
# rescuing on a much weaker checkpoint. It does not merely reweight the
# delta, it nearly DOUBLES its norm (measured on the current checkpoints:
# fear |d| 4.08 -> 7.54 hybrid, 4.50 -> 8.26 symmetric) and more than
# doubles its lateral asymmetry (12.3% -> 28.7% on the symmetric
# checkpoint), because the "most distinctive" components it boosts are
# themselves largely asymmetric. Stacked on emotion_strength=1.5 that is
# ~2.8x the raw delta, which is what produced the visibly twisted,
# diagonally-skewed fear mouth a user reported. Fear no longer needs the
# rescue: it reached 86.0% mean confidence / 13-14 match on the
# closed-mouth checkpoint. Measured effect of removing it on cluo.jpg:
# lower-lip deviation from the source photo 3.69 -> 0.52, the best of any
# class in that sweep. The machinery is kept (distinctive_component_mask
# still works, and emotion_sharpen_boost is still a CLI knob) but no class
# opts into it by default.
#
# Final default: no class is sharpened. Fear sharpening was useful for an
# older, weaker checkpoint, but on the symmetric checkpoint it over-amplifies
# asymmetric components and is a direct cause of diagonally skewed mouths.
# The function remains available for explicit experiments.
CLASSES_WITH_DISTINCTIVE_SHARPENING = set()


def distinctive_component_mask(model, target_class, context, top_k=8, boost=2.5):
    """Per-component multiplier (shape [expr_dim]) that is `boost` at the
    `top_k` expr components most distinctively characteristic of
    `target_class` (see CLASSES_WITH_DISTINCTIVE_SHARPENING) and 1.0
    everywhere else. Multiply the adapter's raw expr_delta by this before
    applying emotion_strength, so scaling doesn't dilute the one real signal
    a class has by spreading it evenly across components it shares with
    other classes.

    `context` is whatever base_expr the caller is about to condition on --
    distinctiveness is computed fresh against it rather than cached, since it
    depends on all seven classes' deltas at that same context.
    """
    device = context.device
    with torch.no_grad():
        ids = torch.arange(len(EMOTION_CLASSES), device=device)
        ctx = context.expand(len(EMOTION_CLASSES), -1)
        all_expr_delta, _ = model(ids, ctx)
    target_idx = EMOTION_CLASSES.index(target_class)
    own = all_expr_delta[target_idx].abs()
    others = torch.cat([all_expr_delta[:target_idx], all_expr_delta[target_idx + 1:]], dim=0).abs()
    distinctiveness = own - others.max(dim=0).values
    top = torch.topk(distinctiveness, k=min(top_k, distinctiveness.shape[0])).indices
    mask = torch.ones_like(own)
    mask[top] = boost
    return mask


class EmotionAdapter(nn.Module):
    def __init__(self, num_emotions=len(EMOTION_CLASSES), embed_dim=32,
                 expr_dim=100, jaw_dim=3, max_expr_delta=3.0, max_jaw_delta=0.15):
        super().__init__()
        self.expr_dim = expr_dim
        self.jaw_dim = jaw_dim
        self.emotion_embed = nn.Embedding(num_emotions, embed_dim)

        # tanh-capped affine, same shape as ReferenceStyleAdapter's
        # max_appearance_scale/max_query_offset -- bounds how far a single
        # emotion selection can push expr/jaw away from the driving video's
        # own values, however confident the network gets.
        #
        # max_expr_delta must stay ABOVE the largest per-component value in
        # the tracked targets, or the cap stops being a safety bound and
        # starts reshaping the delta. It is a PER-COMPONENT clamp, not a
        # norm clamp, so a too-small cap does not merely shrink the
        # expression -- it flattens the leading FLAME expression components
        # (which carry jaw drop / brow raise / eye widening, and are what
        # makes an emotion readable) down to the same magnitude as the
        # near-noise tail components. Measured on the tracked AffectNet
        # refs, the trimmed per-class mean deltas peak at 1.07 (sad) to
        # 2.54 (happy), with surprise at 1.97 on component 8 and 1.54 on
        # component 3. The original 0.6 therefore clipped every class:
        # surprise lost ~35% of its total amplitude and, worse, half its
        # remaining energy moved out of components 0-9 into the tail
        # (||delta[0:10]|| 3.05 -> 1.52 while ||delta[50:100]|| 1.23 ->
        # 1.32), which reads as a mushy, non-specific face rather than a
        # surprised one. 3.0 clears the largest observed target with
        # headroom and leaves tanh out of its saturating region.
        #
        # Note this buffer is persistent, so it is baked into every
        # checkpoint: raising the default here does NOT loosen an adapter
        # trained at 0.6 -- load_state_dict restores the old value. The
        # cap change only takes effect on a retrained checkpoint.
        #
        # emotion_strength at inference multiplies the delta AFTER this cap
        # (see lam/runners/infer/lam.py), so the cap only ever needs to
        # cover the raw target -- it does not need extra room for strength.
        self.register_buffer("max_expr_delta", torch.tensor(float(max_expr_delta)), persistent=True)
        self.register_buffer("max_jaw_delta", torch.tensor(float(max_jaw_delta)), persistent=True)

        hidden = embed_dim + expr_dim
        self.expr_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, expr_dim),
        )
        self.jaw_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, jaw_dim),
        )
        # Zero-initialized final layer: an untrained/freshly-loaded adapter
        # is a no-op (delta 0) rather than injecting noise -- same
        # safe-by-construction init used for appearance_mlp/geometry_mlp.
        nn.init.zeros_(self.expr_head[-1].weight)
        nn.init.zeros_(self.expr_head[-1].bias)
        nn.init.zeros_(self.jaw_head[-1].weight)
        nn.init.zeros_(self.jaw_head[-1].bias)

    def forward(self, emotion_id, base_expr):
        """
        emotion_id: LongTensor, any shape, values in [0, num_emotions).
        base_expr: [..., expr_dim] -- current driving clip's own expr
            (e.g. mean over frames, or a per-frame value).
        Returns (expr_delta, jaw_delta), broadcastable onto base_expr's
        leading dims: expr_delta [..., expr_dim], jaw_delta [..., jaw_dim].
        """
        embed = self.emotion_embed(emotion_id)
        ctx = torch.cat([embed, base_expr], dim=-1)
        expr_delta = torch.tanh(self.expr_head(ctx)) * self.max_expr_delta
        jaw_delta = torch.tanh(self.jaw_head(ctx)) * self.max_jaw_delta
        return expr_delta, jaw_delta
