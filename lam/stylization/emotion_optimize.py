"""Per-instance (test-time) correction of EmotionAdapter's predicted delta.

EmotionAdapter (lam/models/emotion_adapter.py) is trained to predict one
(expr_delta, jaw_delta) per emotion class conditioned only on
(class, driving-clip's own expr) -- it has no way to know the target
person's face shape. FLAME's expression components are linear vertex
offsets, so the exact same delta can read as a frown on one face and a
grinning smile on another: measured directly on one real photo, a full
strength sweep (-3.0 to 3.0x) of the trained 'sad' direction never once
produced a render that an independent classifier read as sad -- it just
slid from neutral (weak) to happy (strong), meaning the DIRECTION itself,
not its magnitude, is wrong for that face (see STYLE_GEOMETRY_PIPELINE.md's
"sad-smile bug" section for the full diagnosis). No scalar strength/sign
knob can fix that.

This is the same problem optimize_style_colors (color_optimize.py) already
solves for color: a small trained network tries to generalize across many
identities from limited data and doesn't always land on the right answer for
a SPECIFIC one, so instead of trusting the network's raw prediction, directly
optimize a per-instance correction against a real target for this exact
face. Here the "real target" is an independent pretrained expression
classifier's own confidence that the render looks like the requested class,
backpropagated through the (frozen) renderer into (expr_delta, jaw_delta).

Mirrors color_optimize.py's structure: the expensive frozen backbone
(image encoding, latent_points, forward_gs) is computed once under
no_grad; only the cheap per-view animate+render step re-runs each
optimization step, with gradients flowing into expr_delta/jaw_delta only
-- model weights and the classifier are both frozen.
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange

# trpakov/vit-face-expression's own label ordering (config.id2label) and how
# it maps onto this project's EMOTION_CLASSES naming ("angry" -> "anger").
_VIT_ID2LABEL = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
_VIT_LABEL_TO_EMOTION = {
    "angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "sad": "sad", "surprise": "surprise",
}
EMOTION_TO_VIT_ID = {_VIT_LABEL_TO_EMOTION[label]: idx for idx, label in _VIT_ID2LABEL.items()}

_classifier_cache = {}


def _get_classifier(device):
    """Loaded once per device and cached -- same judge model used throughout
    this project's evaluation (trpakov/vit-face-expression), reused here as a
    differentiable optimization target instead of just a scoring tool."""
    key = str(device)
    if key not in _classifier_cache:
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression").to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _classifier_cache[key] = model
    return _classifier_cache[key]


def _classifier_preprocess(rgb):
    """Differentiable equivalent of ViTImageProcessor's own preprocessing
    (resize 224x224 bilinear, normalize with mean=std=0.5) -- done in torch
    directly on the rendered tensor so gradients flow through, instead of
    round-tripping through PIL like the plain scoring scripts do."""
    x = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - 0.5) / 0.5


def optimize_emotion_expr(
    model,
    image,
    flame_params,
    render_c2ws,
    render_intrs,
    render_bg_colors,
    view_idx,
    render_h,
    render_w,
    target_emotion,
    init_expr_delta,
    init_jaw_delta,
    style_image=None,
    style_shape_params=None,
    style_geometry_strength=0.0,
    style_opacity_strength=0.0,
    color_style_override=None,
    num_steps=30,
    lr=0.05,
    lambda_reg=20.0,
    lambda_reg_jaw=None,
    project_fn=None,
    log_fn=None,
):
    """Returns (expr_delta, jaw_delta) corrected for THIS specific face,
    starting from the adapter's own prediction and only ever pulled toward
    higher classifier confidence for `target_emotion` on the actual
    rendered output -- not toward a fixed target vector.

    `lambda_reg` penalizes drifting from the adapter's initial prediction,
    both to keep runs reproducible/bounded and to reduce the risk of the
    optimizer finding a classifier-fooling artifact (a real risk with
    unconstrained optimization against a single scorer) rather than a
    genuinely more legible expression -- verify the result by eye, the
    same discipline used everywhere else in this project. This isn't
    hypothetical: at the old default (0.02) a styled 'happy' case scored
    99%+ while rendering a visibly wrong, asymmetric, smeared mouth -- the
    penalty is a MEAN over all 100 expr components, so it's diluted by the
    ~90 that barely move and did nothing to constrain the handful that
    actually drove the distortion. 10x and 100x that default barely changed
    the render; only ~20+ was a real constraint (raised as the new default),
    and even then some residual asymmetry can remain -- this raises the
    floor, it doesn't guarantee an artifact-free result every time.

    `project_fn(expr_delta, jaw_delta) -> (expr_delta, jaw_delta)` is applied
    in-place after every optimiser step, so the search stays inside the
    feasible region rather than being clipped back into it once at the end.
    It must be a true projection (idempotent); pass the hard clamps only, not
    a symmetry blend, which is a contraction and would compound over steps.

    `lambda_reg_jaw` penalizes drifting jaw_delta SEPARATELY from expr_delta
    (defaults to `lambda_reg` if not given, i.e. the old undifferentiated
    behavior). Needed because jaw_delta is only 3 numbers with outsized
    visual leverage (it controls how wide the whole mouth opens) while
    expr_delta is 100 -- sharing one combined mean-squared penalty lets the
    optimizer dump most of its "budget" into jaw_delta almost for free,
    since one lever moving a lot barely shows up in a 100-wide mean. Measured
    directly: with a shared reg, anger/disgust/fear/sad/surprise all
    converged to a near-identical mouth-ajar/teeth-visible look instead of
    each class's own distinguishing shape -- the classifier itself has a
    real bias toward reading "mouth open" as more emotionally intense for
    almost every non-neutral class, and jaw_delta was the cheapest way to
    exploit that shared bias. Decoupling the two lets jaw get its own
    (typically stronger) constraint without weakening expr_delta's ability
    to find each class's actual distinctive shape.
    """
    if lambda_reg_jaw is None:
        lambda_reg_jaw = lambda_reg
    device = image.device
    model.eval()
    classifier = _get_classifier(device)
    target_id = EMOTION_TO_VIT_ID[target_emotion]

    with torch.no_grad():
        flame_blended = model._blend_style_shape_params(
            flame_params, style_shape_params, style_geometry_strength
        )
        query_points = None
        if model.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_blended = model.renderer.get_query_points(flame_blended, device=device)
        latent_points, image_feats, query_points, color_style = model.forward_latent_points(
            image,
            camera=None,
            query_points=query_points,
            style_image=style_image,
            style_appearance_strength=0.0,  # color has no bearing on emotion; geometry only
            style_geometry_strength=style_geometry_strength,
            style_opacity_strength=style_opacity_strength,
        )
        image_feats_bchw = rearrange(
            image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1]))
        )
        if color_style_override is not None:
            # Same override the final render uses (see infer_single_view) --
            # bypasses the raw network color prediction with the exact
            # per-instance-optimized (gamma, beta) from optimize_style_colors,
            # so this proxy's face matches what actually ships instead of an
            # arbitrary/zeroed color. Fixes a measured regression: without
            # this, a case whose in-loop proxy reported 98.1% confidence
            # scored 0.3% on the real color-styled final render -- see
            # STYLE_GEOMETRY_PIPELINE.md.
            color_style = color_style_override
        gs_model_list, query_points_out, flame_out_base, _ = model.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            flame_data=flame_blended,
            additional_features={"image_feats": image_feats, "image": image, "image_feats_bchw": image_feats_bchw},
            color_style=color_style,
        )

    base_expr = flame_params["expr"].to(device)
    base_jaw = flame_params["jaw_pose"].to(device)
    init_expr_delta = init_expr_delta.detach().to(device)
    init_jaw_delta = init_jaw_delta.detach().to(device)

    expr_delta = init_expr_delta.clone().requires_grad_(True)
    jaw_delta = init_jaw_delta.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([expr_delta, jaw_delta], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        flame_step = dict(flame_out_base)
        flame_step["expr"] = base_expr + expr_delta[:, None, :]
        flame_step["jaw_pose"] = base_jaw + jaw_delta[:, None, :]

        render_res = model.renderer.forward_animate_gs(
            gs_model_list,
            query_points_out,
            model.renderer.get_single_view_smpl_data(flame_step, view_idx),
            render_c2ws[:, view_idx:view_idx + 1],
            render_intrs[:, view_idx:view_idx + 1],
            render_h,
            render_w,
            render_bg_colors[:, view_idx:view_idx + 1],
        )
        comp_rgb = render_res["comp_rgb"][:, 0].clamp(0.0, 1.0)  # [1, 3, H, W]

        clf_input = _classifier_preprocess(comp_rgb)
        logits = classifier(pixel_values=clf_input).logits  # [1, 7]
        log_probs = F.log_softmax(logits, dim=-1)
        target_loss = -log_probs[0, target_id]

        reg_expr = ((expr_delta - init_expr_delta) ** 2).mean()
        reg_jaw = ((jaw_delta - init_jaw_delta) ** 2).mean()
        loss = target_loss + lambda_reg * reg_expr + lambda_reg_jaw * reg_jaw
        loss.backward()
        optimizer.step()

        # Projected gradient descent. Without this the optimiser solves an
        # UNCONSTRAINED problem and _apply_emotion_safety truncates the answer
        # afterwards -- so the shipped delta is not the optimum of the
        # constrained problem, it is the unconstrained optimum with a piece cut
        # off. Measured on a real run: jaw_drift reached 0.101 against a clamp
        # of 0.04, i.e. 60% of the solution was discarded, and the optimiser
        # had spent its whole budget on a lever it was never allowed to pull.
        # Projecting inside the loop closes the jaw shortcut (which the ViT
        # rewards because it reads "mouth open" as emotional intensity) and
        # forces any remaining gain to come through expr instead.
        if project_fn is not None:
            with torch.no_grad():
                pe, pj = project_fn(expr_delta, jaw_delta)
                expr_delta.copy_(pe)
                jaw_delta.copy_(pj)

        if log_fn is not None:
            with torch.no_grad():
                target_prob = log_probs[0, target_id].exp().item()
                expr_drift = (expr_delta - init_expr_delta).norm().item()
                jaw_drift = (jaw_delta - init_jaw_delta).norm().item()
            log_fn(step, float(target_loss.item()), target_prob, expr_drift, jaw_drift)

    return expr_delta.detach(), jaw_delta.detach()
