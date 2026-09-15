"""Per-instance (test-time) Gaussian color optimization for style transfer.

style_adapter.appearance_mlp (see modeling_lam.py) tries to learn one small
network that generalizes a color transform across many (content, style) pairs
from limited training data -- in practice this converges slowly and can settle
on a generic bias unrelated to any specific style image (see
STYLE_GEOMETRY_PIPELINE.md). For the actual use case -- one content image, one
style image, one output -- there's a more direct and reliable option: skip
generalization entirely and directly optimize this reconstruction's per-point
Gaussian color against this exact style image's real Gram/AdaIN statistics,
classic Gatys-style optimization (https://arxiv.org/abs/1508.06576) applied to
3D Gaussian color instead of 2D pixels. This is slower per-request (a few
hundred optimization steps instead of one network forward pass) but guarantees
convergence toward the actual target instead of hoping a network generalized.

The encoder/transformer backbone and geometry (FLAME shape blend +
geometry_mlp) stay exactly as they already work today -- computed once,
frozen -- only the per-point color affine (gamma, beta) is optimized here.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lam.runners.train.lam import StyleTransferLoss


# FLAME's own per-vertex part labels (face/scalp/lips/nose/ears/eyes/neck/...),
# loaded from FLAME_masks.pkl inside flame_model.mask.v.<region>. Listed in
# priority order because a few vertices belong to more than one named group
# (e.g. "face" and "forehead" overlap a little) -- first match wins.
_REGION_PRIORITY = [
    'left_eyeball', 'right_eyeball', 'left_eye_region', 'right_eye_region', 'eye_region',
    'lips', 'nose', 'left_ear', 'right_ear', 'scalp', 'forehead', 'face', 'neck', 'boundary',
]


@torch.no_grad()
def _build_region_labels(flame_model, device, chunk_size=4000):
    """One integer region id per query point (0..len(_REGION_PRIORITY)-1),
    used to keep the color-smoothness regularizer from blurring across
    semantic boundaries (see the comment on `region_labels` in
    `_build_knn_graph`).

    FLAME_masks.pkl only labels the original ~5k-vertex template, but query
    points live on the subdivided ~20k-vertex mesh (see flame.py's
    FlameHeadSubdivided). Both are static, shape-independent templates (same
    topology regardless of the actual face's betas), so labels are propagated
    to the subdivided vertices once, by nearest neighbor in the CANONICAL
    template pose -- this has nothing to do with any particular face and can
    be reused for every request.
    """
    v_template = flame_model.v_template.to(device)        # [5023, 3]
    v_template_up = flame_model.v_template_up.to(device)  # [20018, 3]

    n_orig = v_template.shape[0]
    label = torch.full((n_orig,), -1, dtype=torch.long, device=device)
    for region_id, region in enumerate(_REGION_PRIORITY):
        idx = getattr(flame_model.mask.v, region).to(device)
        idx = idx[label[idx] == -1]
        label[idx] = region_id

    labeled_idx = (label >= 0).nonzero(as_tuple=True)[0]
    labeled_pos = v_template[labeled_idx]
    labeled_lbl = label[labeled_idx]

    n_up = v_template_up.shape[0]
    up_label = torch.empty(n_up, dtype=torch.long, device=device)
    for start in range(0, n_up, chunk_size):
        end = min(start + chunk_size, n_up)
        d = torch.cdist(v_template_up[start:end], labeled_pos)  # [chunk, n_labeled]
        up_label[start:end] = labeled_lbl[d.argmin(dim=1)]
    return up_label


@torch.no_grad()
def _build_mirror_index(flame_model, device, chunk_size=4000):
    """For each query point, the index of its left/right mirror partner.

    The style target below is "the style image's own predicted color at this
    same FLAME vertex", produced by reconstructing the style image from a
    single frontal view. Single-view reconstruction is markedly less reliable
    on whichever side is worse lit or more occluded, and that unreliability
    shows up as left/right color disagreement between vertices that are the
    same anatomical part. Measured on
    figure/004_445_841_4k_guzz-soares-fefi-l.jpg: mean target RGB is
    (69.9, 55.8, 53.0) for left_ear against (147.4, 113.7, 90.7) for right
    -- the same ear, twice as dark on one side -- and left/right eyeball
    disagree by a similar factor. Copied onto the content that reads as a
    face lit from one side by nothing in particular.

    Averaging each target with its mirror partner's removes that, because a
    portrait's two sides ARE the same material. Correspondence is computed
    geometrically in the CANONICAL template pose (mirror x, take the nearest
    template vertex), which is shape-independent and identical for every
    request, exactly like _build_region_labels.
    """
    v = flame_model.v_template_up.to(device)  # [N, 3], canonical pose
    mirrored = v.clone()
    mirrored[:, 0] = -mirrored[:, 0]
    n = v.shape[0]
    idx = torch.empty(n, dtype=torch.long, device=device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        idx[start:end] = torch.cdist(mirrored[start:end], v).argmin(dim=1)
    return idx


@torch.no_grad()
def _build_knn_graph(query_points, region_labels=None, k=8, chunk_size=2000):
    """K-nearest-neighbor indices in 3D space, one row of neighbor indices per
    query point. Used to penalize a point's color for disagreeing with its
    actual spatial neighbors (see the smoothness comment below) -- computed
    once against the fixed (frozen) query point positions and reused for
    every optimization step, chunked so an N x N distance matrix (~20k points
    here) isn't ever fully materialized at once.

    When `region_labels` is given, neighbors are restricted to points sharing
    the same FLAME part label -- plain 3D nearest-neighbor doesn't know a
    scalp point and the face-skin point 2mm away are anatomically different
    things, so unrestricted smoothing was pulling hairline colors halfway
    toward skin (and vice versa) at exactly the boundary where users notice
    it most. Restricting to same-part neighbors keeps the silhouette-fringe
    smoothing this was built for while no longer bleeding across regions.
    """
    points = query_points[0]  # [N, 3]
    n = points.shape[0]
    neighbor_idx = torch.empty(n, k, dtype=torch.long, device=points.device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = torch.cdist(points[start:end], points)  # [chunk, N]
        dists[torch.arange(end - start, device=points.device), torch.arange(start, end, device=points.device)] = float('inf')
        if region_labels is not None:
            same_region = region_labels[start:end, None] == region_labels[None, :]  # [chunk, N]
            dists = dists.masked_fill(~same_region, float('inf'))
        neighbor_idx[start:end] = dists.topk(k, largest=False).indices
    return neighbor_idx


@torch.no_grad()
def _encode_own_point_colors(model, image, shape_params):
    """Reconstruct `image` through the model's normal (non-styled) path and
    return its own per-point base color, in the exact same point ordering as
    every other reconstruction -- query points are indexed by canonical FLAME
    vertex identity (see PointEmbed / e2e_flame query points), so point i is
    "the same anatomical vertex" across any two calls of this function
    regardless of whose betas were used. That gives an exact point-for-point
    correspondence between a content reconstruction and a style
    reconstruction for free, without any 2D matching at all: point i's target
    color is simply the style image's OWN predicted color at point i.
    """
    device = image.device
    query_points, flame_data = model.renderer.get_query_points({"betas": shape_params}, device=device)
    latent_points, _, query_points, _ = model.forward_latent_points(
        image, camera=None, query_points=query_points, style_image=None,
    )
    gs_model_list, _, _, _ = model.renderer.forward_gs(
        gs_hidden_features=latent_points,
        query_points=query_points,
        flame_data=flame_data,
        color_style=None,
    )
    return gs_model_list[0].shs[:, 0, :]  # [N, 3], base RGB in [0, 1] (gs_use_rgb=True)


def optimize_style_colors(
    model,
    content_image,
    style_image,
    flame_params,
    render_c2ws,
    render_intrs,
    render_bg_colors,
    view_indices,
    render_h,
    render_w,
    style_shape_params=None,
    style_geometry_strength=0.85,
    style_opacity_strength=0.0,
    num_steps=400,
    lr=0.03,
    max_color_scale=0.9,
    max_darken_ratio=0.5,
    lambda_content=0.3,
    lambda_style=2.0e3,
    lambda_pixel=0.02,
    # Was 20.0 back when this whole-image feature mean/std match was the only
    # thing giving any region a push toward the style's actual palette.
    # AdaIN has no notion of "this pixel is skin, that one is hair" -- it just
    # wants the RENDER's overall color statistics to match the STYLE image's
    # overall statistics, so on a style photo that's mostly warm/golden hair
    # by pixel area, it pulls skin toward that same warmth too. Now that
    # lambda_anatomy gives a real per-point target, AdaIN's job shrinks to
    # "nudge overall contrast/vibrance to feel right", not "decide regional
    # color" -- lowered so it stops fighting the anatomy target on skin tone.
    lambda_adain=5.0,
    lambda_tv=1.0e-4,
    lambda_color_reg=0.3,
    # Was 30.0 back when the only color signal was whole-image AdaIN/Gram
    # statistics, which gave no reason for any point to differ from any other
    # and let sparse multi-view gradients crush isolated points to black --
    # uniformity fought that by discouraging spread. Now that lambda_anatomy
    # gives a real per-point target (hair should legitimately end up a very
    # different color from skin), a high uniformity weight actively fights
    # the thing we now want; kept small, just enough to damp render-noise
    # outliers the anatomy target doesn't fully pin down.
    lambda_color_uniformity=3.0,
    lambda_anatomy=60.0,
    anatomy_symmetrize=1.0,
    anatomy_skip_regions=('boundary',),
    lambda_color_smoothness=150.0,
    smoothness_k=8,
    log_every=50,
    log_fn=None,
):
    device = content_image.device
    model.eval()

    with torch.no_grad():
        flame_blended = model._blend_style_shape_params(flame_params, style_shape_params, style_geometry_strength)
        query_points = None
        if model.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_blended = model.renderer.get_query_points(flame_blended, device=device)
        # style_appearance_strength=0.0 -> style_adapter's own color prediction is
        # zeroed out; geometry (query_points offset) still comes from its trained
        # geometry_mlp / the algebraic FLAME blend, exactly as normal inference.
        latent_points, image_feats, query_points, _ = model.forward_latent_points(
            content_image,
            camera=None,
            query_points=query_points,
            style_image=style_image,
            style_appearance_strength=0.0,
            style_geometry_strength=style_geometry_strength,
            style_opacity_strength=style_opacity_strength,
        )
        image_feats_bchw = rearrange(image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1])))

    n_points = query_points.shape[1]
    color_gamma = nn.Parameter(torch.zeros(1, n_points, 3, device=device, dtype=latent_points.dtype))
    color_beta = nn.Parameter(torch.zeros(1, n_points, 3, device=device, dtype=latent_points.dtype))

    with torch.no_grad():
        # The network's own prediction for content's per-point color, at
        # these exact (possibly geometry-blended) query points -- the base
        # that color_gamma/color_beta perturb. Reused as-is from the
        # forward_gs call the render loop below makes every step (same
        # inputs -> same output), just computed once up front for the
        # anatomy-correspondence target below.
        content_base_color = model.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            flame_data=flame_blended,
            color_style=None,
        )[0][0].shs[:, 0, :].to(latent_points.dtype)  # [N, 3]

        # Reconstruct the STYLE image through the model's own (unstyled) path
        # to get its own per-point color prediction, in the exact same
        # canonical point ordering as content -- point i means "the same
        # anatomical FLAME vertex" in both, so style_point_colors[i] is
        # directly the right target color for content point i's hair/skin/lip
        # region, no 2D matching needed at all. This replaces an earlier
        # coarse 16x16-grid + VGG-texture-similarity matcher that had no real
        # notion of "hair" vs "skin" vs "lips" and would confidently match the
        # wrong region when a style image's composition/proportions differed
        # from the content's (see STYLE_GEOMETRY_PIPELINE.md).
        style_shape_for_points = style_shape_params if style_shape_params is not None else flame_params["betas"]
        style_point_colors = _encode_own_point_colors(
            model, style_image, style_shape_for_points.to(device=device, dtype=flame_params["betas"].dtype)
        ).to(latent_points.dtype)

        # FLAME's own per-vertex part labels (scalp/face/lips/...), propagated
        # to these query points -- restricts the smoothness regularizer below
        # to same-part neighbors only (see _build_knn_graph).
        region_labels = _build_region_labels(model.renderer.flame_model, device)

        # Clean up the style target before anything optimizes against it.
        # Two defects are structural rather than style-specific, so both are
        # corrected here rather than left for the caller to tune around.
        if anatomy_symmetrize > 0:
            mirror_idx = _build_mirror_index(model.renderer.flame_model, device)
            w = float(min(1.0, anatomy_symmetrize)) * 0.5
            style_point_colors = (
                style_point_colors * (1.0 - w) + style_point_colors[mirror_idx] * w
            )

        # 'boundary' is the mesh's own cut edge (the rim where the head model
        # stops), not an anatomical part with a color of its own. Its points
        # sit exactly on the style reconstruction's silhouette, so they pick
        # up whatever was behind the style photo's head rather than any part
        # of the subject -- on the guzz-soares reference that is a muted green
        # backdrop, and the region's mean target comes out (102, 156, 171),
        # a cyan nothing-colour that then gets painted onto the content's
        # crown and neck as visible teal patches. Dropping these points from
        # the anatomy term leaves them to the smoothness regularizer, which
        # carries them from their real in-region neighbors instead.
        anatomy_point_mask = torch.ones_like(style_point_colors[:, :1])
        for region in anatomy_skip_regions:
            if region in _REGION_PRIORITY:
                anatomy_point_mask[region_labels == _REGION_PRIORITY.index(region)] = 0.0

    # Fringing/rainbow noise concentrated at silhouette edges (hair-background
    # boundary) rather than spread across flat regions doesn't come from 2D
    # pixel noise -- image-space TV loss barely touches it (tested: 10000x its
    # default weight, no measurable change). Boundary Gaussians have blend
    # weight that shifts a lot with viewing angle (parallax moves the
    # silhouette), so they see the least consistent gradient of any points
    # across the sampled views. Penalizing each point's color for disagreeing
    # with its actual 3D neighbors -- not just its own gradient history --
    # directly targets that, independent of any single view's quirks.
    knn_idx = _build_knn_graph(query_points, region_labels=region_labels, k=smoothness_k)

    criterion = StyleTransferLoss(
        content_layers=('relu3_3',),
        style_layers=('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'),
        lambda_content=lambda_content,
        lambda_style=lambda_style,
        lambda_pixel=lambda_pixel,
        lambda_adain=lambda_adain,
        lambda_tv=lambda_tv,
    ).to(device)

    optimizer = torch.optim.Adam([color_gamma, color_beta], lr=lr)
    additional_features = {"image_feats": image_feats, "image": content_image, "image_feats_bchw": image_feats_bchw}
    num_views = render_c2ws.shape[1]

    # tanh(x)*max_color_scale is symmetric, but color*(1+gamma) means the
    # darkening direction (gamma -> negative) can crush a point to near-black
    # with a small push, while brightening (gamma -> positive) has no such
    # cheap extreme -- matching a higher-contrast style then preferentially
    # exploits whichever points are already in shadow, crushing them toward
    # black rather than shifting everything evenly (confirmed: the unstyled
    # render already has natural lighting asymmetry, e.g. one cheek darker
    # from the source photo, and styling amplified that gap ~3x). Capping the
    # darkening side to a fraction of the brightening side directly prevents
    # the collapse without limiting the (structurally safe) brightening range.
    darken_scale = max_color_scale * max_darken_ratio

    content_for_loss_cache = {}

    def _content_for_loss(shape_hw):
        if shape_hw not in content_for_loss_cache:
            if shape_hw == content_image.shape[-2:]:
                content_for_loss_cache[shape_hw] = content_image
            else:
                content_for_loss_cache[shape_hw] = F.interpolate(
                    content_image, size=shape_hw, mode='bilinear', align_corners=False
                )
        return content_for_loss_cache[shape_hw]

    for step in range(num_steps):
        optimizer.zero_grad()
        raw_gamma = torch.tanh(color_gamma)
        raw_beta = torch.tanh(color_beta)
        gamma = torch.where(raw_gamma >= 0, raw_gamma * max_color_scale, raw_gamma * darken_scale)
        beta = torch.where(raw_beta >= 0, raw_beta * max_color_scale, raw_beta * darken_scale)

        gs_model_list, query_points_out, flame_out, _ = model.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            flame_data=flame_blended,
            additional_features=additional_features,
            color_style=(gamma, beta),
        )
        # Randomly picking ONE view per step gives points that are only
        # well-angled/visible in a subset of views sparse, bursty gradients.
        # Adam's per-parameter step size is normalized by a running average of
        # squared gradients, which shrinks during a point's zero-gradient
        # steps -- so the next time that point does get a gradient, the
        # effective step is disproportionately large. That produces exactly
        # what was observed: a whole coherent region (points sharing similar
        # visibility across the sampled views) crushed toward black together,
        # not smooth per-point noise. Averaging the loss over ALL sampled
        # views every step gives every point a consistent, non-sparse signal.
        step_loss = 0.0
        step_loss_dict = None
        for v in range(num_views):
            render_res = model.renderer.forward_animate_gs(
                gs_model_list,
                query_points_out,
                model.renderer.get_single_view_smpl_data(flame_out, view_indices[v]),
                render_c2ws[:, v:v + 1],
                render_intrs[:, v:v + 1],
                render_h,
                render_w,
                render_bg_colors[:, v:v + 1],
            )
            comp_rgb = render_res['comp_rgb'][:, 0].clamp(0.0, 1.0)  # [1, 3, H, W]
            content_for_loss = _content_for_loss(comp_rgb.shape[-2:])
            view_loss, view_loss_dict = criterion(comp_rgb, content_for_loss, style_image)
            step_loss = step_loss + view_loss
            if step_loss_dict is None:
                step_loss_dict = {k: 0.0 for k in view_loss_dict}
            for k, val in view_loss_dict.items():
                step_loss_dict[k] += val / num_views
        step_loss = step_loss / num_views

        # Direct per-point anatomical correspondence: point i's optimized
        # color should match style_point_colors[i], the style image's own
        # predicted color at that same FLAME vertex. Computed purely in 3D
        # (no rendering involved), so it's exact and gives every point --
        # including ones poorly covered by the sampled render views -- a
        # clean, region-correct gradient every step. This is what actually
        # keeps hair color out of skin and skin color out of hair/cloth.
        predicted_color = content_base_color * (1.0 + gamma[0]) + beta[0]
        anatomy_loss = (
            anatomy_point_mask * (predicted_color - style_point_colors).pow(2)
        ).sum() / anatomy_point_mask.sum().clamp(min=1.0).mul(style_point_colors.shape[-1])
        step_loss_dict['anatomy'] = float(anatomy_loss.item())

        # AdaIN/Gram losses only look at whole-image statistics, so a region
        # that's already dark (natural shadow, low-light side) can be driven
        # to near-black instead of the whole face shifting evenly -- darkening
        # an already-dark point to ~0 costs little (small absolute change),
        # while shifting an already-bright point by the same amount costs
        # more, so the optimizer preferentially sacrifices whichever points
        # are cheapest rather than moving everything evenly. Two regularizers
        # counter that: a magnitude penalty (don't move any point more than
        # necessary) and, more directly, a uniformity penalty on the *spread*
        # of gamma/beta across points (don't let some points go extreme while
        # others stay untouched -- push toward a more even shift overall).
        color_reg = (gamma.pow(2).mean() + beta.pow(2).mean())
        color_uniformity = (gamma.var() + beta.var())
        neighbor_gamma = gamma[0][knn_idx].mean(dim=1)  # [N, 3], each point's own neighbors' average
        neighbor_beta = beta[0][knn_idx].mean(dim=1)
        color_smoothness = F.mse_loss(gamma[0], neighbor_gamma) + F.mse_loss(beta[0], neighbor_beta)
        loss = (
            step_loss
            + lambda_anatomy * anatomy_loss
            + lambda_color_reg * color_reg
            + lambda_color_uniformity * color_uniformity
            + lambda_color_smoothness * color_smoothness
        )
        step_loss_dict['color_reg'] = float(color_reg.item())
        step_loss_dict['color_uniformity'] = float(color_uniformity.item())
        step_loss_dict['color_smoothness'] = float(color_smoothness.item())
        loss.backward()
        optimizer.step()

        if log_fn is not None and (step % log_every == 0 or step == num_steps - 1):
            log_fn(step, step_loss_dict)

    with torch.no_grad():
        raw_gamma = torch.tanh(color_gamma)
        raw_beta = torch.tanh(color_beta)
        final_gamma = torch.where(raw_gamma >= 0, raw_gamma * max_color_scale, raw_gamma * darken_scale).detach()
        final_beta = torch.where(raw_beta >= 0, raw_beta * max_color_scale, raw_beta * darken_scale).detach()

    return final_gamma, final_beta
