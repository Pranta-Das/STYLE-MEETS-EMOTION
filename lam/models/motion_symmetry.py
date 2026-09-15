"""Remove the driving clip's own left/right lateral bias from replayed FLAME motion.

Every avatar in this pipeline replays a driving clip's tracked `expr` and
`jaw_pose` verbatim, so any lateral asymmetry baked into that clip is
inherited by every output regardless of identity, style, or emotion class.
That is not hypothetical -- it was traced from a user-visible "lower lip
drifts to one side" artifact:

  * It survives with style fully disabled AND emotion fully disabled, so it
    is not the style pipeline, the EmotionAdapter, or the per-instance
    emotion optimizer.
  * Measured on the FLAME mesh's own 3D landmarks (mouth landmarks 48-67
    against the nose-bridge midline 27-30), decomposed by source:
      content shape alone, zero expr : -0.0006  (identity is NOT the cause)
      neutral shape, driving expr    : +0.0009 mouth / +0.0014 lower lip
      driving jaw yaw alone          : +0.0004 lower lip (about a third)
    i.e. the driving clip's expression carries most of it, its jaw yaw the
    rest, and the LOWER lip is displaced roughly 2x the upper lip -- which
    is exactly the asymmetry that is visible by eye.
  * It is systematic across the shipped driving clips, not one bad file.
    Mean lower-lip offset per clip, and the share of frames pulled the same
    way: Look_In_My_Eyes +0.00230 (95% of frames), GEM +0.00180 (90%),
    Michael_Wayne_Rosen +0.00137 (95%), I_Am_Iron_Man +0.00104 (95%),
    Joe_Biden +0.00075 (80%). `Look_In_My_Eyes` -- the default in
    run_7emotions.sh -- is the worst of the twelve.

The correction works in expression space, not on the rendered image. FLAME's
expression basis is linear in the coefficients, so "the laterally symmetric
part of this expression" is itself a fixed linear map: mirror the vertex
offsets each expression direction produces, average with the original, and
project back onto the basis. That yields one cached 100x100 matrix, applied
per request for the cost of a matmul.

Correspondence is computed geometrically in the CANONICAL template pose
(mirror x, take the nearest template vertex) -- shape-independent and
identical for every request, the same approach `_build_mirror_index` in
lam/stylization/color_optimize.py already uses to symmetrize style colors.

Strength is a knob, not a fixed behavior: at 1.0 the replayed expression is
made fully symmetric, which also removes genuine one-sided expressiveness
(a real smirk), so partial correction is usually the better trade.
"""
import torch

_sym_matrix_cache = {}
_gs_mirror_cache = {}
_mouth_mask_cache = {}


def gaussian_mirror_index(flame_model, device):
    """Per-Gaussian-point left/right mirror partner index, cached.

    Computed on `v_template_up` -- the canonical UPSAMPLED template the
    Gaussians are attached to -- so the correspondence is in index space and
    therefore identity-independent: vertex i's partner is vertex j for every
    subject, and this never needs recomputing per request. Same construction
    as `_build_mirror_index` in lam/stylization/color_optimize.py.
    """
    key = str(device)
    if key in _gs_mirror_cache:
        return _gs_mirror_cache[key]
    v = flame_model.v_template_up.to(device).float()
    mirrored = v.clone()
    mirrored[:, 0] = -mirrored[:, 0]
    n = v.shape[0]
    idx = torch.empty(n, dtype=torch.long, device=device)
    for start in range(0, n, 4000):
        end = min(start + 4000, n)
        idx[start:end] = torch.cdist(mirrored[start:end], v).argmin(dim=1)
    _gs_mirror_cache[key] = idx
    return idx


def mouth_region_mask(flame_model, device, scale=1.4):
    """Boolean mask over the upsampled template selecting the mouth region,
    cached. Located from FLAME's own mouth landmarks (48-67) on the neutral
    template rather than hardcoded indices, so it tracks the actual model."""
    key = (str(device), float(scale))
    if key in _mouth_mask_cache:
        return _mouth_mask_cache[key]
    v = flame_model.v_template_up.to(device).float()
    with torch.no_grad():
        zeros_shape = torch.zeros(1, flame_model.n_shape_params, device=device)
        out = flame_model(
            zeros_shape, torch.zeros(1, 100, device=device),
            torch.zeros(1, 3, device=device), torch.zeros(1, 3, device=device),
            torch.zeros(1, 3, device=device), torch.zeros(1, 6, device=device),
            torch.zeros(1, 3, device=device), return_landmarks=True,
        )
    lmk = out["landmarks"][0]
    centre = lmk[48:68].mean(dim=0)
    radius = (lmk[48:68] - centre[None]).norm(dim=-1).max() * scale
    mask = (v - centre[None]).norm(dim=-1) < radius
    _mouth_mask_cache[key] = mask
    return mask


def symmetrize_gaussian_offsets(offset, query_points, flame_model, strength,
                                mouth_only=True):
    """Damp the lateral asymmetry of the decoder's learned per-point xyz
    offsets, returning (new_offset, new_xyz).

    This is the residual `gs_net` predicts ON TOP of the FLAME surface
    point, not the surface itself -- so symmetrizing it leaves the subject's
    real facial geometry (which lives in `betas`, and measures near-symmetric
    anyway) completely untouched, and only removes asymmetry the decoder
    invented.

    Measured motivation (cluo.jpg, style and emotion both disabled): a
    point's offset and its mirror partner's should mirror each other, and
    the residual of that is 58.4% of offset magnitude over the whole head
    and 37.3% in the mouth region. That is the artifact -- rendered lip
    SHAPE asymmetry sits at ~6.2 against ~1.8 for the source photos, and it
    survives disabling style, emotion, and every FLAME-parameter correction,
    because it is introduced here rather than in expr/jaw/betas.

    `mouth_only` restricts the correction to the mouth region, which is
    where the visible defect is; the rest of the head keeps whatever
    asymmetry the decoder gave it (some of which is legitimately driven by
    lighting and pose in the source photo).
    """
    if not strength:
        return offset, query_points + offset

    idx = gaussian_mirror_index(flame_model, offset.device)
    if idx.shape[0] != offset.shape[0]:
        # Point count doesn't match the template (different subdivision
        # level) -- skip rather than silently mis-pair vertices.
        return offset, query_points + offset

    partner = offset[idx].clone()
    partner[:, 0] *= -1.0          # reflection through the x=0 plane
    symmetric = 0.5 * (offset + partner)

    if mouth_only:
        mask = mouth_region_mask(flame_model, offset.device)[:, None].to(offset.dtype)
    else:
        mask = torch.ones_like(offset[:, :1])

    blend = strength * mask
    new_offset = offset * (1.0 - blend) + symmetric * blend
    return new_offset, query_points + new_offset


def _build_mirror_index(v_template, chunk_size=4000):
    """For each template vertex, the index of its left/right mirror partner."""
    mirrored = v_template.clone()
    mirrored[:, 0] = -mirrored[:, 0]
    n = v_template.shape[0]
    idx = torch.empty(n, dtype=torch.long, device=v_template.device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        idx[start:end] = torch.cdist(mirrored[start:end], v_template).argmin(dim=1)
    return idx


def expr_symmetry_matrix(flame_model, device):
    """Cached [100, 100] matrix M with `expr_symmetric = M @ expr`.

    Built once per device: the mirror correspondence and the expression
    basis are both fixed properties of the FLAME template, independent of
    identity, so this never needs recomputing per request.
    """
    key = str(device)
    if key in _sym_matrix_cache:
        return _sym_matrix_cache[key]

    v = flame_model.v_template.to(device).float()
    idx = _build_mirror_index(v)

    n_shape = flame_model.n_shape_params
    basis = flame_model.shapedirs[:, :, n_shape:].to(device).float()  # [V, 3, 100]

    # Mirroring a vertex offset field: take the partner vertex's offset and
    # flip its x component (a reflection through the x=0 plane).
    basis_mirrored = basis[idx].clone()
    basis_mirrored[:, 0, :] *= -1.0
    basis_symmetric = 0.5 * (basis + basis_mirrored)

    n_expr = basis.shape[-1]
    flat = basis.reshape(-1, n_expr)
    flat_symmetric = basis_symmetric.reshape(-1, n_expr)
    # Least squares: the coefficients that best reproduce the symmetrized
    # offsets within the same basis.
    matrix = torch.linalg.lstsq(flat, flat_symmetric).solution  # [100, 100]

    _sym_matrix_cache[key] = matrix
    return matrix


def symmetrize_motion(flame_params, flame_model, strength):
    """Damp the lateral asymmetry in a driving clip's `expr` / `jaw_pose`.

    `strength` 0.0 leaves the motion untouched; 1.0 makes the replayed
    expression fully left/right symmetric and zeroes the jaw's yaw and roll.
    Jaw pitch (mouth opening) is never touched -- it is the symmetric axis
    and carries the actual articulation.

    Edits only `expr` and `jaw_pose` and returns a new dict; the driving
    clip's pose, camera and every other key pass through untouched.
    """
    if not strength:
        return flame_params

    out = dict(flame_params)

    expr = out.get("expr", None)
    if expr is not None:
        matrix = expr_symmetry_matrix(flame_model, expr.device).to(expr.dtype)
        expr_symmetric = expr @ matrix.transpose(0, 1)
        out["expr"] = expr * (1.0 - strength) + expr_symmetric * strength

    jaw = out.get("jaw_pose", None)
    if jaw is not None:
        jaw = jaw.clone()
        # index 0 is pitch (jaw drop, symmetric); 1 is yaw and 2 is roll,
        # both purely lateral and the measured jaw-side contribution.
        jaw[..., 1] *= (1.0 - strength)
        jaw[..., 2] *= (1.0 - strength)
        out["jaw_pose"] = jaw

    return out
