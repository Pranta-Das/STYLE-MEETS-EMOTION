"""Train EmotionAdapter: pure FLAME-parameter-space regression, no rendering.

Each tracked reference photo (tools/track_emotion_refs.py's manifest) gives
one (emotion_label, expr_target, jaw_target) example -- the network's job is
to predict, given the emotion label and *some* current base expr context,
the delta that moves toward that target. Reference photos only ever provide
one static context each (whatever expression state the photo happened to
capture), so training pairs each real target with several *synthetic*
context vectors sampled from real per-frame expr trajectories already on
disk in assets/sample_motion/export/*/flame_param/*.npz -- same target
regardless of which context was sampled, teaching "move toward this emotion
from any starting point" instead of memorizing one context.

No differentiable rendering, no VGG, no GPU strictly required -- this is a
few-thousand-step regression over ~100-dim vectors and should take well
under a minute.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from lam.models.emotion_adapter import EMOTION_CLASSES, EmotionAdapter


def _load_expr_jaw(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["expr"].reshape(-1).astype(np.float32), d["jaw_pose"].reshape(-1).astype(np.float32)


def load_manifest_targets(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    targets = {cls: [] for cls in EMOTION_CLASSES}
    for item in manifest:
        expr, jaw = _load_expr_jaw(item["flame_param_path"])
        targets[item["emotion_label"]].append((expr, jaw))
    return targets


def load_context_pool(motion_glob="assets/sample_motion/export/*/flame_param/*.npz", max_files=3000):
    paths = [p for p in glob.glob(motion_glob) if "._" not in os.path.basename(p)]
    paths = sorted(paths)[:max_files]
    contexts = []
    for p in paths:
        try:
            expr, _ = _load_expr_jaw(p)
            contexts.append(expr)
        except Exception:
            continue
    return np.stack(contexts, axis=0) if contexts else np.zeros((1, 100), dtype=np.float32)


def trimmed_inlier_indices(class_expr_deltas, keep_fraction=0.5, iters=5, min_keep=8):
    """Indices of the most directionally prototypical references in a class.

    Training regresses every reference with MSE, so it converges to the
    per-class *mean* delta -- which means a class whose references disagree
    about direction gets its signal cancelled rather than averaged. Measured
    on the tracked AffectNet refs, individual 'sad' deltas are as large as
    every other class's (mean per-ref norm 6.80 vs 6.2-7.3), but their mean
    cosine to the class mean is only 0.23 (lowest of all seven classes), so
    the mean collapses to norm 1.54 while happy/anger/surprise land at
    2.7-3.0. That ~55% amplitude is the actual reason 'sad' reads faintly in
    generated video -- NOT the small negative jaw-x delta, which is only
    -0.0013 rad in the data and is anatomically right (sadness doesn't open
    the jaw).

    So instead of averaging everything, iteratively keep the top
    `keep_fraction` of references by cosine to the running mean and
    recompute. This recovers a prototypical direction rather than merely
    amplifying a cancelled one (which a flat per-class gain would do -- it
    would scale the disagreement up too). Effect on the tracked refs:
    sad 1.54 -> 2.85, and all six emotion classes land in one 2.85-4.31
    band instead of spanning 1.54-2.96.

    Never call this for 'neutral': neutral IS the baseline the deltas are
    measured against, so its mean must stay pinned at exactly zero.
    Trimming it toward its own most-coherent subset would manufacture a
    spurious nonzero neutral offset (measured: norm 2.08).
    """
    deltas = np.asarray(class_expr_deltas, dtype=np.float32)
    n = deltas.shape[0]
    keep = max(min(min_keep, n), int(round(n * keep_fraction)))
    if keep >= n:
        return np.arange(n)
    mu = deltas.mean(axis=0)
    idx = np.arange(n)
    for _ in range(iters):
        cos = (deltas @ mu) / (np.linalg.norm(deltas, axis=1) * np.linalg.norm(mu) + 1e-8)
        idx = np.argsort(-cos)[:keep]
        mu = deltas[idx].mean(axis=0)
    return np.sort(idx)


def _cos_to_vec(deltas, vec):
    return (deltas @ vec) / (np.linalg.norm(deltas, axis=1) * np.linalg.norm(vec) + 1e-8)


def discriminative_trim_indices(expr_deltas_by_class, keep_fraction=0.5, iters=5, min_keep=8,
                                 exempt=("neutral",)):
    """Per-class reference selection that optimizes for "distinctly THIS
    class", not just "internally self-consistent" -- see
    `trimmed_inlier_indices` for the self-consistency version this replaces,
    and why it isn't enough on its own.

    Measured on the hybrid-filtered AffectNet refs: self-consistency
    trimming alone gave every class a healthy delta *magnitude* (norm 3.7-5.3
    across all seven), yet anger and disgust converged to cosine similarity
    +0.77 -- nearly the same direction -- and rendered anger/disgust/sad were
    all independently misclassified as "fear" by an outside judge. A pure
    per-class self-consistency trim can't see this: it only ever asks "do
    this class's own references agree with each other", never "does this
    class's direction actually differ from its neighbors". A class whose
    references are internally coherent can still coherently converge on
    something another class also converges on.

    So this scores each candidate reference by MARGIN: how much more its
    delta resembles its own class's running mean than it resembles the
    single closest *other* class's running mean
    (cos(delta, own_mean) - max_other cos(delta, other_mean)), keeps the top
    `keep_fraction` by that margin, recomputes means from the kept subset,
    and repeats. A reference that happens to sit near the anger/disgust
    overlap gets a small (or negative) margin and is the first to be dropped,
    even if it was one of the internally "most typical" anger photos --
    typical-but-ambiguous is exactly what a purely self-consistency trim
    can't distinguish from typical-and-distinct.

    `exempt` classes (neutral) keep all their references and are still used
    as a repulsion target for everyone else's margin score, but are never
    themselves trimmed -- neutral's mean must stay pinned near zero (it's
    the baseline every delta is measured against), not pulled toward
    whichever of its own references look least like other emotions.
    """
    classes = list(expr_deltas_by_class.keys())
    trim_classes = [c for c in classes if c not in exempt]
    means = {c: expr_deltas_by_class[c].mean(axis=0) for c in classes}
    kept_idx = {c: np.arange(len(expr_deltas_by_class[c])) for c in classes}

    for _ in range(iters):
        for c in trim_classes:
            deltas = expr_deltas_by_class[c]
            n = deltas.shape[0]
            keep = max(min(min_keep, n), int(round(n * keep_fraction)))
            if keep >= n:
                kept_idx[c] = np.arange(n)
                continue
            own_cos = _cos_to_vec(deltas, means[c])
            other_cos = np.stack(
                [_cos_to_vec(deltas, means[o]) for o in classes if o != c], axis=1
            )
            margin = own_cos - other_cos.max(axis=1)
            kept_idx[c] = np.sort(np.argsort(-margin)[:keep])
        for c in trim_classes:
            means[c] = expr_deltas_by_class[c][kept_idx[c]].mean(axis=0)
    return kept_idx


def train(cfg):
    manifest_path = cfg.get("manifest_path", "exps/train_lam/emotion_refs/tracked_emotion_refs.json")
    output_dir = cfg.get("output_dir", "exps/train_lam/emotion_adapter")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    targets_by_class = load_manifest_targets(manifest_path)
    for cls in EMOTION_CLASSES:
        n = len(targets_by_class[cls])
        print(f"[data] {cls}: {n} tracked references")
        if n == 0:
            raise ValueError(f"No tracked references found for class '{cls}' -- check {manifest_path}")

    if not targets_by_class["neutral"]:
        raise ValueError("Need at least one 'neutral' reference to use as the baseline for deltas.")
    neutral_expr = np.mean([e for e, _ in targets_by_class["neutral"]], axis=0)
    neutral_jaw = np.mean([j for _, j in targets_by_class["neutral"]], axis=0)

    context_pool = load_context_pool(cfg.get("context_glob", "assets/sample_motion/export/*/flame_param/*.npz"))
    print(f"[data] context pool: {context_pool.shape[0]} real per-frame expr vectors")

    # Flatten to (class_id, expr_delta_target, jaw_delta_target) rows,
    # dropping each emotion class's least-prototypical (trim_mode="self") or
    # least class-distinctive (trim_mode="discriminative", the default --
    # see discriminative_trim_indices for why self-consistency alone isn't
    # enough) references first. Neutral is always exempt from being trimmed
    # (see either trim function's docstring for why), but still participates
    # as a repulsion target in discriminative mode.
    trim_keep_fraction = float(cfg.get("trim_keep_fraction", 0.5))
    trim_iters = int(cfg.get("trim_iters", 5))
    trim_mode = cfg.get("trim_mode", "discriminative")

    expr_deltas_by_class = {
        cls: np.stack([e - neutral_expr for e, _ in targets_by_class[cls]], axis=0)
        for cls in EMOTION_CLASSES
    }
    jaw_deltas_by_class = {
        cls: np.stack([j - neutral_jaw for _, j in targets_by_class[cls]], axis=0)
        for cls in EMOTION_CLASSES
    }

    # Optionally make the TARGETS laterally symmetric before any trimming or
    # regression, so the learned delta is symmetric by construction rather
    # than needing an inference-time correction (which lands after the
    # per-instance optimizer and interacts with it unpredictably -- measured:
    # symmetrizing at inference fixed 'surprise' but made 'sad' overshoot and
    # flip sign).
    #
    # Why this is needed at all: each reference is one photo of one person,
    # carrying that person's own facial asymmetry plus tracking/pose noise.
    # Averaging ~50-100 of them per class does not cancel it -- measured on
    # the shipped checkpoints, the asymmetric residual is 35-72% of every
    # class's delta norm (both the hybrid and closed-mouth checkpoints), and
    # discriminative trimming can actively preserve a coherent asymmetric
    # bias when the kept subset happens to share one. A prototypical
    # anger/happy/surprise is anatomically symmetric, so that residual is
    # noise being learned as signal, and it is what makes the rendered mouth
    # pull sideways by a different amount in a different direction for every
    # emotion class.
    symmetrize_targets = float(cfg.get("symmetrize_targets", 0.0))
    if symmetrize_targets:
        import torch as _torch
        from lam.models.motion_symmetry import expr_symmetry_matrix
        from lam.models.rendering.flame_model.flame import FlameHeadSubdivided
        human_model_path = cfg.get("human_model_path", "./model_zoo/human_parametric_models")
        _flame = FlameHeadSubdivided(
            300, 100, add_teeth=False, add_shoulder=False,
            flame_model_path=f"{human_model_path}/flame_assets/flame/flame2023.pkl",
            flame_lmk_embedding_path=f"{human_model_path}/flame_assets/flame/landmark_embedding_with_eyes.npy",
            flame_template_mesh_path=f"{human_model_path}/flame_assets/flame/head_template_mesh.obj",
            flame_parts_path=f"{human_model_path}/flame_assets/flame/FLAME_masks.pkl",
            subdivide_num=0,
        )
        sym = expr_symmetry_matrix(_flame, _torch.device("cpu")).numpy()
        for cls in EMOTION_CLASSES:
            d = expr_deltas_by_class[cls]
            d_sym = d @ sym.T
            expr_deltas_by_class[cls] = (
                d * (1.0 - symmetrize_targets) + d_sym * symmetrize_targets
            ).astype(np.float32)
            # jaw yaw/roll are purely lateral; pitch (index 0) is the
            # symmetric open/close axis and must survive untouched.
            j = jaw_deltas_by_class[cls].copy()
            j[:, 1] *= (1.0 - symmetrize_targets)
            j[:, 2] *= (1.0 - symmetrize_targets)
            jaw_deltas_by_class[cls] = j.astype(np.float32)
        print(f"[symmetrize] applied lateral symmetrization to targets at strength {symmetrize_targets}")

    if trim_keep_fraction >= 1.0:
        kept_idx_by_class = {cls: np.arange(expr_deltas_by_class[cls].shape[0]) for cls in EMOTION_CLASSES}
    elif trim_mode == "discriminative":
        kept_idx_by_class = discriminative_trim_indices(
            expr_deltas_by_class, keep_fraction=trim_keep_fraction, iters=trim_iters, exempt=("neutral",)
        )
    else:
        kept_idx_by_class = {
            cls: (np.arange(expr_deltas_by_class[cls].shape[0]) if cls == "neutral" else
                  trimmed_inlier_indices(expr_deltas_by_class[cls], keep_fraction=trim_keep_fraction, iters=trim_iters))
            for cls in EMOTION_CLASSES
        }

    rows_class, rows_expr_delta, rows_jaw_delta = [], [], []
    kept_per_class = {}
    for class_id, cls in enumerate(EMOTION_CLASSES):
        expr_deltas = expr_deltas_by_class[cls]
        jaw_deltas = jaw_deltas_by_class[cls]
        keep_idx = kept_idx_by_class[cls]
        kept_per_class[cls] = int(len(keep_idx))
        before = float(np.linalg.norm(expr_deltas.mean(axis=0)))
        after = float(np.linalg.norm(expr_deltas[keep_idx].mean(axis=0)))
        print(f"[trim/{trim_mode}] {cls}: kept {len(keep_idx)}/{expr_deltas.shape[0]} refs, "
              f"||mean expr-delta|| {before:.3f} -> {after:.3f}")
        for i in keep_idx:
            rows_class.append(class_id)
            rows_expr_delta.append(expr_deltas[i])
            rows_jaw_delta.append(jaw_deltas[i])
    rows_class = np.array(rows_class, dtype=np.int64)
    rows_expr_delta = np.stack(rows_expr_delta, axis=0).astype(np.float32)
    rows_jaw_delta = np.stack(rows_jaw_delta, axis=0).astype(np.float32)
    n_rows = len(rows_class)

    rows_class_t = torch.from_numpy(rows_class).to(device)
    rows_expr_delta_t = torch.from_numpy(rows_expr_delta).to(device)
    rows_jaw_delta_t = torch.from_numpy(rows_jaw_delta).to(device)
    context_pool_t = torch.from_numpy(context_pool.astype(np.float32)).to(device)

    max_expr_delta = float(cfg.get("max_expr_delta", 3.0))
    max_jaw_delta = float(cfg.get("max_jaw_delta", 0.15))
    model = EmotionAdapter(
        num_emotions=len(EMOTION_CLASSES),
        embed_dim=int(cfg.get("embed_dim", 32)),
        max_expr_delta=max_expr_delta,
        max_jaw_delta=max_jaw_delta,
    ).to(device)

    lr = float(cfg.get("lr", 1e-3))
    num_steps = int(cfg.get("num_steps", 3000))
    batch_size = int(cfg.get("batch_size", 64))
    log_every = int(cfg.get("log_every", 200))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for step in range(num_steps):
        row_idx = torch.randint(0, n_rows, (batch_size,), device=device)
        ctx_idx = torch.randint(0, context_pool_t.shape[0], (batch_size,), device=device)

        emotion_id = rows_class_t[row_idx]
        base_expr = context_pool_t[ctx_idx]
        target_expr_delta = rows_expr_delta_t[row_idx]
        target_jaw_delta = rows_jaw_delta_t[row_idx]

        pred_expr_delta, pred_jaw_delta = model(emotion_id, base_expr)
        loss_expr = F.mse_loss(pred_expr_delta, target_expr_delta)
        loss_jaw = F.mse_loss(pred_jaw_delta, target_jaw_delta)
        loss = loss_expr + loss_jaw

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == num_steps - 1:
            print(f"[train] step={step} loss_expr={loss_expr.item():.4f} loss_jaw={loss_jaw.item():.6f} total={loss.item():.4f}")
            loss_history.append({"step": step, "loss_expr": loss_expr.item(), "loss_jaw": loss_jaw.item()})

    # Verification: per-class predicted delta should differ meaningfully
    # between classes -- if they collapse to the same vector, the model
    # isn't actually differentiating emotions.
    #
    # Evaluate at the MEAN of the real context pool, not at base_expr=0.
    # The adapter conditions on base_expr, and a zero vector is not a
    # context it ever sees at inference (real per-frame expr vectors have
    # norm ~10, and even their mean has norm ~7), so probing at zero
    # reports deltas from an out-of-distribution corner of the input
    # space. That distinction is not cosmetic: at zero context the model
    # reports jaw_delta[0] = -0.044 for BOTH neutral and sad, which reads
    # as "sad clamps the jaw shut" -- while at the real context mean the
    # same checkpoint gives -0.0001 (neutral) and -0.0034 (sad), i.e.
    # sad barely touches the jaw at all, matching the tracked data
    # (sad's true mean jaw-x delta is -0.0013 rad, ~0.07 degrees).
    model.eval()
    with torch.no_grad():
        real_ctx = context_pool_t.mean(dim=0, keepdim=True).expand(len(EMOTION_CLASSES), -1)
        all_ids = torch.arange(len(EMOTION_CLASSES), device=device)
        class_expr_delta, class_jaw_delta = model(all_ids, real_ctx)
        class_expr_delta = class_expr_delta.cpu().numpy()
        class_jaw_delta = class_jaw_delta.cpu().numpy()

    print("\n[verify] per-class predicted expr-delta norm (at mean real driving context):")
    for i, cls in enumerate(EMOTION_CLASSES):
        print(f"  {cls}: norm={np.linalg.norm(class_expr_delta[i]):.4f} jaw_delta={class_jaw_delta[i]}")

    print("\n[verify] pairwise cosine similarity between class expr-deltas (should NOT all be ~1.0):")
    norm = class_expr_delta / (np.linalg.norm(class_expr_delta, axis=1, keepdims=True) + 1e-8)
    cos_sim = norm @ norm.T
    for i, cls in enumerate(EMOTION_CLASSES):
        row = " ".join(f"{cos_sim[i, j]:+.2f}" for j in range(len(EMOTION_CLASSES)))
        print(f"  {cls:10s} {row}")

    ckpt_path = os.path.join(output_dir, "emotion_adapter.pt")
    torch.save(model.state_dict(), ckpt_path)
    summary = {
        "manifest_path": manifest_path,
        "num_references_per_class": {cls: len(targets_by_class[cls]) for cls in EMOTION_CLASSES},
        "num_references_kept_per_class": kept_per_class,
        "trim_mode": trim_mode,
        "trim_keep_fraction": trim_keep_fraction,
        "trim_iters": trim_iters,
        "max_expr_delta": max_expr_delta,
        "max_jaw_delta": max_jaw_delta,
        "context_pool_size": int(context_pool.shape[0]),
        "num_steps": num_steps,
        "final_loss_expr": loss_history[-1]["loss_expr"],
        "final_loss_jaw": loss_history[-1]["loss_jaw"],
        "class_expr_delta_norm_at_real_context": {cls: float(np.linalg.norm(class_expr_delta[i])) for i, cls in enumerate(EMOTION_CLASSES)},
        "class_pairwise_cosine_sim": cos_sim.tolist(),
        "checkpoint": ckpt_path,
    }
    summary_path = os.path.join(output_dir, "emotion_adapter_train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\ncheckpoint={ckpt_path}")
    print(f"summary={summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the discrete-emotion EmotionAdapter.")
    parser.add_argument("--manifest_path", default="exps/train_lam/emotion_refs/tracked_emotion_refs.json")
    parser.add_argument("--output_dir", default="exps/train_lam/emotion_adapter")
    parser.add_argument("--num_steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--trim_keep_fraction", type=float, default=0.5,
                        help="Fraction of each emotion class's refs to keep, by cosine to the "
                             "running class mean. 1.0 restores the old plain-mean behaviour.")
    parser.add_argument("--trim_iters", type=int, default=5)
    parser.add_argument("--symmetrize_targets", type=float, default=0.0,
                        help="Make per-reference target deltas laterally symmetric before "
                             "trimming/regression (0=off, 1=fully symmetric). Removes the "
                             "35-72%% asymmetric residual that otherwise makes the rendered "
                             "mouth pull sideways differently for every emotion class.")
    parser.add_argument("--trim_mode", choices=["discriminative", "self"], default="discriminative",
                        help="'discriminative' (default) also pushes confusable classes apart from "
                             "each other, not just each class toward its own internal consensus -- "
                             "see discriminative_trim_indices. 'self' is the old per-class-only "
                             "behavior.")
    parser.add_argument("--max_expr_delta", type=float, default=3.0,
                        help="Per-component tanh cap on the predicted expr delta. Must exceed the "
                             "largest per-component value in the tracked targets (measured max: "
                             "2.54, happy) -- below that it reshapes the delta instead of bounding "
                             "it, flattening the leading expression components. See "
                             "lam/models/emotion_adapter.py.")
    parser.add_argument("--max_jaw_delta", type=float, default=0.15,
                        help="Per-component tanh cap on the predicted jaw delta (measured max "
                             "target: 0.084 rad, surprise -- not currently binding).")
    args = parser.parse_args()
    train({
        "manifest_path": args.manifest_path,
        "output_dir": args.output_dir,
        "num_steps": args.num_steps,
        "lr": args.lr,
        "trim_keep_fraction": args.trim_keep_fraction,
        "trim_iters": args.trim_iters,
        "trim_mode": args.trim_mode,
        "symmetrize_targets": args.symmetrize_targets,
        "max_expr_delta": args.max_expr_delta,
        "max_jaw_delta": args.max_jaw_delta,
    })


if __name__ == "__main__":
    main()
