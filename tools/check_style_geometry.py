"""Quantitative check for whether face geometry actually changed.

Rendered pixels are a bad way to judge this -- lighting, color, and camera
angle can all make a face LOOK different without the underlying shape having
moved, and vice versa. Every inference run with save_img=true already dumps
the actual FLAME shape mesh (exps/cano_gs/<name>_shaped_mesh.obj) -- the real
3D vertex positions, before any Gaussian xyz-offset or color. Since it's
always the same FLAME topology, vertex N in one mesh is the same anatomical
point as vertex N in another, so a direct per-vertex distance is an
unambiguous, ground-truth measurement of shape change -- no guessing from
pixels needed.

Usage:
    # 1. Run inference with style_geometry_strength=0.0, save the mesh aside:
    #    (after running with save_img=true)
    cp exps/cano_gs/status_shaped_mesh.obj /tmp/baseline_shaped_mesh.obj

    # 2. Run inference again with your real style_geometry_strength value.
    #    exps/cano_gs/status_shaped_mesh.obj now holds the styled version.

    # 3. Compare:
    python tools/check_style_geometry.py \\
        --baseline /tmp/baseline_shaped_mesh.obj \\
        --styled exps/cano_gs/status_shaped_mesh.obj

Note: exps/cano_gs/<name>_shaped_mesh.obj is a FIXED path (not affected by
image_dump/video_dump overrides) and gets overwritten by every run with the
same image_input -- always copy the baseline aside before generating the
styled version, or you'll be comparing a file to itself.
"""
import argparse

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="_shaped_mesh.obj with style_geometry_strength=0.0")
    parser.add_argument("--styled", required=True, help="_shaped_mesh.obj with your real geometry strength")
    args = parser.parse_args()

    base = trimesh.load(args.baseline, process=False)
    styled = trimesh.load(args.styled, process=False)

    if base.vertices.shape != styled.vertices.shape:
        raise SystemExit(
            f"Vertex count mismatch ({base.vertices.shape} vs {styled.vertices.shape}) -- "
            "these aren't the same FLAME topology, can't compare directly. "
            "Did both runs use the same model/config?"
        )

    disp = np.linalg.norm(styled.vertices - base.vertices, axis=-1)  # per-vertex displacement, meters (FLAME's own scale)

    print("=" * 60)
    print("GEOMETRY CHANGE (per-vertex displacement, FLAME's own scale -- roughly meters)")
    print("=" * 60)
    print(f"vertex count       : {len(disp)}")
    print(f"mean displacement   : {disp.mean():.5f}")
    print(f"max displacement    : {disp.max():.5f}")
    print(f"vertices moved >1mm : {(disp > 0.001).sum()} / {len(disp)} ({100 * (disp > 0.001).mean():.1f}%)")
    print(f"vertices moved >5mm : {(disp > 0.005).sum()} / {len(disp)} ({100 * (disp > 0.005).mean():.1f}%)")

    face_bbox_diag = np.linalg.norm(base.vertices.max(0) - base.vertices.min(0))
    relative = disp.max() / face_bbox_diag
    print(f"max displacement as % of face bounding-box diagonal: {100 * relative:.2f}%")

    print()
    if disp.max() < 1e-5:
        print(">> VERDICT: essentially zero movement -- geometry is NOT changing at all.")
        print("   Check: is style_geometry_strength actually being passed and > 0? Is")
        print("   style_track_geometry=true? Did FLAME tracking succeed on the style image")
        print("   (check the log for 'style flametracking failed' warnings)?")
    elif relative < 0.01:
        print(">> VERDICT: some movement, but very small relative to face size -- likely")
        print("   barely visible. Try a style image with more different proportions, or")
        print("   raise style_geometry_strength further.")
    else:
        print(">> VERDICT: real, meaningfully-sized geometry change.")


if __name__ == "__main__":
    main()
