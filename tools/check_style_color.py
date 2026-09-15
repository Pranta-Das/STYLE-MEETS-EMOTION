"""Quantitative check for whether style color transfer actually worked.

A visual glance can be misleading -- a render can look "different" while
actually drifting in the wrong direction, or barely moving at all (see
STYLE_GEOMETRY_PIPELINE.md for real cases of both). This compares an unstyled
render against a styled render of the SAME content+style pair and reports:

  1. Whether the color moved TOWARD the style image or away from it
     (cosine similarity between the actual RGB shift and the target shift --
     1.0 = perfect direction, 0 = no relationship, negative = wrong way).
  2. How much of the way there it got (magnitude ratio).
  3. Saturation, for a quick "did it desaturate/saturate as expected" read.
  4. Region differentiation (hair vs. skin) so a uniform wash and a properly
     differentiated multi-region transfer don't look the same on paper.

Usage:
    python tools/check_style_color.py \\
        --unstyled path/to/unstyled_frame.png \\
        --styled path/to/styled_frame.png \\
        --style path/to/style_image.png

Generate the unstyled baseline once per content+geometry setting with the
same CLI, style_optimize_color=false and
style_internal_appearance_strength=0.0 style_geometry_strength=0.0
style_strength=0.0 -- see STYLE_GEOMETRY_PIPELINE.md's "Validating changes"
section.
"""
import argparse

import numpy as np
from PIL import Image


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def foreground_mask(img, white_thresh=245):
    # LAM renders on a plain white background (background_color=1.0). For a
    # bust/portrait crop that doesn't fill the frame, background pixels can
    # be the majority of the image -- whole-image means then get dominated
    # by tiny, largely noisy shifts in that background rather than reflecting
    # the actual subject, which can even flip the reported direction sign
    # while the subject itself looks essentially unchanged. Excluding
    # near-white pixels keeps the statistics about the subject.
    return ~((img[..., 0] > white_thresh) & (img[..., 1] > white_thresh) & (img[..., 2] > white_thresh))


def masked_mean(img, mask):
    px = img[mask]
    if px.size == 0:
        return img.reshape(-1, 3).mean(0)
    return px.reshape(-1, 3).mean(0)


def mean_saturation(img, mask=None):
    px = img / 255.0
    if mask is not None:
        px = px[mask]
    maxc = px.max(axis=-1)
    minc = px.min(axis=-1)
    sat = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0)
    return float(sat.mean())


def region_mean(img, y0, y1, x0, x1):
    h, w, _ = img.shape
    region = img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    mask = foreground_mask(region)
    return masked_mean(region, mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unstyled", required=True, help="Rendered frame with style OFF")
    parser.add_argument("--styled", required=True, help="Same content+pose, style ON")
    parser.add_argument("--style", required=True, help="The style reference image itself")
    parser.add_argument(
        "--hair-box", default="0.05,0.20,0.30,0.70",
        help="y0,y1,x0,x1 fractional box for the hair region (default roughly top of a portrait frame)",
    )
    parser.add_argument(
        "--skin-box", default="0.45,0.60,0.40,0.60",
        help="y0,y1,x0,x1 fractional box for a skin region (default roughly cheek/nose of a portrait frame)",
    )
    args = parser.parse_args()

    unstyled = load_rgb(args.unstyled)
    styled = load_rgb(args.styled)
    style = load_rgb(args.style)

    # Masking to foreground/subject pixels only -- see foreground_mask()'s
    # docstring for why the raw whole-frame mean is unreliable here.
    u_mean = masked_mean(unstyled, foreground_mask(unstyled))
    s_mean = masked_mean(styled, foreground_mask(styled))
    style_mean = masked_mean(style, foreground_mask(style))

    actual_shift = s_mean - u_mean
    target_shift = style_mean - u_mean
    actual_norm = np.linalg.norm(actual_shift)
    target_norm = np.linalg.norm(target_shift)
    cos_sim = float(np.dot(actual_shift, target_shift) / (actual_norm * target_norm + 1e-8))
    magnitude_ratio = float(actual_norm / (target_norm + 1e-8))

    hy0, hy1, hx0, hx1 = (float(v) for v in args.hair_box.split(","))
    sy0, sy1, sx0, sx1 = (float(v) for v in args.skin_box.split(","))
    styled_hair = region_mean(styled, hy0, hy1, hx0, hx1)
    styled_skin = region_mean(styled, sy0, sy1, sx0, sx1)
    unstyled_hair = region_mean(unstyled, hy0, hy1, hx0, hx1)
    unstyled_skin = region_mean(unstyled, sy0, sy1, sx0, sx1)
    region_diff_styled = float(np.linalg.norm(styled_hair - styled_skin))
    region_diff_unstyled = float(np.linalg.norm(unstyled_hair - unstyled_skin))

    print("=" * 60)
    print("COLOR DIRECTION")
    print("=" * 60)
    print(f"unstyled mean RGB : {u_mean.round(1)}")
    print(f"styled mean RGB   : {s_mean.round(1)}")
    print(f"style image RGB   : {style_mean.round(1)}")
    print(f"actual shift      : {actual_shift.round(1)}")
    print(f"target shift      : {target_shift.round(1)}")
    print(f"direction (cosine similarity, 1.0=perfect, 0=unrelated, <0=WRONG WAY): {cos_sim:.3f}")
    print(f"magnitude (how far along the target shift, 1.0=matched it exactly): {magnitude_ratio:.3f}")
    if cos_sim < 0.3:
        print(">> VERDICT: color is NOT moving toward the style image (wrong direction or no real effect).")
    elif magnitude_ratio < 0.15:
        print(">> VERDICT: direction is right but the effect is very weak -- barely visible.")
    else:
        print(">> VERDICT: color is moving toward the style image at a real magnitude.")

    print()
    print("=" * 60)
    print("SATURATION (0=grayscale, higher=more colorful)")
    print("=" * 60)
    print(f"unstyled : {mean_saturation(unstyled, foreground_mask(unstyled)):.4f}")
    print(f"styled   : {mean_saturation(styled, foreground_mask(styled)):.4f}")
    print(f"style    : {mean_saturation(style, foreground_mask(style)):.4f}")

    print()
    print("=" * 60)
    print("REGION DIFFERENTIATION (hair vs. skin -- uses --hair-box/--skin-box)")
    print("=" * 60)
    print(f"unstyled hair-skin L2 diff: {region_diff_unstyled:.1f}")
    print(f"styled   hair-skin L2 diff: {region_diff_styled:.1f}")
    if region_diff_styled < region_diff_unstyled * 0.5:
        print(">> Regions got MORE similar after styling -- likely a uniform wash, not real per-region transfer.")
    else:
        print(">> Regions kept (or gained) differentiation -- consistent with real per-region color transfer.")
    print()
    print("Note: --hair-box/--skin-box default to a generic portrait framing -- adjust")
    print("them (fractions of image height/width: y0,y1,x0,x1) if your subject's hair")
    print("or face isn't roughly in those regions.")


if __name__ == "__main__":
    main()
