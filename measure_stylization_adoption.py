#!/usr/bin/env python3
"""Exactly how much of the style image does the output adopt, for GEOMETRY and
for COLOUR, measured on the real 5x5 grid pairs.

GEOMETRY is measured in 3D vertex space, not parameter space. `betas` are PCA
coefficients, so a percentage of them means nothing physical; the meshes are
built with the actual FLAME model in a fixed neutral pose (expression and jaw
zeroed, so only identity shape differs) and distances are reported in
millimetres and as a fraction of the content->style gap.

COLOUR is measured on the rendered image against the two references in CIELAB
mean/std space (6 numbers: L,a,b mean + L,a,b std), head pixels only. Adoption
is the projection of (render - content) onto (style - content) divided by
|style - content|^2 -- i.e. how far along the line from the content photo's
colour statistics to the style's the render actually travelled. 0.0 = kept the
content's colour, 1.0 = matched the style exactly.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import numpy as np, torch, cv2, importlib.util
_sp = importlib.util.spec_from_file_location("_ct", "lam/stylization/color_transfer.py")
_ct = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_ct)
_lab_stats = _ct._lab_stats
from PIL import Image

CONTENT = {"cluo": "assets/sample_input/cluo.jpg", "james": "assets/sample_input/james.png",
           "00028": "original-image/00028.jpg", "00052": "original-image/00052.jpg",
           "00062": "original-image/00062.jpg"}
STYLES = {0: "assets/sample_input/pop.png",
          1: "style-image/000_256_505_4k_mark-makovey-p-029_00.png",
          2: "style-image/001_339_379_4k_irakli-nadar-jinxartstation_00.png",
          3: "style-image/002_345_899_4k_nikita-orlov-portarait-womanblack_00.png",
          4: "style-image/003_832_838_4k_hany-abbas-sayed-ragab4.jpg"}
E = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TRACK = "tracking_output/export"

def betas(name):
    p = os.path.join(TRACK, name, "canonical_flame_param.npz")
    if not os.path.exists(p):
        return None
    return np.asarray(np.load(p, allow_pickle=True)["shape"]).reshape(-1)

# ---------- geometry ----------
# FLAME identity shape is exactly linear in betas: v = v_template + shapedirs @ betas.
# So the vertex displacement between two identities is shapedirs @ (b1 - b2) with no
# forward pass, no pose and no teeth/subdivision complications to confound it.
import pickle
with open("model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl", "rb") as f:
    fm = pickle.load(f, encoding="latin1")
SD = np.array(fm["shapedirs"])[:, :, :300]          # [V, 3, 300]
VT = np.array(fm["v_template"])                     # [V, 3], metres
def disp(db):
    """vertex displacement (metres) produced by a beta difference"""
    return SD @ db[:300]

print("GEOMETRY -- 3D vertex displacement, neutral pose, identity shape only\n")
print(f"{'content':9s}{'style':16s}{'content->style':>15s}{'blend->style':>14s}{'adopted':>9s}")
print("-" * 63)
geo = []
for cid, cpath in CONTENT.items():
    bc = betas(os.path.splitext(os.path.basename(cpath))[0])
    if bc is None: continue
    diag = np.linalg.norm(VT.max(0) - VT.min(0))
    for si, spath in STYLES.items():
        bs = betas(os.path.splitext(os.path.basename(spath))[0])
        if bs is None:
            print(f"{cid:9s}{'s%d' % si:16s}   style not tracked"); continue
        n = min(len(bc), len(bs), 300)
        bb = bc.copy(); bb[:n] = bc[:n] + 1.0 * (bs[:n] - bc[:n])   # strength=1.0
        d_cs = np.linalg.norm(disp(bs - bc), axis=1).mean()
        d_bs = np.linalg.norm(disp(bs - bb), axis=1).mean()
        adopt = 1.0 - d_bs / (d_cs + 1e-12)
        geo.append((cid, si, d_cs * 1000, d_bs * 1000, adopt, d_cs / diag * 100))
        print(f"{cid:9s}{'s%d' % si:16s}{d_cs*1000:12.2f} mm{d_bs*1000:11.3f} mm{adopt*100:8.1f}%")
if geo:
    print("-" * 63)
    print(f"{'MEAN':25s}{np.mean([g[2] for g in geo]):12.2f} mm"
          f"{np.mean([g[3] for g in geo]):11.3f} mm{np.mean([g[4] for g in geo])*100:8.1f}%")
    print(f"\n  content->style shape gap = {np.mean([g[5] for g in geo]):.1f}% of head bbox diagonal")

# ---------- colour ----------
def head_lab(path, is_style=False):
    """CIELAB (mean, std) using the PIPELINE'S OWN definition of style pixels.

    `_lab_stats(..., ignore_style_background=True)` is the exact function
    stylize_frames_with_reference measures with, so this asks "did the output's
    colour statistics move to where the transfer was aiming", not "did they move
    to where I would have aimed".

    For the style, the input is the tracker's matted head crop (the `style_rgb`
    the pipeline itself passes), NOT the raw file -- _style_pixel_mask's own
    docstring explains that it is the wrong filter for a raw photo with a
    mid-tone backdrop. Note the tracker's fg_masks/00000_00.png is degenerate
    here (covers 100% of the crop); using it put the style's white matte into
    the statistics and gave L=202 instead of ~139, which is what produced the
    nonsensical -205% and -11.7% readings before this was caught.
    """
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    mean, std = _lab_stats(im, mask=None, ignore_style_background=True)
    return np.concatenate([mean, std])

def tracked(path):
    """The cached tracked head crop for this file, falling back to the raw file."""
    n = os.path.splitext(os.path.basename(path))[0]
    i = os.path.join(TRACK, n, "images", "00000_00.png")
    return i if os.path.exists(i) else path

# Adoption is GAP CLOSURE against the pipeline's OWN UNSTYLED RENDER:
#   1 - |styled - style| / |unstyled - style|
# 1.0 = the styled render's colour statistics match the style exactly, 0.0 = no
# closer to the style than the unstyled render was, negative = further away.
#
# The baseline MUST be the unstyled render, not the content photograph. A render
# is a Gaussian reconstruction on a white background, so its colour statistics
# differ from the source photo even with stylisation fully off; measuring
# against the photo measures the reconstruction, not the stylisation. (Same
# reason section 2a of EVALUATION_TABLES.md measures PSNR against the unstyled
# reconstruction.) Measured against the photo this same code reported -205%,
# which is how the error was caught.
print("\n\nCOLOUR -- CIELAB (mean,std) gap closure vs the pipeline's own unstyled render")
print("   head pixels only; pipeline applies LAB statistics transfer at style_strength=0.75\n")
print(f"{'style':16s}{'|unstyled-style|':>18s}{'|styled-style|':>16s}{'adopted':>9s}")
print("-" * 60)
col = {}
for si, spath in STYLES.items():
    sl = head_lab(tracked(spath), is_style=True)
    d_us, d_ss, ad = [], [], []
    for ci, cid in enumerate(CONTENT):
        for e in E:
            ps = f"exps/images/grid55/c{ci}_s{si}/{e}/{cid}/stylized_preview.png"
            pu = f"exps/images/grid55_unstyled/c{ci}/{e}/{cid}/stylized_preview.png"
            if not (os.path.exists(ps) and os.path.exists(pu)): continue
            du = float(np.linalg.norm(head_lab(pu) - sl))
            ds = float(np.linalg.norm(head_lab(ps) - sl))
            d_us.append(du); d_ss.append(ds); ad.append(1.0 - ds / (du + 1e-9))
    col[si] = ad
    print(f"{'s%d' % si:16s}{np.mean(d_us):18.1f}{np.mean(d_ss):16.1f}{np.mean(ad)*100:8.1f}%")
allad = [v for a in col.values() for v in a]
print("-" * 60)
print(f"{'MEAN':16s}{'':18s}{'':16s}{np.mean(allad)*100:8.1f}%")
print(f"\n  n = {len(allad)} paired renders;  sd {np.std(allad)*100:.1f} points;  "
      f"per-style {min(np.mean(a) for a in col.values())*100:.0f}% to "
      f"{max(np.mean(a) for a in col.values())*100:.0f}%")

json.dump({"geometry": [list(map(float, g[2:])) for g in geo],
           "colour": {str(k): [float(x) for x in v] for k, v in col.items()}},
          open("stylization_adoption.json", "w"), indent=1)
