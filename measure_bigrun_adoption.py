#!/usr/bin/env python3

import os, sys, glob, json, pickle, warnings, importlib.util
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image

_sp = importlib.util.spec_from_file_location("_ct", "lam/stylization/color_transfer.py")
_ct = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_ct)
_lab_stats = _ct._lab_stats

ROOT = "exps/images/bigrun"
TRACK = "tracking_output/export"
E = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CONTENT = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob("new_input_image/*"))]
STYLES = sorted(glob.glob("new-style-image/*"))[:7]
done = [i for i in range(len(STYLES))
        if len(glob.glob(f"{ROOT}/styled/s{i}/*/*/stylized_preview.png")) >= 126]
print(f"{len(CONTENT)} contents, styles complete: {done}\n")

# ---------------- geometry ----------------
fm = pickle.load(open("model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl", "rb"),
                 encoding="latin1")
SD = np.array(fm["shapedirs"])[:, :, :300]
VT = np.array(fm["v_template"])
DIAG = np.linalg.norm(VT.max(0) - VT.min(0))
def betas(name):
    p = os.path.join(TRACK, name, "canonical_flame_param.npz")
    return np.asarray(np.load(p, allow_pickle=True)["shape"]).reshape(-1) if os.path.exists(p) else None

print("GEOMETRY  (style_geometry_strength=1.0)")
print(f"{'style':34s}{'content->style gap':>20s}{'residual':>11s}{'adopted':>9s}")
print("-" * 74)
gall = []
for i in done:
    sn = os.path.splitext(os.path.basename(STYLES[i]))[0]
    bs = betas(sn)
    if bs is None:
        print(f"s{i} {sn[:30]:31s}  style not tracked"); continue
    ds = []
    for cid in CONTENT:
        bc = betas(cid)
        if bc is None: continue
        ds.append(np.linalg.norm(SD @ (bs[:300] - bc[:300]), axis=1).mean())
    if not ds: continue
    gall += ds
    print(f"s{i} {sn[:30]:31s}{np.mean(ds)*1000:17.2f} mm{0.0:10.3f} mm{100.0:8.1f}%")
if gall:
    print("-" * 74)
    print(f"{'MEAN':34s}{np.mean(gall)*1000:17.2f} mm{0.0:10.3f} mm{100.0:8.1f}%")
    print(f"\n  adoption is 100% by construction (betas = content + 1.0*(style-content));")
    print(f"  the gap being closed is {np.mean(gall)*1000:.2f} mm = "
          f"{np.mean(gall)/DIAG*100:.2f}% of the {DIAG*100:.0f} cm head bbox diagonal")

# ---------------- colour ----------------
def stats(p):
    m, s = _lab_stats(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8), None, True)
    return np.concatenate([m, s])
def tracked(p):
    n = os.path.splitext(os.path.basename(p))[0]
    i = os.path.join(TRACK, n, "images", "00000_00.png")
    return i if os.path.exists(i) else p

print("\n\nCOLOUR  gap closure vs the pipeline's own unstyled render")
print(f"{'style':34s}{'|unstyled-style|':>18s}{'|styled-style|':>16s}{'adopted':>9s}{'n':>6s}")
print("-" * 83)
res = {}
for i in done:
    sl = stats(tracked(STYLES[i]))
    du, ds_, ad = [], [], []
    for e in E:
        for cid in CONTENT:
            ps = f"{ROOT}/styled/s{i}/{e}/{cid}/stylized_preview.png"
            pu = f"{ROOT}/unstyled/{e}/{cid}/stylized_preview.png"
            if not (os.path.exists(ps) and os.path.exists(pu)): continue
            a = float(np.linalg.norm(stats(pu) - sl)); b = float(np.linalg.norm(stats(ps) - sl))
            du.append(a); ds_.append(b); ad.append(1 - b / (a + 1e-9))
    res[i] = ad
    sn = os.path.splitext(os.path.basename(STYLES[i]))[0]
    print(f"s{i} {sn[:30]:31s}{np.mean(du):18.1f}{np.mean(ds_):16.1f}"
          f"{np.mean(ad)*100:8.1f}%{len(ad):6d}")
allad = [v for a in res.values() for v in a]
if allad:
    print("-" * 83)
    print(f"{'MEAN':34s}{'':18s}{'':16s}{np.mean(allad)*100:8.1f}%{len(allad):6d}")
    print(f"\n  sd {np.std(allad)*100:.1f} points;  per-style "
          f"{min(np.mean(a) for a in res.values())*100:.0f}% to "
          f"{max(np.mean(a) for a in res.values())*100:.0f}%")

# per-component: which parts of colour actually transfer
NAMES = ["L mean", "a mean", "b mean", "L sd", "a sd", "b sd"]
acc = [[] for _ in range(6)]
for i in done:
    sl = stats(tracked(STYLES[i]))
    for e in E:
        for cid in CONTENT:
            ps = f"{ROOT}/styled/s{i}/{e}/{cid}/stylized_preview.png"
            pu = f"{ROOT}/unstyled/{e}/{cid}/stylized_preview.png"
            if not (os.path.exists(ps) and os.path.exists(pu)): continue
            u, s_ = stats(pu), stats(ps)
            for k in range(6):
                g = sl[k] - u[k]
                if abs(g) > 1.0: acc[k].append((s_[k] - u[k]) / g)
print("\n  per-component (median; the ratio is unstable where a gap is near zero)")
for k in range(6):
    if acc[k]: print(f"    {NAMES[k]:8s}{np.median(acc[k])*100:8.1f}%   n={len(acc[k])}")
json.dump({"colour": {str(k): [float(x) for x in v] for k, v in res.items()},
           "geometry_mm": float(np.mean(gall) * 1000) if gall else None},
          open("bigrun_adoption.json", "w"), indent=1)
