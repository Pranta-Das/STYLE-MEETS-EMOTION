#!/usr/bin/env python3
"""Demo figure: ONE person's performance transferred onto SEVERAL other faces.

This is the base reenactment path, NOT the 7-class emotion adapter. No emotion
label is passed anywhere: every expression in every output row is the driving
clip's own tracked FLAME expr/jaw, replayed on a different identity. Holding the
driving clip and the style fixed while varying only the identity is what makes
this evidence for disentanglement rather than a single nice-looking transfer --
if the same six expressions appear on a robot, a marble bust and a photograph,
the expression code is demonstrably not carrying identity.

Frames are chosen by greedy farthest-point sampling in (expr, jaw) space so the
columns are genuinely different moments from across the clip, not six
near-duplicates from wherever it happens to start.

Usage:
  lam_env/bin/python make_reenactment_figure.py --style <style.jpg> \
      --out fig.png <content1> <content2> ...
"""
import os, sys, glob, subprocess, argparse
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
CLIP = "assets/sample_motion/export/Look_In_My_Eyes"
NCOL = 6

ap = argparse.ArgumentParser()
ap.add_argument("content", nargs="+")
ap.add_argument("--style", default="style-image/000_256_505_4k_mark-makovey-p-029_00.png")
ap.add_argument("--out", default="reenactment_demo.png")
ap.add_argument("--labels", default="", help="comma-separated row labels")
a = ap.parse_args()
labels = [x.strip() for x in a.labels.split(",")] if a.labels else \
         [os.path.splitext(os.path.basename(c))[0] for c in a.content]

for f in list(a.content) + [a.style]:
    if not os.path.exists(f):
        sys.exit(f"no such file: {f}")

def render(content):
    cid = os.path.splitext(os.path.basename(content))[0]
    # The style MUST be part of the cache key. Keying on the content id alone
    # silently reused a render made with a different style reference, which put
    # one row of an earlier figure in the wrong style while the caption claimed
    # the style was held fixed.
    sid = os.path.splitext(os.path.basename(a.style))[0][:24]
    dump = f"exps/images/reenact_{cid}__{sid}"
    if os.path.exists(os.path.join(dump, cid, "0000.png")):
        return sorted(glob.glob(os.path.join(dump, cid, "0*.png")))
    print(f"rendering {cid} ...", flush=True)
    r = subprocess.run([
        "lam_env/bin/python", "-m", "lam.launch", "infer.lam",
        "--config", "configs/inference/lam-20k-8gpu.yaml",
        "model_name=exps/train_lam/style_appearance_e2e_v8/model_final.pt",
        f"image_input={content}", f"motion_seqs_dir={CLIP}/",
        "export_video=false", "export_mesh=true", "test_sample=true",
        f"style_image_path={a.style}",
        "style_optimize_color=true", "style_optimize_steps=100",
        "style_optimize_num_views=4",
        "style_geometry_strength=1.0", "style_track_geometry=true",
        "seed=42", "save_img=true",
        f"image_dump={dump}", f"video_dump={dump}_v",
    ], env=dict(os.environ, PYTHONPATH=".", CUDA_VISIBLE_DEVICES="0"),
       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = sorted(glob.glob(os.path.join(dump, cid, "0*.png")))
    if not out:
        print(r.stdout[-2500:]); print(r.stderr[-2500:]); sys.exit(f"render failed: {cid}")
    return out

rows = [render(c) for c in a.content]
n = min(len(r) for r in rows)
print(f"{n} frames per identity")

# test_sample=true keeps 50 evenly spaced frames; same mapping head_utils uses,
# so render i is driving frame ids[i] and every row stays in sync.
fs = sorted(glob.glob(os.path.join(CLIP, "flame_param", "*.npz")))
ids = np.linspace(0, len(fs) - 1, 50).astype(np.int32)[:n]
feat = np.array([np.concatenate([np.load(fs[i])["expr"].reshape(-1)[:100],
                                 np.load(fs[i])["jaw_pose"].reshape(-1)[:3] * 50.0])
                 for i in ids])
sel = [int(np.argmax(np.linalg.norm(feat - feat.mean(0), axis=1)))]
while len(sel) < NCOL:
    d = np.min([np.linalg.norm(feat - feat[s], axis=1) for s in sel], axis=0)
    sel.append(int(np.argmax(d)))
sel = sorted(sel)
print("driving frames:", ids[sel].tolist())

cap = cv2.VideoCapture(os.path.join(CLIP, "Look_In_My_Eyes.mp4"))
drive = {}
for k in sel:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ids[k])); ok, fr = cap.read()
    if ok:
        h, w = fr.shape[:2]; s = min(h, w)
        fr = fr[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
        drive[k] = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
cap.release()

C, PAD, HEAD, GUT = 230, 22, 52, 190
NROW = 1 + len(rows)
W = PAD*2 + GUT + NCOL*C
H = PAD*2 + HEAD + NROW*(C+26) + 46
im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
fb = ImageFont.truetype(F % "-Bold", 21)
fr_ = ImageFont.truetype(F % "-Bold", 15)
fn = ImageFont.truetype(F % "", 14)

d.text((PAD, PAD), "One performance, three faces: the expression transfers, the identity does not",
       fill="black", font=fb)
d.text((PAD, PAD+26),
       f"style reference held fixed for every output row  ({os.path.basename(a.style)[:46]})",
       fill="#666", font=fn)

y = PAD + HEAD
d.text((PAD, y+6), "DRIVING VIDEO", fill="#111", font=fr_)
d.text((PAD, y+24), "source performance", fill="#666", font=fn)
for j, k in enumerate(sel):
    if k in drive:
        im.paste(drive[k].resize((C-6, C-6)), (PAD+GUT+j*C+3, y))
d.line([(PAD, y+C+10), (W-PAD, y+C+10)], fill="#bbb", width=2)
y += C + 26

for ri, (frames, lab, cpath) in enumerate(zip(rows, labels, a.content)):
    im.paste(Image.open(cpath).convert("RGB").resize((72, 72)), (PAD, y+2))
    d.text((PAD+80, y+8), lab[:16], fill="#111", font=fr_)
    d.text((PAD+80, y+28), "output", fill="#666", font=fn)
    for j, k in enumerate(sel):
        im.paste(Image.open(frames[k]).convert("RGB").resize((C-6, C-6)),
                 (PAD+GUT+j*C+3, y))
    y += C + 26

d.text((PAD, H-PAD-16),
       "No emotion label is used anywhere. Every expression above is the driving clip's own "
       "tracked FLAME expression and jaw pose, replayed on each input identity.",
       fill="#333", font=fn)
im.save(a.out)
print(f"\n{a.out}  {im.size[0]}x{im.size[1]}")
