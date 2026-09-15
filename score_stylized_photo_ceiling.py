#!/usr/bin/env python3
"""Control for the styled emotion table: how much accuracy does each judge lose
purely because the image is stylised?

If EmoAcc drops when stylisation is switched on, that drop has two possible
causes which the styled column alone cannot separate:

  (a) our stylisation damages the expression, or
  (b) the judge is simply worse at reading a non-photorealistic face.

This measures (b) directly. It takes the same real RAVDESS photographs used for
judge calibration -- whose emotion is ground truth and is NOT touched by
anything here -- applies the pipeline's own 2D stylisation pass to them, and
re-scores. Any accuracy lost is loss caused by stylised appearance alone.

Scope caveat, needed when reporting this: the pipeline's stylisation is 2D
colour transfer PLUS geometry blending and an in-model appearance adapter.
Only the colour transfer can be applied to a flat photograph, so this is a
LOWER bound on the judge's degradation, not the whole of it.
"""
import os, glob, sys, json, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
import numpy as np
from PIL import Image

E = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
N = int(os.environ.get("N_PER_CLASS", "12"))
STYLE = sys.argv[1] if len(sys.argv) > 1 else "assets/sample_input/pop.png"
OUT = "exps/images/stylized_photo_ceiling"

# Load the module by path: importing it through the `lam.stylization` package
# pulls in the whole inference runner and hits a circular import. Only the
# colour-transfer function is needed here, and it depends on nothing else.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_color_transfer", "lam/stylization/color_transfer.py")
_ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ct)
stylize_frames_with_reference = _ct.stylize_frames_with_reference
from transformers import pipeline
from feat import Detector
vit = pipeline("image-classification", model="trpakov/vit-face-expression", device=-1)
det = Detector(au_model="xgb", emotion_model="resmasknet", device="cpu")
VL = {"angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "happy",
      "neutral": "neutral", "sad": "sad", "surprise": "surprise"}
FM = {"anger": "anger", "disgust": "disgust", "fear": "fear", "happiness": "happy",
      "happy": "happy", "neutral": "neutral", "sadness": "sad", "sad": "sad",
      "surprise": "surprise"}

def judge(path, e):
    pr = vit(path, top_k=7)
    v = VL[max(pr, key=lambda x: x["score"])["label"]] == e
    try:
        row = det.detect_image(path).emotions.iloc[0]
        f = FM.get(str(max(row.index, key=lambda c: row[c])).lower()) == e
    except Exception:
        f = None
    return v, f

def files(e):
    fs = sorted(glob.glob(f"data/emotion_refs/images/{e}/ravdess_*-02-01-01-*.jpg"))[:N]
    return fs or sorted(glob.glob(f"data/emotion_refs/images/{e}/ravdess_*.jpg"))[:N]

raw, sty = {e: {"vit": [], "feat": []} for e in E}, {e: {"vit": [], "feat": []} for e in E}
for e in E:
    d = os.path.join(OUT, e)
    os.makedirs(d, exist_ok=True)
    for p in files(e):
        v, f = judge(p, e)
        raw[e]["vit"].append(v)
        if f is not None:
            raw[e]["feat"].append(f)
        sp = os.path.join(d, os.path.basename(p).replace(".jpg", ".png"))
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        out = stylize_frames_with_reference(arr[None], style_image_path=STYLE,
                                            masks=None, strength=0.75)[0]
        Image.fromarray(out).save(sp)
        v, f = judge(sp, e)
        sty[e]["vit"].append(v)
        if f is not None:
            sty[e]["feat"].append(f)
    print(f"  {e}: {len(raw[e]['vit'])} photographs", flush=True)

def pct(v):
    return np.mean(v) * 100 if v else float("nan")

print(f"\nreal RAVDESS photographs, stylised with {STYLE}")
head = (f"{'emotion':10s}{'py-feat raw':>13s}{'py-feat sty':>13s}"
        f"{'ViT raw':>10s}{'ViT sty':>10s}")
print(head); print("-" * len(head))
for e in E:
    print(f"{e:10s}{pct(raw[e]['feat']):12.0f}%{pct(sty[e]['feat']):12.0f}%"
          f"{pct(raw[e]['vit']):9.0f}%{pct(sty[e]['vit']):9.0f}%")
allf = lambda d, k: [x for e in E for x in d[e][k]]
print("-" * len(head))
print(f"{'OVERALL':10s}{pct(allf(raw,'feat')):12.0f}%{pct(allf(sty,'feat')):12.0f}%"
      f"{pct(allf(raw,'vit')):9.0f}%{pct(allf(sty,'vit')):9.0f}%")
print(f"\ncost of stylised appearance alone: py-feat "
      f"{pct(allf(sty,'feat')) - pct(allf(raw,'feat')):+.0f} points, "
      f"ViT {pct(allf(sty,'vit')) - pct(allf(raw,'vit')):+.0f} points")
json.dump({"raw": {e: {k: [bool(x) for x in v] for k, v in d.items()} for e, d in raw.items()},
           "styled": {e: {k: [bool(x) for x in v] for k, v in d.items()} for e, d in sty.items()},
           "style_image": STYLE},
          open("stylized_photo_ceiling.json", "w"), indent=1)
