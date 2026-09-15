#!/usr/bin/env python3
"""Score emotion accuracy on one or more render arms with both judges.

Usage:
  lam_env/bin/python score_emotion_arms.py unstyled=exps/images/emoeval_unstyled \
                                           styled=exps/images/emoeval_styled_pop

Each arm directory is laid out <arm>/<emotion>/<content_id>/stylized_preview.png,
which is what run_styled_emotion_eval.sh produces.

Two judges are reported for every arm:

  py-feat (resmasknet)  -- INDEPENDENT of this pipeline. This is the reportable
                           EmoAcc.
  ViT (trpakov/vit-face-expression) -- CIRCULAR whenever emotion_optimize=true,
                           because optimize_emotion_expr backpropagates through
                           this exact classifier. Printed only so the gap
                           between the two stays visible; never report it as an
                           evaluation result.

Both judges were benchmarked on real RAVDESS photographs (n=12/class); those
accuracies are printed as the reference ceiling, since a judge that cannot
recognise an emotion in a real human face says nothing about a render of it.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

E = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CEIL = {"anger": (0, 33), "disgust": (0, 67), "fear": (33, 83), "happy": (100, 67),
        "neutral": (92, 100), "sad": (50, 25), "surprise": (67, 50)}  # (vit, feat)

arms = {}
for a in sys.argv[1:]:
    if "=" not in a:
        sys.exit(f"expected name=path, got {a!r}")
    k, v = a.split("=", 1)
    arms[k] = v
if not arms:
    sys.exit(__doc__)

from transformers import pipeline
from feat import Detector
vit = pipeline("image-classification", model="trpakov/vit-face-expression", device=-1)
det = Detector(au_model="xgb", emotion_model="resmasknet", device="cpu")
VL = {"angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "happy",
      "neutral": "neutral", "sad": "sad", "surprise": "surprise"}
FM = {"anger": "anger", "disgust": "disgust", "fear": "fear", "happiness": "happy",
      "happy": "happy", "neutral": "neutral", "sadness": "sad", "sad": "sad",
      "surprise": "surprise"}

results = {}
for arm, root in arms.items():
    res = {e: {"vit": [], "feat": []} for e in E}
    n_missing = 0
    for e in E:
        d = os.path.join(root, e)
        if not os.path.isdir(d):
            continue
        for cid in sorted(os.listdir(d)):
            p = os.path.join(d, cid, "stylized_preview.png")
            if not os.path.exists(p):
                n_missing += 1
                continue
            pr = vit(p, top_k=7)
            res[e]["vit"].append(VL[max(pr, key=lambda x: x["score"])["label"]] == e)
            try:
                row = det.detect_image(p).emotions.iloc[0]
                res[e]["feat"].append(FM.get(str(max(row.index, key=lambda c: row[c])).lower()) == e)
            except Exception:
                pass          # py-feat found no face; that image simply does not vote
        print(f"  [{arm}] {e}: {len(res[e]['vit'])} scored", flush=True)
    results[arm] = res
    if n_missing:
        print(f"  [{arm}] {n_missing} renders missing", flush=True)

def pct(v):
    return np.mean(v) * 100 if v else float("nan")

names = list(arms)
print()
print("py-feat (INDEPENDENT judge) -- this is the reportable EmoAcc")
head = f"{'emotion':10s}" + "".join(f"{n:>12s}" for n in names) + f"{'real photos':>13s}"
print(head); print("-" * len(head))
for e in E:
    line = f"{e:10s}" + "".join(f"{pct(results[n][e]['feat']):11.1f}%" for n in names)
    print(line + f"{CEIL[e][1]:12d}%")
line = f"{'OVERALL':10s}" + "".join(
    f"{pct([x for e in E for x in results[n][e]['feat']]):11.1f}%" for n in names)
print("-" * len(head)); print(line + f"{61:12d}%")

print()
print("ViT -- CIRCULAR under emotion_optimize=true, do not report")
print(head); print("-" * len(head))
for e in E:
    line = f"{e:10s}" + "".join(f"{pct(results[n][e]['vit']):11.1f}%" for n in names)
    print(line + f"{CEIL[e][0]:12d}%")
line = f"{'OVERALL':10s}" + "".join(
    f"{pct([x for e in E for x in results[n][e]['vit']]):11.1f}%" for n in names)
print("-" * len(head)); print(line + f"{49:12d}%")

if len(names) == 2:
    a, b = names
    fa = pct([x for e in E for x in results[a][e]["feat"]])
    fb = pct([x for e in E for x in results[b][e]["feat"]])
    print(f"\npy-feat delta ({b} - {a}): {fb - fa:+.1f} points")

json.dump({n: {e: {k: [bool(x) for x in v] for k, v in d.items()}
               for e, d in r.items()} for n, r in results.items()},
          open("emotion_arm_scores.json", "w"), indent=1)
print("\nper-image votes written to emotion_arm_scores.json")
