#!/usr/bin/env python3
"""EDTalk vs ours: emotion accuracy on the same identities, same driving clip.

Both are scored by the same independent judge (py-feat) on a rendered frame.
Ours is scored on frame 0 of the driven sequence, so EDTalk is scored on its
frame 0 too -- the same instant of the same driving performance.

Cross-representation caveat: EDTalk emits 256x256 2D video from audio; ours
emits a 3D Gaussian avatar from a discrete label with no audio. This compares
emotion recognisability of the output, not equivalent systems.
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image
from feat import Detector
det=Detector(au_model="xgb", emotion_model="resmasknet", device="cpu")
FM={"anger":"anger","disgust":"disgust","fear":"fear","happiness":"happy","happy":"happy",
    "neutral":"neutral","sadness":"sad","sad":"sad","surprise":"surprise"}
E=["anger","disgust","fear","happy","neutral","sad","surprise"]
TMP="/tmp/edtalk_frames"; os.makedirs(TMP,exist_ok=True)

def judge(p):
    try:
        r=det.detect_image(p).emotions.iloc[0]
        return FM.get(str(max(r.index,key=lambda c:r[c])).lower())
    except Exception: return None

def frame0(mp4):
    out=os.path.join(TMP, os.path.basename(mp4).replace(".mp4",".png"))
    if not os.path.exists(out):
        c=cv2.VideoCapture(mp4); ok,f=c.read(); c.release()
        if not ok: return None
        cv2.imwrite(out,f)
    return out

ed={e:[] for e in E}; ours={e:[] for e in E}
cids=[os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob("baselines/EDTalk/prep/src/*.png"))]
for e in E:
    for cid in cids:
        mp4=f"exps/images/edtalk/{e}/{cid}.mp4"
        if os.path.exists(mp4):
            f=frame0(mp4)
            if f: ed[e].append(judge(f)==e)
        p=f"exps/images/bigrun/unstyled/{e}/{cid}/stylized_preview.png"
        if os.path.exists(p): ours[e].append(judge(p)==e)
    print(f"  {e}: edtalk {len(ed[e])}, ours {len(ours[e])}",flush=True)

print(f"\nEMOTION ACCURACY (py-feat, independent judge), same {len(cids)} identities")
print(f"{'emotion':10s}{'OURS':>10s}{'EDTalk':>10s}")
print("-"*30)
for e in E:
    a=np.mean(ours[e])*100 if ours[e] else float('nan')
    b=np.mean(ed[e])*100 if ed[e] else float('nan')
    print(f"{e:10s}{a:9.1f}%{b:9.1f}%")
A=[x for e in E for x in ours[e]]; B=[x for e in E for x in ed[e]]
print("-"*30)
print(f"{'OVERALL':10s}{np.mean(A)*100:9.1f}%{np.mean(B)*100:9.1f}%")
print(f"{'n':10s}{len(A):10d}{len(B):10d}")
print(f"\nreal-photograph ceiling for this judge: 61%")
json.dump({"ours":{e:[bool(x) for x in ours[e]] for e in E},
           "edtalk":{e:[bool(x) for x in ed[e]] for e in E}}, open("edtalk_comparison.json","w"))
