#!/usr/bin/env python3
"""Experiment 1: does halving style geometry strength recover emotion legibility?

Paired per cell (same content, same style, same emotion, same seed) against the
geometry_strength=1.0 renders in the 5x5x7 grid, for the two styles whose own
face geometry is least human -- the gothic doll (s3) and the caricature (s4).
Reuses the grid's cached judgments so only the new arm needs scoring.
"""
import os, json, warnings, itertools
from math import comb
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from feat import Detector

CONTENT = ["cluo", "james", "00028", "00052", "00062"]
E = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SN = {3: "gothic pale", 4: "yellow caric."}
FM = {"anger": "anger", "disgust": "disgust", "fear": "fear", "happiness": "happy",
      "happy": "happy", "neutral": "neutral", "sadness": "sad", "sad": "sad",
      "surprise": "surprise"}
det = Detector(au_model="xgb", emotion_model="resmasknet", device="cpu")
cache = json.load(open("grid_judge_cache.json")) if os.path.exists("grid_judge_cache.json") else {}

def judge(p):
    if p in cache:
        return cache[p]
    try:
        r = det.detect_image(p).emotions.iloc[0]
        v = FM.get(str(max(r.index, key=lambda c: r[c])).lower(), "?")
    except Exception:
        v = None
    cache[p] = v
    return v

def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0: return n10, n01, 1.0
    k = min(n01, n10)
    return n10, n01, min(1.0, 2*sum(comb(n, i) for i in range(k+1))/2**n)

res = {}
for si in (3, 4):
    for tag, root in (("geo1.0", "exps/images/grid55"), ("geo0.5", "exps/images/grid55_geo05")):
        v = []
        for ci, cid in enumerate(CONTENT):
            for e in E:
                p = f"{root}/c{ci}_s{si}/{e}/{cid}/stylized_preview.png"
                v.append(judge(p) == e if os.path.exists(p) else None)
        res[(si, tag)] = v
json.dump(cache, open("grid_judge_cache.json", "w"))

print(f"{'style':16s}{'geo=1.0':>10s}{'geo=0.5':>10s}{'lost':>6s}{'gained':>8s}{'p':>8s}")
print("-" * 58)
A, B = [], []
for si in (3, 4):
    a = [x for x in res[(si, "geo1.0")] if x is not None]
    b = [x for x in res[(si, "geo0.5")] if x is not None]
    if len(a) != len(b):
        print(f"{SN[si]:16s}  incomplete: {len(a)} vs {len(b)}"); continue
    A += a; B += b
    n10, n01, p = mcnemar(a, b)
    print(f"{SN[si]:16s}{np.mean(a)*100:9.1f}%{np.mean(b)*100:9.1f}%{n10:6d}{n01:8d}{p:8.3f}")
if A:
    n10, n01, p = mcnemar(A, B)
    print("-" * 58)
    print(f"{'BOTH':16s}{np.mean(A)*100:9.1f}%{np.mean(B)*100:9.1f}%{n10:6d}{n01:8d}{p:8.3f}")
    print(f"\nn={len(A)} paired cells per arm")
