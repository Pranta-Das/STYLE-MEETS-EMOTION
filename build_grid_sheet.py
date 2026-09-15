#!/usr/bin/env python3
"""One contact sheet for the 5 content x 5 style x 7 emotion grid.

Layout: 25 rows (grouped into 5 blocks, one per content face) x 7 emotion
columns, with the content photograph and the style reference repeated in the
left gutter of every row so each render can be compared against both of its
inputs without scrolling away.

Each cell is annotated with what the INDEPENDENT judge (py-feat) reads in it:
a green tick when its top-1 matches the requested emotion, otherwise the label
it actually saw. This is the part that makes the sheet diagnostic rather than
decorative -- it shows exactly where your eye and the classifier disagree,
which is the question that has come up repeatedly. Set JUDGE=0 to skip it.

Usage:
  lam_env/bin/python build_grid_sheet.py [grid_dir] [out.png]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw, ImageFont

GRID = sys.argv[1] if len(sys.argv) > 1 else "exps/images/grid55"
OUT = sys.argv[2] if len(sys.argv) > 2 else "grid_5x5x7_preview.png"
JUDGE = os.environ.get("JUDGE", "1") != "0"

CONTENT = ["assets/sample_input/cluo.jpg", "assets/sample_input/james.png",
           "original-image/00028.jpg", "original-image/00052.jpg",
           "original-image/00062.jpg"]
STYLES = ["assets/sample_input/pop.png",
          "style-image/000_256_505_4k_mark-makovey-p-029_00.png",
          "style-image/001_339_379_4k_irakli-nadar-jinxartstation_00.png",
          "style-image/002_345_899_4k_nikita-orlov-portarait-womanblack_00.png",
          "style-image/003_832_838_4k_hany-abbas-sayed-ragab4.jpg"]
STYLE_NAMES = ["pop art", "grey painterly", "neon blue", "gothic pale", "yellow caric."]
E = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

CELL, GAP, HEAD, PAD = 210, 14, 46, 22
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
font = ImageFont.truetype(F % "", 15)
fontb = ImageFont.truetype(F % "-Bold", 19)
fonts = ImageFont.truetype(F % "", 13)

# Judging 175 images on CPU takes ~8 minutes, and the layout usually needs a
# couple of iterations, so cache the verdicts and reuse them on rebuild.
CACHE = "grid_judge_cache.json"
import json
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
judge = None
if JUDGE:
    from feat import Detector
    det = Detector(au_model="xgb", emotion_model="resmasknet", device="cpu")
    FM = {"anger": "anger", "disgust": "disgust", "fear": "fear", "happiness": "happy",
          "happy": "happy", "neutral": "neutral", "sadness": "sad", "sad": "sad",
          "surprise": "surprise"}
    def judge(path):
        if path in cache:
            return cache[path]
        try:
            row = det.detect_image(path).emotions.iloc[0]
            v = FM.get(str(max(row.index, key=lambda c: row[c])).lower(), "?")
        except Exception:
            v = None
        cache[path] = v
        return v

W = PAD * 2 + 2 * CELL + 7 * CELL
H = PAD * 2 + HEAD + 25 * CELL + 4 * GAP + 5 * 24 + 30   # +30 for the footer line
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)

x0 = PAD + 2 * CELL
for j, e in enumerate(E):
    d.text((x0 + j * CELL + CELL // 2, PAD + HEAD // 2), e.upper(),
           fill="black", font=fontb, anchor="mm")
d.text((PAD + CELL // 2, PAD + HEAD // 2), "content", fill="#666", font=font, anchor="mm")
d.text((PAD + CELL + CELL // 2, PAD + HEAD // 2), "style", fill="#666", font=font, anchor="mm")

def thumb(p, s):
    return Image.open(p).convert("RGB").resize((s, s), Image.LANCZOS)

y = PAD + HEAD
n_ok = n_tot = 0
for ci, cpath in enumerate(CONTENT):
    cid = os.path.splitext(os.path.basename(cpath))[0]
    d.text((PAD, y + 4), f"content {ci}:  {cid}", fill="#111", font=fontb)
    y += 24
    for si, spath in enumerate(STYLES):
        im.paste(thumb(cpath, CELL - 6), (PAD + 3, y + 3))
        im.paste(thumb(spath, CELL - 6), (PAD + CELL + 3, y + 3))
        d.rectangle([PAD + CELL + 3, y + CELL - 22, PAD + 2 * CELL - 3, y + CELL - 3],
                    fill=(0, 0, 0, 200))
        d.text((PAD + CELL + 8, y + CELL - 20), STYLE_NAMES[si], fill="white", font=fonts)
        for j, e in enumerate(E):
            x = x0 + j * CELL
            p = os.path.join(GRID, f"c{ci}_s{si}", e, cid, "stylized_preview.png")
            if not os.path.exists(p):
                d.rectangle([x + 3, y + 3, x + CELL - 3, y + CELL - 3], fill="#eee")
                d.text((x + CELL // 2, y + CELL // 2), "missing", fill="#999",
                       font=font, anchor="mm")
                continue
            im.paste(thumb(p, CELL - 6), (x + 3, y + 3))
            if judge is not None:
                v = judge(p)
                n_tot += 1
                ok = (v == e)
                n_ok += ok
                txt = "OK" if ok else (v or "no face")
                col = (32, 140, 60) if ok else (185, 40, 40)
                w = d.textlength(txt, font=fonts) + 10
                d.rectangle([x + 5, y + 5, x + 5 + w, y + 26], fill=col)
                d.text((x + 10, y + 8), txt, fill="white", font=fonts)
        y += CELL
    y += GAP
    print(f"  content {ci} done", flush=True)

if judge is not None and n_tot:
    d.text((PAD, H - PAD - 16),
           f"green = py-feat (independent judge) top-1 matches the requested emotion; "
           f"red = what it read instead.   {n_ok}/{n_tot} = {n_ok/n_tot*100:.1f}%",
           fill="#333", font=font)
if judge is not None:
    json.dump(cache, open(CACHE, "w"))
im.save(OUT)
print(f"\n{OUT}  {im.size[0]}x{im.size[1]}"
      + (f"   py-feat agreement {n_ok}/{n_tot} = {n_ok/n_tot*100:.1f}%" if n_tot else ""))
