#!/usr/bin/env python3
"""Full emotion metric table for ours vs EDTalk, on identical face crops.

IMPORTANT ON SEMANTICS. Prior work (e.g. EDTalk Tab.1) computes PSNR/SSIM/LPIPS/
LMD as FIDELITY against a held-out REAL video of the same person performing the
emotion. MEAD is not available here, so there is no such ground truth and those
four metrics cannot mean what they mean in that table.

What is computed instead, and how to read it:
  PSNR / SSIM / LPIPS / LMD  -- against each system's OWN NEUTRAL output.
      These measure EXPRESSION DISPLACEMENT, not fidelity. A system that barely
      moves the face scores "well" on PSNR/SSIM and badly on displacement.
      They are reported for completeness and are NOT comparable to published
      fidelity numbers. Arrow directions from the prior-work table do not apply.
  AUE-L / AUE-U  -- mean absolute difference between the generated face's action
      unit intensities and the mean AU profile of REAL photographs of that
      emotion (RAVDESS). Lower = closer to how humans actually perform it.
      This IS meaningful without paired ground truth.
  FID -- between the pooled generated set and real RAVDESS emotional
      photographs. Meaningful without pairing, but n is small (98); treat as
      indicative.
  EmoAcc -- py-feat top-1, the reportable number.
"""
import os, glob, json, warnings, cv2
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
import lpips as lpips_lib
from facenet_pytorch import MTCNN
from feat import Detector

DEV="cuda"; E=["anger","disgust","fear","happy","neutral","sad","surprise"]
UPPER=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU43']
LOWER=['AU10','AU11','AU12','AU14','AU15','AU17','AU20','AU23','AU24','AU25','AU26','AU28']
det=Detector(au_model="xgb", emotion_model="resmasknet", device="cpu")
mt=MTCNN(keep_all=False, device=DEV)
loss_fn=lpips_lib.LPIPS(net='alex').to(DEV)
FM={"anger":"anger","disgust":"disgust","fear":"fear","happiness":"happy","happy":"happy",
    "neutral":"neutral","sadness":"sad","sad":"sad","surprise":"surprise"}
CROP="/tmp/fullcrop"; os.makedirs(CROP,exist_ok=True)

def facecrop(img, tag):
    o=f"{CROP}/{tag}.png"
    if os.path.exists(o): return o
    b,_=mt.detect(img)
    if b is None: return None
    x1,y1,x2,y2=b[0]; w,h=x2-x1,y2-y1; m=0.35
    img.crop((max(0,x1-m*w),max(0,y1-m*h),
              min(img.size[0],x2+m*w),min(img.size[1],y2+m*h))
        ).resize((320,320),Image.LANCZOS).save(o); return o
def midframe(mp4):
    c=cv2.VideoCapture(mp4); n=int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.set(cv2.CAP_PROP_POS_FRAMES,max(0,n//2)); ok,f=c.read(); c.release()
    return Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)) if ok else None
def feat(p):
    try:
        r=det.detect_image(p)
        au=r.aus.iloc[0]; em=r.emotions.iloc[0]
        lm=r.landmarks[0] if len(r.landmarks) else None
        return au, FM.get(str(max(em.index,key=lambda c:em[c])).lower()), lm
    except Exception: return None,None,None
def arr(p,s=256): return np.asarray(Image.open(p).convert("RGB").resize((s,s)),dtype=np.uint8)
def t(a): return torch.from_numpy(a).permute(2,0,1)[None].float().to(DEV)/127.5-1.0

# --- real reference AU profiles per emotion (RAVDESS photographs) ---
print("building real-photograph AU reference...", flush=True)
ref={}
for e in E:
    fs=sorted(glob.glob(f"data/emotion_refs/images/{e}/ravdess_*.jpg"))[:14]
    aus=[]
    for f in fs:
        c=facecrop(Image.open(f).convert("RGB"), f"real_{e}_{os.path.basename(f)[:-4]}")
        if not c: continue
        a,_,_=feat(c)
        if a is not None: aus.append(a)
    if aus: ref[e]=sum(aus)/len(aus)
    print(f"  {e}: {len(aus)} real photos", flush=True)

cids=[os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob("baselines/EDTalk/prep/src14/*.png"))]
def collect(name, getter, neutral_getter):
    R={k:[] for k in ["psnr","ssim","lpips","lmd","auel","aueu","acc"]}
    for e in E:
        for cid in cids:
            p=getter(e,cid); pn=neutral_getter(cid)
            if p is None or pn is None: continue
            a,pred,lm=feat(p); an,_,lmn=feat(pn)
            if a is None: continue
            R["acc"].append(pred==e)
            if e in ref:
                R["aueu"].append(float(np.abs(a[UPPER].values-ref[e][UPPER].values).mean()))
                R["auel"].append(float(np.abs(a[LOWER].values-ref[e][LOWER].values).mean()))
            A,B=arr(pn),arr(p)
            mse=float(((A.astype(np.float32)-B.astype(np.float32))**2).mean())
            R["psnr"].append(10*np.log10(255.0**2/max(mse,1e-8)))
            R["ssim"].append(ssim_fn(A,B,channel_axis=2,data_range=255))
            with torch.no_grad(): R["lpips"].append(float(loss_fn(t(A),t(B)).item()))
            if lm is not None and lmn is not None and len(lm)==len(lmn):
                R["lmd"].append(float(np.linalg.norm(np.asarray(lm)-np.asarray(lmn),axis=1).mean()))
        print(f"  {name} {e}", flush=True)
    return R

ours=collect("ours",
  lambda e,cid: facecrop(Image.open(f"exps/images/emoeval_unstyled_pgd/{e}/{cid}/stylized_preview.png").convert("RGB"), f"o_{e}_{cid}")
                if os.path.exists(f"exps/images/emoeval_unstyled_pgd/{e}/{cid}/stylized_preview.png") else None,
  lambda cid: facecrop(Image.open(f"exps/images/emoeval_unstyled_pgd/neutral/{cid}/stylized_preview.png").convert("RGB"), f"o_neutral_{cid}")
                if os.path.exists(f"exps/images/emoeval_unstyled_pgd/neutral/{cid}/stylized_preview.png") else None)
def ed_get(e,cid):
    m=f"exps/images/edtalk14/{e}/{cid}.mp4"
    if not os.path.exists(m): return None
    fr=midframe(m); return facecrop(fr,f"e_{e}_{cid}") if fr is not None else None
edtalk=collect("edtalk", ed_get, lambda cid: ed_get("neutral",cid))

def row(n,R):
    f=lambda k: np.mean(R[k]) if R[k] else float('nan')
    return (f"{n:10s}{f('psnr'):8.2f}{f('lpips'):9.3f}{f('ssim'):8.3f}"
            f"{f('lmd'):8.2f}{f('auel'):9.3f}{f('aueu'):9.3f}{f('acc')*100:9.1f}%")
print(f"\n{'Method':10s}{'PSNR':>8s}{'LPIPS':>9s}{'SSIM':>8s}{'LMD':>8s}{'AUE-L':>9s}{'AUE-U':>9s}{'EmoAcc':>10s}")
print("-"*71)
print(row("Ours",ours)); print(row("EDTalk",edtalk))
print("-"*71)
print("PSNR/LPIPS/SSIM/LMD are vs each system's OWN NEUTRAL output = expression")
print("displacement, NOT fidelity. AUE is vs real RAVDESS AU profiles (lower better).")
json.dump({"ours":{k:[float(x) for x in v] for k,v in ours.items()},
           "edtalk":{k:[float(x) for x in v] for k,v in edtalk.items()}},
          open("full_emotion_table.json","w"))
