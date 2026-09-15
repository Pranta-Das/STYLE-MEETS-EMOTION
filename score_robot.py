#!/usr/bin/env python3
"""Metrics for the robotic-head run: 3 mechanical heads x 7 emotions.

Style reference identified as marion-volpe-boy1 by matching the output's CIELAB
statistics against every candidate on disk (7.14 vs 11.13 for the runner-up),
corroborated by the reenactment output directories from the same session.

Emotion metrics follow the paper's protocol exactly: py-feat on identical MTCNN
face crops, displacement measured against the run's own neutral render.
Stylisation metrics are measured against a freshly rendered UNSTYLED baseline of
the same heads, as everywhere else in this project.
"""
import os, glob, json, warnings, importlib.util
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
import lpips as lpips_lib
from facenet_pytorch import MTCNN
from feat import Detector
sp=importlib.util.spec_from_file_location("_ct","lam/stylization/color_transfer.py")
ct=importlib.util.module_from_spec(sp); sp.loader.exec_module(ct)
DEV="cuda"; IDS=["113","115","116"]
E=["anger","disgust","fear","happy","neutral","sad","surprise"]
STYLE="style-image/011_577_623_4k_marion-volpe-marion-volpe-boy1.jpg"
UPPER=['AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU43']
LOWER=['AU10','AU11','AU12','AU14','AU15','AU17','AU20','AU23','AU24','AU25','AU26','AU28']
det=Detector(au_model="xgb", emotion_model="resmasknet", device="cpu")
mt=MTCNN(keep_all=False, device=DEV); loss_fn=lpips_lib.LPIPS(net='alex').to(DEV)
from transformers import CLIPModel, CLIPImageProcessor
clip=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
proc=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
FM={"anger":"anger","disgust":"disgust","fear":"fear","happiness":"happy","happy":"happy",
    "neutral":"neutral","sadness":"sad","sad":"sad","surprise":"surprise"}
CC="/tmp/robotcrop"; os.makedirs(CC,exist_ok=True)
def crop(p,tag):
    o=f"{CC}/{tag}.png"
    if not os.path.exists(o):
        im=Image.open(p).convert("RGB"); b,_=mt.detect(im)
        if b is None:
            w,h=im.size; s=int(min(w,h)*0.6); box=((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2)
        else:
            x1,y1,x2,y2=b[0]; w_,h_=x2-x1,y2-y1; m=0.35
            box=(max(0,x1-m*w_),max(0,y1-m*h_),min(im.size[0],x2+m*w_),min(im.size[1],y2+m*h_))
        im.crop(box).resize((320,320),Image.LANCZOS).save(o)
    return o
def facelab(p):
    a=np.asarray(Image.open(p).convert("RGB").resize((224,224)),dtype=np.uint8)
    L=cv2.cvtColor(a,cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1,3)
    return np.concatenate([L.mean(0),L.std(0)])
def lab_full(p):
    m,s=ct._lab_stats(np.asarray(Image.open(p).convert("RGB"),dtype=np.uint8),None,True)
    return np.concatenate([m,s])
def arr(p,s=256): return np.asarray(Image.open(p).convert("RGB").resize((s,s),Image.LANCZOS),dtype=np.uint8)
def t(a): return torch.from_numpy(a).permute(2,0,1)[None].float().to(DEV)/127.5-1.0
@torch.no_grad()
def clipe(p):
    px=proc(images=Image.open(p).convert("RGB"),return_tensors="pt")["pixel_values"].to(DEV)
    e=clip.get_image_features(pixel_values=px)[0]; return (e/e.norm()).cpu().numpy()
def feat(p):
    try:
        r=det.detect_image(p); au=r.aus.iloc[0]; em=r.emotions.iloc[0]
        return au, FM.get(str(max(em.index,key=lambda c:em[c])).lower())
    except Exception: return None,None
# real-photo AU reference
ref={}
for e in E:
    aus=[a for a,_ in (feat(f) for f in sorted(glob.glob(f"data/emotion_refs/images/{e}/ravdess_*.jpg"))[:14]) if a is not None]
    if aus: ref[e]=sum(aus)/len(aus)

styp=lambda e,i: f"my_check/{e}/{i}/stylized_preview.png"
unsp=lambda e,i: f"exps/images/robot_unstyled/{e}/{i}/stylized_preview.png"
scrop=crop(STYLE,"style"); slab=facelab(scrop); se=clipe(STYLE)

EM={k:[] for k in ["acc","psnr","ssim","lpips","auel","aueu"]}
ST={k:[] for k in ["adopt","psnr","ssim","lpips","clip"]}
for e in E:
    for i in IDS:
        ps,pu=styp(e,i),unsp(e,i)
        if not os.path.exists(ps): continue
        c=crop(ps,f"s_{e}_{i}"); a,pred=feat(c)
        if a is not None:
            EM["acc"].append(pred==e)
            if e in ref:
                EM["aueu"].append(float(np.abs(a[UPPER].values-ref[e][UPPER].values).mean()))
                EM["auel"].append(float(np.abs(a[LOWER].values-ref[e][LOWER].values).mean()))
        pn=styp("neutral",i)
        if os.path.exists(pn):
            A,B=arr(pn),arr(ps)
            mse=float(((A.astype(np.float32)-B.astype(np.float32))**2).mean())
            EM["psnr"].append(10*np.log10(255.0**2/max(mse,1e-8)))
            EM["ssim"].append(ssim_fn(A,B,channel_axis=2,data_range=255))
            with torch.no_grad(): EM["lpips"].append(float(loss_fn(t(A),t(B)).item()))
        if os.path.exists(pu):
            du=np.linalg.norm(facelab(crop(pu,f"u_{e}_{i}"))-slab)
            ds=np.linalg.norm(facelab(c)-slab)
            ST["adopt"].append(1-ds/(du+1e-9))
            A,B=arr(pu),arr(ps)
            mse=float(((A.astype(np.float32)-B.astype(np.float32))**2).mean())
            ST["psnr"].append(10*np.log10(255.0**2/max(mse,1e-8)))
            ST["ssim"].append(ssim_fn(A,B,channel_axis=2,data_range=255))
            with torch.no_grad(): ST["lpips"].append(float(loss_fn(t(A),t(B)).item()))
            ST["clip"].append(float(clipe(ps)@se))
    print(f"  {e} done",flush=True)
f=lambda d,k: np.mean(d[k]) if d[k] else float('nan')
print(f"\nROBOTIC HEADS -- 3 mechanical heads x 7 emotions")
print(f"style reference: {os.path.basename(STYLE)}\n")
print(f"EMOTION   n={len(EM['acc'])}")
print(f"  EmoAcc                 {f(EM,'acc')*100:6.1f}%   (human faces: 58.2%)")
print(f"  AUE-L / AUE-U          {f(EM,'auel'):.3f} / {f(EM,'aueu'):.3f}")
print(f"  displacement PSNR/SSIM/LPIPS  {f(EM,'psnr'):.2f} / {f(EM,'ssim'):.3f} / {f(EM,'lpips'):.3f}")
print(f"\nSTYLISATION   n={len(ST['adopt'])}")
print(f"  colour adoption        {f(ST,'adopt')*100:6.1f}%   (human faces: 50.1%)")
print(f"  LPIPS / SSIM / PSNR    {f(ST,'lpips'):.3f} / {f(ST,'ssim'):.3f} / {f(ST,'psnr'):.2f}")
print(f"  CLIP to style          {f(ST,'clip'):.3f}")
json.dump({"emotion":{k:[float(x) for x in v] for k,v in EM.items()},
           "style":{k:[float(x) for x in v] for k,v in ST.items()}},open("robot_metrics.json","w"))
