#!/usr/bin/env python3
"""Four-way stylisation comparison on the SAME 18 content x 7 style pairs.

Ours, JoJoGAN, StyTR-2 and AdaAttN, every metric by the same code. Fidelity is
always against that system's own un-stylised output:
  ours     -> our unstyled render
  JoJoGAN  -> its e4e reconstruction
  StyTR-2  -> the content photograph (it is a pure 2D transfer with no
              reconstruction stage, so the photo IS its unstylised output)
  AdaAttN  -> same reasoning as StyTR-2
Colour adoption is measured on MTCNN face crops for all four, so background and
framing differences cannot bias it.
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
import lpips as lpips_lib
from facenet_pytorch import MTCNN
DEV="cuda"
E=["anger","disgust","fear","happy","neutral","sad","surprise"]
CONT=[os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob("new_input_image/*"))]
CPATH={os.path.splitext(os.path.basename(p))[0]: p for p in sorted(glob.glob("new_input_image/*"))}
STYLES=sorted(glob.glob("new-style-image/*"))[:7]
TRACK="tracking_output/export"
loss_fn=lpips_lib.LPIPS(net='alex').to(DEV)
mt=MTCNN(keep_all=False, device=DEV)
from transformers import CLIPModel, CLIPImageProcessor
clip=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
proc=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
CC="/tmp/style4crop"; os.makedirs(CC,exist_ok=True)
def facelab(p,tag):
    o=f"{CC}/{tag}.png"
    if not os.path.exists(o):
        im=Image.open(p).convert("RGB"); b,_=mt.detect(im)
        if b is None:
            w,h=im.size; s=int(min(w,h)*0.6); box=((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2)
        else:
            x1,y1,x2,y2=b[0]; w_,h_=x2-x1,y2-y1; m=0.35
            box=(max(0,x1-m*w_),max(0,y1-m*h_),min(im.size[0],x2+m*w_),min(im.size[1],y2+m*h_))
        im.crop(box).resize((224,224),Image.LANCZOS).save(o)
    a=np.asarray(Image.open(o).convert("RGB"),dtype=np.uint8)
    L=cv2.cvtColor(a,cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1,3)
    return np.concatenate([L.mean(0),L.std(0)])
def arr(p,s=256): return np.asarray(Image.open(p).convert("RGB").resize((s,s),Image.LANCZOS),dtype=np.uint8)
def t(a): return torch.from_numpy(a).permute(2,0,1)[None].float().to(DEV)/127.5-1.0
@torch.no_grad()
def clipe(p):
    px=proc(images=Image.open(p).convert("RGB"),return_tensors="pt")["pixel_values"].to(DEV)
    e=clip.get_image_features(pixel_values=px)[0]; return (e/e.norm()).cpu().numpy()
def tracked(p):
    n=os.path.splitext(os.path.basename(p))[0]
    i=os.path.join(TRACK,n,"images","00000_00.png")
    return i if os.path.exists(i) else p

SYS={
 "Ours":      (lambda si,cid: f"exps/images/bigrun/styled/s{si}/neutral/{cid}/stylized_preview.png",
               lambda si,cid: f"exps/images/bigrun/unstyled/neutral/{cid}/stylized_preview.png"),
 "JoJoGAN":   (lambda si,cid: f"exps/images/jojogan/s{si}/{cid}.png",
               lambda si,cid: f"exps/images/jojogan/recon/{cid}.png"),
 "StyTR-2":   (lambda si,cid: f"exps/images/stytr2/s{si}/{cid}.jpg",
               lambda si,cid: CPATH[cid]),
 "AdaAttN":   (lambda si,cid: f"exps/images/adaattn/s{si}/{cid}.png",
               lambda si,cid: CPATH[cid]),
}
res={}
for name,(gs,gu) in SYS.items():
    R={k:[] for k in ["adopt","psnr","ssim","lpips","clip"]}
    for si,sp in enumerate(STYLES):
        sl=facelab(tracked(sp),f"style_{si}"); se=clipe(tracked(sp))
        for cid in CONT:
            ps,pu=gs(si,cid),gu(si,cid)
            if not(os.path.exists(ps) and os.path.exists(pu)): continue
            du=np.linalg.norm(facelab(pu,f"{name}_u_{si}_{cid}")-sl)
            ds=np.linalg.norm(facelab(ps,f"{name}_s_{si}_{cid}")-sl)
            R["adopt"].append(1-ds/(du+1e-9))
            A,B=arr(pu),arr(ps)
            mse=float(((A.astype(np.float32)-B.astype(np.float32))**2).mean())
            R["psnr"].append(10*np.log10(255.0**2/max(mse,1e-8)))
            R["ssim"].append(ssim_fn(A,B,channel_axis=2,data_range=255))
            with torch.no_grad(): R["lpips"].append(float(loss_fn(t(A),t(B)).item()))
            R["clip"].append(float(clipe(ps)@se))
    res[name]=R; print(f"  {name}: n={len(R['adopt'])}",flush=True)
print(f"\n{'Method':10s}{'LPIPS':>9s}{'SSIM':>8s}{'PSNR':>8s}{'CLIP':>8s}{'colour adopt':>14s}{'n':>6s}")
print("-"*63)
for n,R in res.items():
    f=lambda k: np.mean(R[k]) if R[k] else float('nan')
    print(f"{n:10s}{f('lpips'):9.3f}{f('ssim'):8.3f}{f('psnr'):8.2f}{f('clip'):8.3f}{f('adopt')*100:13.1f}%{len(R['adopt']):6d}")
json.dump({n:{k:[float(x) for x in v] for k,v in R.items()} for n,R in res.items()},
          open("style_4way.json","w"))
