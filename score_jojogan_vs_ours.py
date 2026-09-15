#!/usr/bin/env python3
"""JoJoGAN vs ours on the SAME 18 content photos and SAME 7 style references.

Every metric is computed by the same code for both systems, and always against
that system's OWN un-stylised output -- our unstyled render, JoJoGAN's e4e
reconstruction. That is what makes the two columns comparable: each measures
"how far did this system move from its own starting point toward the style",
not an absolute pixel distance between two different renderers.
"""
import os, glob, json, warnings, importlib.util
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
import lpips as lpips_lib
from facenet_pytorch import MTCNN, InceptionResnetV1

sp=importlib.util.spec_from_file_location("_ct","lam/stylization/color_transfer.py")
ct=importlib.util.module_from_spec(sp); sp.loader.exec_module(ct)
DEV="cuda"
E=["anger","disgust","fear","happy","neutral","sad","surprise"]
CONT=[os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob("new_input_image/*"))]
STYLES=sorted(glob.glob("new-style-image/*"))[:7]
TRACK="tracking_output/export"
loss_fn=lpips_lib.LPIPS(net='alex').to(DEV)
from transformers import CLIPModel, CLIPImageProcessor
clip=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
proc=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
mtcnn=MTCNN(image_size=160,margin=20,device=DEV); fnet=InceptionResnetV1(pretrained="vggface2").eval().to(DEV)

def arr(p,s=256): return np.asarray(Image.open(p).convert("RGB").resize((s,s),Image.LANCZOS),dtype=np.uint8)
def t(a): return torch.from_numpy(a).permute(2,0,1)[None].float().to(DEV)/127.5-1.0
def lab(p):
    m,s=ct._lab_stats(np.asarray(Image.open(p).convert("RGB"),dtype=np.uint8),None,True)
    return np.concatenate([m,s])
@torch.no_grad()
def clipe(p):
    px=proc(images=Image.open(p).convert("RGB"),return_tensors="pt")["pixel_values"].to(DEV)
    e=clip.get_image_features(pixel_values=px)[0]; return (e/e.norm()).cpu().numpy()
def fembed(p):
    f=mtcnn(Image.open(p).convert("RGB"))
    if f is None: return None
    with torch.no_grad(): e=fnet(f.unsqueeze(0).to(DEV))[0]
    return (e/e.norm()).cpu().numpy()
def tracked(p):
    n=os.path.splitext(os.path.basename(p))[0]
    i=os.path.join(TRACK,n,"images","00000_00.png")
    return i if os.path.exists(i) else p

def evaluate(name, pair_fn):
    """pair_fn(si, cid) -> (styled_path, own_unstyled_path) or None"""
    R={k:[] for k in ["adopt","psnr","ssim","lpips","clip","idsim"]}
    for si,spath in enumerate(STYLES):
        sl=lab(tracked(spath)); se=clipe(tracked(spath))
        for cid in CONT:
            pr=pair_fn(si,cid)
            if pr is None: continue
            ps,pu=pr
            if not(os.path.exists(ps) and os.path.exists(pu)): continue
            du=np.linalg.norm(lab(pu)-sl); ds=np.linalg.norm(lab(ps)-sl)
            R["adopt"].append(1-ds/(du+1e-9))
            A,B=arr(pu),arr(ps)
            mse=float(((A.astype(np.float32)-B.astype(np.float32))**2).mean())
            R["psnr"].append(10*np.log10(255.0**2/max(mse,1e-8)))
            R["ssim"].append(ssim_fn(A,B,channel_axis=2,data_range=255))
            with torch.no_grad(): R["lpips"].append(float(loss_fn(t(A),t(B)).item()))
            R["clip"].append(float(clipe(ps)@se))
            a,b=fembed(ps),fembed(pu)
            if a is not None and b is not None: R["idsim"].append(float(a@b))
        print(f"  {name} s{si} done",flush=True)
    return R

ours = evaluate("ours", lambda si,cid: (
    f"exps/images/bigrun/styled/s{si}/neutral/{cid}/stylized_preview.png",
    f"exps/images/bigrun/unstyled/neutral/{cid}/stylized_preview.png"))
jojo = evaluate("jojo", lambda si,cid: (
    f"exps/images/jojogan/s{si}/{cid}.png",
    f"exps/images/jojogan/recon/{cid}.png"))

print(f"\n{'metric':34s}{'OURS':>12s}{'JoJoGAN':>12s}")
print("-"*58)
rows=[("colour adoption (gap closure)","adopt",100,"%"),
      ("PSNR vs own unstyled (dB)","psnr",1,""),
      ("SSIM vs own unstyled","ssim",1,""),
      ("LPIPS vs own unstyled","lpips",1,""),
      ("CLIP sim output<->style","clip",1,""),
      ("FaceNet identity vs own unstyled","idsim",1,"")]
for lbl,k,sc,u in rows:
    print(f"{lbl:34s}{np.mean(ours[k])*sc:11.3f}{u}{np.mean(jojo[k])*sc:11.3f}{u}")
print("-"*58)
print(f"{'n (content x style pairs)':34s}{len(ours['adopt']):12d}{len(jojo['adopt']):12d}")
json.dump({"ours":{k:[float(x) for x in v] for k,v in ours.items()},
           "jojogan":{k:[float(x) for x in v] for k,v in jojo.items()}},
          open("jojogan_comparison.json","w"))
