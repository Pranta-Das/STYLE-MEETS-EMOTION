#!/usr/bin/env python3
"""Stylisation pixel/perceptual metrics on the 882-render bigrun.

Fidelity metrics (PSNR/SSIM/LPIPS) are measured against the pipeline's OWN
UNSTYLED RENDER of the same content and emotion -- not the content photograph.
A render is a Gaussian reconstruction on a white background; comparing it to a
photograph measures the reconstruction, not the stylisation, and yields ~5 dB
PSNR driven almost entirely by background mismatch.

CLIP similarity is image-image cosine, NOT CLIP directional similarity (which
needs paired text prompts this dataset does not have).
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
import lpips as lpips_lib

DEV = "cuda" if torch.cuda.is_available() else "cpu"
E = ["anger","disgust","fear","happy","neutral","sad","surprise"]
CONTENT = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob("new_input_image/*"))]
STYLES = sorted(glob.glob("new-style-image/*"))[:7]
ROOT = "exps/images/bigrun"; TRACK = "tracking_output/export"

loss_fn = lpips_lib.LPIPS(net='alex').to(DEV)
from transformers import CLIPModel, CLIPImageProcessor
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

def arr(p, size=256):
    return np.asarray(Image.open(p).convert("RGB").resize((size,size), Image.LANCZOS), dtype=np.uint8)
def t(a):
    x = torch.from_numpy(a).permute(2,0,1)[None].float().to(DEV)/127.5 - 1.0
    return x
@torch.no_grad()
def clip_emb(p):
    im = Image.open(p).convert("RGB")
    px = proc(images=im, return_tensors="pt")["pixel_values"].to(DEV)
    e = clip.get_image_features(pixel_values=px)[0]
    return (e/e.norm()).cpu().numpy()

def gram_style_loss(a, b):
    """VGG-free Gram proxy on raw RGB channel statistics -- reported for
    completeness only. Style-loss magnitude is implementation dependent and is
    NOT comparable to another paper's number under the same metric name."""
    x = a.reshape(-1,3).astype(np.float32)/255.; y = b.reshape(-1,3).astype(np.float32)/255.
    gx = (x.T@x)/len(x); gy = (y.T@y)/len(y)
    return float(((gx-gy)**2).mean())

res = {k: [] for k in ["psnr","ssim","lpips","clip_style","clip_content","style_loss"]}
n = 0
for si, spath in enumerate(STYLES):
    sn = os.path.splitext(os.path.basename(spath))[0]
    scrop = os.path.join(TRACK, sn, "images", "00000_00.png")
    scrop = scrop if os.path.exists(scrop) else spath
    s_emb = clip_emb(scrop); s_arr = arr(scrop)
    for cid in CONTENT:
        for e in E:
            ps = f"{ROOT}/styled/s{si}/{e}/{cid}/stylized_preview.png"
            pu = f"{ROOT}/unstyled/{e}/{cid}/stylized_preview.png"
            if not (os.path.exists(ps) and os.path.exists(pu)): continue
            A, B = arr(pu), arr(ps)
            mse = float(((A.astype(np.float32)-B.astype(np.float32))**2).mean())
            res["psnr"].append(10*np.log10(255.0**2/max(mse,1e-8)))
            res["ssim"].append(ssim_fn(A, B, channel_axis=2, data_range=255))
            with torch.no_grad():
                res["lpips"].append(float(loss_fn(t(A), t(B)).item()))
            res["clip_style"].append(float(clip_emb(ps) @ s_emb))
            res["clip_content"].append(float(clip_emb(ps) @ clip_emb(pu)))
            res["style_loss"].append(gram_style_loss(B, s_arr))
            n += 1
    print(f"  s{si} done ({n} pairs)", flush=True)

print(f"\nSTYLISATION METRICS  (n = {n} styled/unstyled pairs, 18 identities x 7 styles x 7 emotions)")
print(f"{'metric':34s}{'mean':>10s}{'sd':>10s}")
print("-"*54)
lab = [("PSNR vs unstyled recon (dB)","psnr"),("SSIM vs unstyled recon","ssim"),
       ("LPIPS vs unstyled recon","lpips"),("CLIP sim  output<->style","clip_style"),
       ("CLIP sim  output<->unstyled","clip_content"),("Gram style loss (RGB proxy)","style_loss")]
for name,k in lab:
    print(f"{name:34s}{np.mean(res[k]):10.4f}{np.std(res[k]):10.4f}")
json.dump({k:[float(x) for x in v] for k,v in res.items()}, open("bigrun_stylization_metrics.json","w"))
