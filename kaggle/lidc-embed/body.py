# Embed LIDC nodule crops (2.5D: slices -1,0,+1 as RGB, lung window) with BiomedCLIP and DINOv3.
# Views: 4 test-time augmentations (perceptual spread per nodule) + blur / noise degradations.
import os, glob, subprocess, sys
import numpy as np, torch
from PIL import Image, ImageFilter
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "open_clip_torch", "ftfy"], check=True, timeout=600)
OUT = "/kaggle/working"
fs = glob.glob("/kaggle/input/**/lidc_crops_*.npz", recursive=True)
Z = {}
for f in fs:
    z = np.load(f); Z.update({k: z[k] for k in z.files})
ids = sorted(Z)
print(len(ids), "crops from", fs, flush=True)

def to_img(p, noise=0.0, rng=np.random.default_rng(0)):
    v = p[:, :, 1:4].astype(np.float32)
    if noise: v = v + rng.normal(0, noise, v.shape)
    v = np.clip((v + 1000) / 1400, 0, 1)                 # lung window [-1000, 400] HU
    return Image.fromarray((v * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)

VIEWS = {"tta0": lambda im: im, "tta1": lambda im: im.transpose(Image.ROTATE_90),
         "tta2": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT), "tta3": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
         "blur2": lambda im: im.filter(ImageFilter.GaussianBlur(2)), "blur4": lambda im: im.filter(ImageFilter.GaussianBlur(4)),
         "blur8": lambda im: im.filter(ImageFilter.GaussianBlur(8))}
NOISE = {"noise100": 100.0, "noise200": 200.0}           # HU std

def run(name, enc, B=64):
    out = {"ids": np.array(ids)}
    for vname in list(VIEWS) + list(NOISE):
        feats = []
        for i in range(0, len(ids), B):
            if vname in VIEWS: ims = [VIEWS[vname](to_img(Z[k])) for k in ids[i:i + B]]
            else: ims = [to_img(Z[k], NOISE[vname]) for k in ids[i:i + B]]
            feats.append(enc(ims))
        out[vname] = np.concatenate(feats)
    np.savez_compressed(f"{OUT}/lidc_emb_{name}.npz", **out)
    print(name, {k: v.shape for k, v in out.items() if k != "ids"}, flush=True)

dev = "cuda"
norm = lambda x: torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy()
import open_clip
bm, bpre = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
bm = bm.to(dev).eval()
run("biomedclip", torch.no_grad()(lambda ims: norm(bm.encode_image(torch.stack([bpre(x) for x in ims]).to(dev)))))
del bm; torch.cuda.empty_cache()
import timm
dm = timm.create_model("hf-hub:timm/vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).to(dev).eval()
cfg = timm.data.resolve_data_config({}, model=dm); tf = timm.data.create_transform(**cfg)
run("dinov3", torch.no_grad()(lambda ims: norm(dm(torch.stack([tf(x) for x in ims]).to(dev)))))
