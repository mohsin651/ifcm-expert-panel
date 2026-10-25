# Perception side: embed every Derm7pt dermoscopy image (clean + degraded) and all concept/diagnosis
# prompts with MONET and BiomedCLIP. Zero-shot scores and probes are computed later on CPU.
import os, glob, subprocess, sys
import numpy as np, pandas as pd, torch
from PIL import Image, ImageFilter

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = "/kaggle/working"
# (name, kind, level): blur radius in px of the original image, noise std on [0,1] pixels
CORRUPTIONS = [("clean", None, 0), ("blur", "blur", 4), ("blur", "blur", 8), ("blur", "blur", 16),
               ("noise", "noise", 0.08), ("noise", "noise", 0.16)]
BATCH = 32

meta_path = glob.glob("/kaggle/input/**/meta/meta.csv", recursive=True)[0]
root = os.path.dirname(os.path.dirname(meta_path))
meta = pd.read_csv(meta_path)
if SMOKE:
    meta = meta.head(40)
paths = [os.path.join(root, "images", p) for p in meta["derm"]]
print(len(paths), "images; root", root, flush=True)

POS = ["This is a dermoscopy image of a skin lesion with {d}.",
       "Dermoscopy showing {d}.",
       "This is skin image of {s}."]
NEG = ["This is a dermoscopy image of a skin lesion without {s}.",
       "Dermoscopy showing no {s}.",
       "This is skin image of a lesion with no {s}."]
DX = ["This is a dermoscopy image of {d}.", "Dermoscopy of a {d}.", "This is skin image of {d}."]

texts, text_keys = [], []
for ck, (d, s) in CONCEPTS.items():
    for i, t in enumerate(POS):
        texts.append(t.format(d=d, s=s)); text_keys.append(f"pos|{ck}|{i}")
    for i, t in enumerate(NEG):
        texts.append(t.format(d=d, s=s)); text_keys.append(f"neg|{ck}|{i}")
for dk, d in DIAGNOSES.items():
    for i, t in enumerate(DX):
        texts.append(t.format(d=d)); text_keys.append(f"dx|{dk}|{i}")

rng = np.random.default_rng(0)
def corrupt(img, kind, level):
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(level))
    if kind == "noise":
        a = np.asarray(img).astype(np.float32) / 255
        a = np.clip(a + rng.normal(0, level, a.shape), 0, 1)
        return Image.fromarray((a * 255).astype(np.uint8))
    return img

def run(name, encode_images, encode_texts):
    out = {"text_keys": np.array(text_keys), "text": encode_texts(texts)}
    for cname, kind, level in CORRUPTIONS:
        feats = []
        for i in range(0, len(paths), BATCH):
            imgs = [corrupt(Image.open(p).convert("RGB"), kind, level) for p in paths[i:i + BATCH]]
            feats.append(encode_images(imgs))
        out[f"img_{cname}_{level}"] = np.concatenate(feats)
        print(name, cname, level, out[f"img_{cname}_{level}"].shape, flush=True)
    np.savez_compressed(f"{OUT}/emb_{name}.npz", **out)

dev = "cuda"
# ---- MONET (HF CLIP format)
from transformers import CLIPModel, CLIPProcessor
m = CLIPModel.from_pretrained("suinleelab/monet").to(dev).eval().half()
proc = CLIPProcessor.from_pretrained("suinleelab/monet")
norm = lambda x: torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy()
@torch.no_grad()
def mi(imgs):
    px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev, torch.float16)
    return norm(m.visual_projection(m.vision_model(pixel_values=px).pooler_output))
@torch.no_grad()
def mt(ts):
    t = proc(text=ts, return_tensors="pt", padding=True, truncation=True).to(dev)
    return norm(m.text_projection(m.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output))
run("monet", mi, mt)
print("monet logit_scale", m.logit_scale.exp().item(), flush=True)
del m; torch.cuda.empty_cache()

# ---- BiomedCLIP (open_clip)
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "open_clip_torch", "ftfy"], check=True, timeout=600)
    import open_clip
    bm, bpre = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    btok = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    bm = bm.to(dev).eval()
    @torch.no_grad()
    def bi(imgs): return norm(bm.encode_image(torch.stack([bpre(x) for x in imgs]).to(dev)))
    @torch.no_grad()
    def bt(ts): return norm(bm.encode_text(btok(ts, context_length=256).to(dev)))
    run("biomedclip", bi, bt)
except Exception as e:
    print("BiomedCLIP FAILED", repr(e))
