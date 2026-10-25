# 2.5D CNN on LIDC nodule crops, trained on SOFT labels (fraction of readers marking each finding).
# 5-fold patient-grouped cross-fitting x N_SEEDS seeds -> out-of-fold finding probabilities for every nodule,
# for several test views (TTA + degradations). Same CNN trained on malignancy as a black-box baseline.
import os, glob, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision

SMOKE = os.environ.get("SMOKE", "0") == "1"
CNN_PROFILE = os.environ.get("CNN_PROFILE", "resnet18_3slice")
if CNN_PROFILE not in {"resnet18_3slice", "resnet34_5slice"}:
    raise ValueError(f"Unknown CNN_PROFILE: {CNN_PROFILE}")
IN_CH = 3 if CNN_PROFILE == "resnet18_3slice" else 5
N_FOLDS, N_SEEDS, EPOCHS, BATCH, LR, SIZE = 5, (1 if SMOKE else 5), (2 if SMOKE else 25), 64, 3e-4, 96
OUT, dev = "/kaggle/working", "cuda"

# ---- crops + ratings from the lidc-prep kernels
Z = {}
for f in glob.glob("/kaggle/input/**/lidc_crops_*.npz", recursive=True):
    z = np.load(f); Z.update({k: z[k] for k in z.files})
R = pd.read_csv(glob.glob("/kaggle/input/**/lidc_ratings.csv", recursive=True)[0])
B = pd.DataFrame({"nodule": R.nodule, "patient": R.patient, "malignancy": R.malignancy,
    "spiculated": R.spiculation >= 3, "lobulated": R.lobulation >= 3, "ill_defined": R.margin <= 2,
    "sharp_margin": R.margin >= 4, "solid": R.texture == 5, "part_solid": R.texture.between(2, 4),
    "ground_glass": R.texture == 1, "round": R.sphericity >= 4, "irregular_shape": R.sphericity <= 2,
    "benign_calc": R.calcification.isin([1, 2, 3, 5]), "other_calc": R.calcification == 4,
    "fat": R.internalStructure == 3, "large": R.diameter > 15, "small": R.diameter < 6})
CONC = list(B.columns[3:])
g = B.groupby("nodule")
N = g[CONC].mean().astype(float); N["patient"] = g.patient.first(); N["mal"] = g.malignancy.mean()
N = N.loc[[i for i in N.index if i in Z]]
ids = list(N.index)
if SMOKE: ids = ids[:300]; N = N.loc[ids]
soft = N[CONC].values.astype(np.float32)
ymal = np.where(N.mal > 3, 1.0, np.where(N.mal < 3, 0.0, np.nan)).astype(np.float32)
print(len(ids), "nodules,", len(CONC), "findings", flush=True)

def prep(p):  # (h,w,5) HU -> centered 3 or all 5 CT slices in lung window
    selected = p[:, :, 1:4] if IN_CH == 3 else p[:, :, :5]
    v = np.clip((selected.astype(np.float32) + 1000) / 1400, 0, 1)
    t = torch.from_numpy(v).permute(2, 0, 1)[None]
    return F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
X = torch.stack([prep(Z[i]) for i in ids])                      # (n,IN_CH,S,S) in [0,1]
if IN_CH == 3:
    MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
else:
    MEAN = torch.full((IN_CH, 1, 1), 0.449)
    STD = torch.full((IN_CH, 1, 1), 0.226)

def view(x, name, gen=torch.Generator().manual_seed(0)):
    if name == "tta0": return x
    if name == "tta1": return torch.rot90(x, 1, (2, 3))
    if name == "tta2": return torch.flip(x, (3,))
    if name == "tta3": return torch.flip(x, (2,))
    if name.startswith("blur"):
        s = float(name[4:]); k = int(4 * s) | 1; ax = torch.arange(k) - k // 2
        w = torch.exp(-ax.float() ** 2 / (2 * s * s)); w = w / w.sum()
        x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), w.view(1, 1, 1, k).repeat(IN_CH, 1, 1, 1), groups=IN_CH)
        return F.conv2d(x, w.view(1, 1, k, 1).repeat(IN_CH, 1, 1, 1), groups=IN_CH)
    if name.startswith("noise"):   # std in HU, window width 1400 HU
        return (x + torch.randn(x.shape, generator=gen) * float(name[5:]) / 1400).clamp(0, 1)
VIEWS = ["tta0", "tta1", "tta2", "tta3", "blur1", "blur2", "blur4", "noise100", "noise200"]

def augment(x):
    if np.random.rand() < 0.5: x = torch.flip(x, (3,))
    if np.random.rand() < 0.5: x = torch.flip(x, (2,))
    return torch.rot90(x, np.random.randint(4), (2, 3))

def model(n_out):
    if IN_CH == 3:
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    else:
        m = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        w = m.conv1.weight.data.mean(1, keepdim=True).repeat(1, IN_CH, 1, 1) * 3 / IN_CH
        m.conv1 = nn.Conv2d(IN_CH, 64, 7, 2, 3, bias=False)
        m.conv1.weight.data = w
    m.fc = nn.Linear(512, n_out)
    return m.to(dev)

def train_predict(tr, te, target, n_out, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = model(n_out); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(tr) / BATCH)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
    T = torch.from_numpy(target)
    for ep in range(EPOCHS):
        m.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            xb = augment((X[b] - MEAN) / STD).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(m(xb).float(), T[b].view(len(b), -1).to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval(); out = {}
    with torch.no_grad():
        for v in VIEWS:
            xv = (view(X[te], v) - MEAN) / STD
            out[v] = torch.cat([torch.sigmoid(m(xv[i:i + 256].to(dev)).float()).cpu() for i in range(0, len(te), 256)]).numpy()
    return out

# patient-grouped folds
FOLD_SEED = int(os.environ.get("PART", "0"))      # different fold assignment per run (step 4)
pats = np.array(sorted(N.patient.unique())); np.random.default_rng(FOLD_SEED).shuffle(pats)
fold_of = {p: i % N_FOLDS for i, p in enumerate(pats)}
fold = N.patient.map(fold_of).values
P = {v: np.full((len(ids), len(CONC), N_SEEDS), np.nan, np.float32) for v in VIEWS}
M = {v: np.full((len(ids), N_SEEDS), np.nan, np.float32) for v in VIEWS}
t0 = time.time()
for k in range(N_FOLDS):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    trm = tr[~np.isnan(ymal[tr])]
    for s in range(N_SEEDS):
        o = train_predict(tr, te, soft, len(CONC), 100 * k + s)
        for v in VIEWS: P[v][te, :, s] = o[v]
        o = train_predict(trm, te, np.nan_to_num(ymal), 1, 100 * k + s + 50)
        for v in VIEWS: M[v][te, s] = o[v][:, 0]
    # quick progress: soft-label fit on this fold
    pm = P["tta0"][te].mean(-1); maj = soft[te] >= 0.5
    aucs = []
    from sklearn.metrics import roc_auc_score
    for j in range(len(CONC)):
        if 0 < maj[:, j].sum() < len(te): aucs.append(roc_auc_score(maj[:, j], pm[:, j]))
    print(f"fold {k}: mean finding AUROC {np.mean(aucs):.3f}, {time.time()-t0:.0f}s", flush=True)

np.savez_compressed(f"{OUT}/lidc_cnn.npz", ids=np.array(ids), concepts=np.array(CONC), fold=fold,
                    **{f"P_{v}": P[v] for v in VIEWS}, **{f"M_{v}": M[v] for v in VIEWS})
print("saved; total", round(time.time() - t0), "s")
