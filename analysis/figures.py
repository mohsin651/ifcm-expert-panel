# Figures for the report. Quantities are loaded from analysis outputs, never transcribed here.
import json, re, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "report/figs"; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a94", "#222222", "#6b6b66"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})

def read_text(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()

def match_float(pattern, source, path):
    result = re.search(pattern, source, re.M)
    if result is None:
        raise ValueError(f"Could not parse {pattern!r} from {path}; rerun its analysis")
    return tuple(float(value) for value in result.groups())

def lidc_auc(encoder):
    path = f"results_lidc/log_{encoder}.txt"
    log = read_text(path)
    names = ["true findings + LLM panel edges", "predicted findings + LLM panel edges [OURS]",
             "true findings + learned edges", "predicted findings + learned edges",
             "black-box CNN trained on malignancy" if encoder == "cnn" else "black-box probe on embedding"]
    return {name: match_float(r"^\s*" + re.escape(name) + r"\s+([0-9.]+) / [0-9.]+\s*$", log, path)[0]
            for name in names}

def attribution_lidc(encoder):
    path = f"results_lidc/log_{encoder}.txt"; log = read_text(path)
    ours = match_float(r"^\s*perc_bern\s+- know_flip\s+AUROC ([0-9.]+)\s+95% CI \[([0-9.]+), ([0-9.]+)\]", log, path)
    unc = match_float(r"^\s*uncertainty \(1-conf\) alone\s+AUROC ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", log, path)
    return ours, unc

def attribution_derm(encoder):
    path = f"results/why_{encoder}_orig+comp+choice.txt"; log = read_text(path)
    point = match_float(r"^\s*perc_bern\s+- know_flip\s+: ([0-9.]+)", log, path)[0]
    lo, hi = match_float(r"^A\) attribution AUROC 95% CI: \[([0-9.]+), ([0-9.]+)\]", log, path)
    unc = match_float(r"^uncertainty \(1-conf\) attribution AUROC ([0-9.]+) 95% CI \[([0-9.]+), ([0-9.]+)\]", log, path)
    return (point, lo, hi), unc

# ---------- Fig: Derm7pt melanoma edges vs 7-point checklist
W = pd.read_csv("results/edges_mid.csv", index_col=0); H = pd.read_csv("results/edges_hesitancy.csv", index_col=0)
names = {"atypical_pn": "atypical pigment network", "typical_pn": "typical pigment network", "irregular_streaks": "irregular streaks",
         "regular_streaks": "regular streaks", "irregular_pigm": "irregular pigmentation", "regular_pigm": "regular pigmentation",
         "regression": "regression structures", "irregular_dg": "irregular dots/globules", "regular_dg": "regular dots/globules",
         "bwv": "blue-whitish veil", "arborizing": "arborizing vessels", "atypical_vessels": "atypical vessels",
         "comma": "comma vessels", "hairpin": "hairpin vessels", "vessels_regr": "vessels in regression"}
pts = {"atypical_pn": 2, "bwv": 2, "atypical_vessels": 2, "irregular_streaks": 1, "irregular_pigm": 1, "irregular_dg": 1, "regression": 1}
d = pd.DataFrame({"w": W.MEL, "h": H.MEL}); d["pts"] = [pts.get(c, 0) for c in d.index]
d = d.sort_values(["pts", "w"]); y = np.arange(len(d))
fig, ax = plt.subplots(figsize=(5.6, 3.8))
col = [BLUE if p == 2 else ("#86b6ef" if p == 1 else GREY) for p in d.pts]
lo = np.clip(d.w - d.h, -1, 1); hi = np.clip(d.w + d.h, -1, 1)
ax.hlines(y, lo, hi, color="#c9c9c3", lw=4, zorder=1)
ax.scatter(d.w, y, c=col, s=34, zorder=3, edgecolor="white", linewidth=0.8)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(y, [names[c] + ("  (+2)" if p == 2 else "  (+1)" if p == 1 else "") for c, p in zip(d.index, d.pts)])
ax.set_xlim(-1, 1); ax.set_xlabel("melanoma edge weight (grey bar: hesitancy interval)")
ax.scatter([], [], c=BLUE, s=30, label="major criterion (+2)"); ax.scatter([], [], c="#86b6ef", s=30, label="minor criterion (+1)")
ax.scatter([], [], c=GREY, s=30, label="not in checklist")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=7, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.2)
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(f"{OUT}/derm_edges.pdf"); plt.close()

# ---------- Fig: LIDC malignancy edges vs literature direction
txt = read_text("results_lidc/log_cnn.txt")
rows = re.findall(r"^(\w+)\s+(-?\d)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)$", txt, re.M)
L = pd.DataFrame(rows, columns=["c", "guide", "phi", "w", "h"]).set_index("c").astype(float)
lnames = {"spiculated": "spiculated margin", "lobulated": "lobulated contour", "ill_defined": "poorly defined margin",
          "sharp_margin": "sharp margin", "solid": "solid", "part_solid": "part-solid", "ground_glass": "ground-glass",
          "round": "round shape", "irregular_shape": "irregular shape", "benign_calc": "benign calcification",
          "other_calc": "eccentric calcification", "fat": "fat density", "large": "diameter > 15 mm", "small": "diameter < 6 mm"}
L = L.sort_values(["guide", "w"]); y = np.arange(len(L))
fig, ax = plt.subplots(figsize=(5.6, 3.6))
col = [ORANGE if g > 0 else (BLUE if g < 0 else GREY) for g in L.guide]
ax.hlines(y, np.clip(L.w - L.h, -1, 1), np.clip(L.w + L.h, -1, 1), color="#c9c9c3", lw=4, zorder=1)
ax.scatter(L.w, y, c=col, s=34, zorder=3, edgecolor="white", linewidth=0.8)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(y, [lnames[c] for c in L.index]); ax.set_xlim(-1, 1)
ax.set_xlabel("malignancy edge weight (grey bar: hesitancy interval)")
ax.scatter([], [], c=ORANGE, s=30, label="literature: raises risk"); ax.scatter([], [], c=BLUE, s=30, label="literature: lowers risk")
ax.scatter([], [], c=GREY, s=30, label="literature: unclear")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=7, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.2)
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(f"{OUT}/lidc_edges.pdf"); plt.close()

# ---------- Fig: diagnosis performance (test AUROC)
methods = ["7-point checklist rule", "LLM-panel edges (ours)", "edges learned from labels", "black-box image probe"]
with open("results/report_monet_orig+comp+choice.json", encoding="utf-8") as handle:
    dm = json.load(handle)["diagnosis"]
def da(name): return dm[name]["mel_auc"]
derm = {"true": [da("oracle concepts + 7-point guideline (MEL only)"), da("oracle concepts + LLM panel edges"),
                 da("oracle concepts + learned edges (CBM)"), np.nan],
        "pred": [da("predicted concepts + 7-point guideline (MEL only)"), da("predicted concepts + LLM panel edges [OURS]"),
                 da("predicted concepts + learned edges (CBM)"), da("blackbox_probe")], "cnn": [np.nan] * 4}
lf, lc = lidc_auc("biomedclip"), lidc_auc("cnn")
lidc = {"true": [np.nan, lf["true findings + LLM panel edges"], lf["true findings + learned edges"], np.nan],
        "pred": [np.nan, lf["predicted findings + LLM panel edges [OURS]"],
                 lf["predicted findings + learned edges"], lf["black-box probe on embedding"]],
        "cnn": [np.nan, lc["predicted findings + LLM panel edges [OURS]"],
                lc["predicted findings + learned edges"], lc["black-box CNN trained on malignancy"]]}
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
for ax, (title, D) in zip(axs, [("Derm7pt: melanoma", derm), ("LIDC-IDRI: malignancy", lidc)]):
    y = np.arange(len(methods))[::-1]
    for k, (lab, c, mk) in {"true": ("true findings", ORANGE, "D"), "pred": ("predicted, frozen encoder", BLUE, "o"),
                            "cnn": ("predicted, soft-label CNN", AQUA, "s")}.items():
        off = {"true": 0.18, "pred": 0.0, "cnn": -0.18}[k]
        ax.scatter(D[k], y + off, c=c, marker=mk, s=36, label=lab, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, methods); ax.set_xlim(0.65, 0.95); ax.set_title(title, fontsize=8.5, color=INK)
    ax.set_xlabel("test AUROC"); ax.grid(axis="y", visible=False)
fig.legend(*axs[1].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5, handletextpad=0.2)
plt.tight_layout(rect=(0, 0.1, 1, 1)); plt.savefig(f"{OUT}/diagnosis.pdf"); plt.close()

# ---------- Fig: error-type attribution (AUROC for "this error is a perception error", 95% bootstrap CI)
A = [("Derm7pt, MONET", *attribution_derm("monet")),
     ("Derm7pt, BiomedCLIP", *attribution_derm("biomedclip")),
     ("LIDC, DINOv3", *attribution_lidc("dinov3")),
     ("LIDC, BiomedCLIP", *attribution_lidc("biomedclip")),
     ("LIDC, soft-label CNN", *attribution_lidc("cnn"))]
fig, ax = plt.subplots(figsize=(4.8, 2.9))
for i, (lab, ours, unc) in enumerate(A):
    yv = len(A) - 1 - i
    ax.errorbar(ours[0], yv + 0.12, xerr=[[ours[0] - ours[1]], [ours[2] - ours[0]]], fmt="o", color=BLUE, ms=5, capsize=0, lw=1.6,
                label="perceptual minus knowledge hesitancy (ours)" if i == 0 else None)
    ax.errorbar(unc[0], yv - 0.12, xerr=[[unc[0] - unc[1]], [unc[2] - unc[0]]], fmt="s", color=ORANGE, ms=5, capsize=0, lw=1.6,
                label="plain uncertainty (1 - confidence)" if i == 0 else None)
ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
ax.set_yticks(range(len(A))[::-1], [a[0] for a in A]); ax.set_xlim(0.35, 0.9)
ax.set_xlabel("AUROC for telling perception errors from knowledge errors"); ax.grid(axis="y", visible=False)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{OUT}/attribution.pdf"); plt.close()

# ---------- Fig: soft-label CNN under blur (results_lidc/log_cnn.txt, degradation sweep)
cnn_log = read_text("results_lidc/log_cnn.txt")
blur, hp, hk, fa = [], [], [], []
for sigma in [0, 1, 2, 4]:
    view = "tta0" if sigma == 0 else f"blur{sigma}"
    p, k, auc = match_float(r"^\s*" + view + r"\s+perc_bern ([0-9.]+)\s+know_flip ([0-9.]+)\s+concept AUROC ([0-9.]+)", cnn_log, "results_lidc/log_cnn.txt")
    blur.append(sigma); hp.append(p); hk.append(k); fa.append(auc)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.2))
axs[0].plot(blur, hp, "-o", color=BLUE, lw=2, ms=5); axs[0].plot(blur, hk, "-s", color=ORANGE, lw=2, ms=5)
axs[0].text(4.3, hp[-1], "perceptual", color=INK, va="center", fontsize=7.5); axs[0].text(4.3, hk[-1], "knowledge", color=INK, va="center", fontsize=7.5)
axs[0].set_xlim(-0.2, 5.9); axs[0].set_ylabel("mean hesitancy (test)"); axs[0].set_xlabel("blur sigma (px)")
axs[1].plot(blur, fa, "-o", color=GREY, lw=2, ms=5); axs[1].set_ylabel("finding AUROC"); axs[1].set_xlabel("blur sigma (px)")
for a in axs: a.set_xticks(blur)
plt.tight_layout(); plt.savefig(f"{OUT}/cnn_blur.pdf"); plt.close()

# ---------- Fig: example LIDC nodules with reader ratings
Z = np.load(os.environ.get("CROPS", "results_lidc/lidc_crops_3.npz"))
R = pd.read_csv("results_lidc/lidc_ratings.csv")
g = R.groupby("nodule"); S = pd.DataFrame({"n": g.size(), "mal": g.malignancy.apply(list), "spic": g.spiculation.apply(list)})
S = S[(S.n == 4) & S.index.isin(Z.files)]
S["mm"] = S.mal.apply(np.mean); S["ms"] = S.mal.apply(np.std)
pick = list(S[S.ms < 0.5].sort_values("mm").index[[0, len(S[S.ms < 0.5]) // 2]]) + \
       list(S[S.ms < 0.5].sort_values("mm").index[[-1]]) + list(S.sort_values("ms").index[-3:])
fig, axs = plt.subplots(1, 6, figsize=(6.6, 1.65))
for ax, k in zip(axs, pick):
    ax.imshow(np.clip((Z[k][:, :, 2] + 1000) / 1400, 0, 1), cmap="gray"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("malignancy " + ",".join(str(v) for v in S.mal[k]) + "\nspiculation " + ",".join(str(v) for v in S.spic[k]), fontsize=6.5, color=INK)
plt.tight_layout(); plt.savefig(f"{OUT}/lidc_examples.pdf"); plt.close()
print("figures written to", OUT)
