# LIDC-IDRI: does the perceptual / knowledge split of hesitancy match WHY radiologists themselves disagree?
#   perceptual ground truth: reader disagreement on the attributes (spiculation, margin, ...)
#   knowledge ground truth : reader disagreement on malignancy that is NOT explained by attribute disagreement
# usage: python analysis/lidc.py <results_lidc dir> <encoder: biomedclip|dinov3|cnn> [cnn npz] [split seed]
import sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from ifcm.uncertainty import exact_flip_probabilities

RES = sys.argv[1] if len(sys.argv) > 1 else "results_lidc"
ENC = sys.argv[2] if len(sys.argv) > 2 else "biomedclip"
N_BOOT = 30
N_DRAW = 1000           # only for probability-spread sensitivity measure; flip rates are exact
TEST_FRAC = 0.4
CNN_FILE = sys.argv[3] if len(sys.argv) > 3 else f"{RES}/lidc_cnn.npz"
SPLIT_SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
summary = {"enc": ENC, "cnn_file": CNN_FILE, "split_seed": SPLIT_SEED}
DIAGS = ["MAL", "BEN"]
# literature direction for malignancy (Brock model / Lung-RADS / Fleischner): +1 raises, -1 lowers, 0 weak/unclear
GUIDE = {"spiculated": 1, "lobulated": 1, "ill_defined": 1, "sharp_margin": -1, "solid": 0, "part_solid": 1,
         "ground_glass": -1, "round": -1, "irregular_shape": 1, "benign_calc": -1, "other_calc": 0, "fat": -1,
         "large": 1, "small": -1}
# Separate streams prevent adding one analysis from changing another's result.
rng_fit = np.random.default_rng(0)
rng_soft = np.random.default_rng(1)
N_RESAMPLES = 5000

# ---------------- reader ratings -> per-nodule concept soft labels, labels, disagreement
R = pd.read_csv(f"{RES}/lidc_ratings.csv")
B = pd.DataFrame({"nodule": R.nodule, "patient": R.patient, "malignancy": R.malignancy,
    "spiculated": R.spiculation >= 3, "lobulated": R.lobulation >= 3, "ill_defined": R.margin <= 2,
    "sharp_margin": R.margin >= 4, "solid": R.texture == 5, "part_solid": R.texture.between(2, 4),
    "ground_glass": R.texture == 1, "round": R.sphericity >= 4, "irregular_shape": R.sphericity <= 2,
    "benign_calc": R.calcification.isin([1, 2, 3, 5]), "other_calc": R.calcification == 4,
    "fat": R.internalStructure == 3, "large": R.diameter > 15, "small": R.diameter < 6})
CONC = list(GUIDE)
g = B.groupby("nodule")
N = g[CONC].mean().astype(float)                                  # fraction of readers marking each finding
N["n_readers"] = g.size(); N["patient"] = g.patient.first()
N["mal_mean"] = g.malignancy.mean(); N["mal_std"] = g.malignancy.std()
N["mal_split"] = g.malignancy.apply(lambda v: float(v.min() <= 2 and v.max() >= 4))
N["attr_dis"] = (4 * N[CONC] * (1 - N[CONC])).mean(1)               # 0 = readers agree on every finding
N_ind = N[N.mal_mean == 3]                                          # indeterminate nodules, used in step-2 test only
N = N[N.mal_mean != 3]                                              # standard: drop indeterminate mean rating
N["y"] = (N.mal_mean > 3).astype(int)                               # 1 = malignant (MAL)

# ---------------- image side (only nodules with crops)
# ENC = biomedclip / dinov3: frozen embeddings, findings from bootstrap probes
# ENC = cnn: out-of-fold finding probabilities (n, finding, seed) from the soft-label 2.5D CNN (kaggle/lidc-cnn)
CNN = ENC == "cnn"
E = np.load(CNN_FILE if CNN else f"{RES}/lidc_emb_{ENC}.npz")
eid = list(E["ids"]); keep = [i for i in N.index if i in set(eid)]
N = N.loc[keep]; pos = [eid.index(i) for i in keep]
if CNN:
    assert list(E["concepts"]) == CONC
    V = {k[2:]: E[k][pos] for k in E.files if k.startswith("P_")}
    MBB = {k[2:]: E[k][pos].mean(-1) for k in E.files if k.startswith("M_")}
else:
    V = {k: E[k][pos] for k in E.files if k != "ids"}
X0 = V["tta0"]
pats = N.patient.unique(); np.random.default_rng(SPLIT_SEED).shuffle(pats)
te_p = set(pats[:int(len(pats) * TEST_FRAC)])
te = np.where(N.patient.isin(te_p))[0]; fit = np.where(~N.patient.isin(te_p))[0]
C = (N[CONC].values >= 0.5).astype(float)                           # reader-majority concepts ("true findings")
y = N.y.values
print(f"nodules with label+crop: {len(N)} (malignant {y.mean():.2f}) | fit {len(fit)} / test {len(te)} | "
      f">=3 readers in test: {(N.n_readers.values[te] >= 3).sum()}")

# ---------------- knowledge: LLM panel edges (comparative + forced-choice; each model x style is one member)
def load(style):
    v = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/llm_{style}_*.csv"))])
    if style == "comp":
        v = v.groupby(["model", "concept", "diag"])[["p_up", "p_down", "p_unsure"]].mean().reset_index()
        v["w"] = v.p_up - v.p_down
    else:
        v = v.groupby(["model", "concept"])[[f"p_{d}" for d in DIAGS] + ["p_unsure"]].mean().reset_index()
        v = v.melt(id_vars=["model", "concept", "p_unsure"], value_vars=[f"p_{d}" for d in DIAGS], var_name="diag", value_name="q")
        v["diag"] = v.diag.str[2:]
        qn = v.q / (1 - v.p_unsure + 1e-6)
        v["w"] = 2 * qn - 1                                         # 2 diagnoses: q=1 -> +1, q=0.5 -> 0, q=0 -> -1
    v["model"] = v.model + "|" + style
    return v[["model", "concept", "diag", "w", "p_unsure"]]
pm = pd.concat([load("comp"), load("choice")])
models = sorted(pm.model.unique())
def edges(df):
    gg = df.groupby(["concept", "diag"])
    W = gg.w.mean().unstack().loc[CONC, DIAGS].values
    H = (gg.p_unsure.mean() + gg.w.std().fillna(0)).unstack().loc[CONC, DIAGS].values
    return W, H
W_mid, H_edge = edges(pm)
W_models = [edges(pm[pm.model == m])[0] for m in models]
gs = np.array([GUIDE[c] for c in CONC])
nz = gs != 0
print(f"\npanel members {len(models)} | guideline sign agreement (MAL column, {nz.sum()} directional findings): "
      f"{np.mean(np.sign(W_mid[nz, 0]) == gs[nz]):.2f} | Spearman with guideline {spearmanr(W_mid[:, 0], gs).correlation:.2f}")
phi = np.array([np.corrcoef(C[fit, i], y[fit])[0, 1] if C[fit, i].std() > 0 else 0 for i in range(len(CONC))])
print(f"LLM MAL edge vs data phi Spearman {spearmanr(W_mid[:, 0], phi).correlation:.2f}")
print(pd.DataFrame({"guide": gs, "data_phi": phi.round(2), "LLM_w": W_mid[:, 0].round(2), "edge_hes": H_edge[:, 0].round(2)}, index=CONC).to_string())

# ---------------- perception: bootstrap probes on the clean view (or CNN seeds)
if not CNN:
    boot = [rng_fit.choice(fit, len(fit)) for _ in range(N_BOOT)]
    probes = [[LogisticRegression(max_iter=2000, class_weight="balanced").fit(X0[b], C[b, i]) if C[b, i].std() > 0 else None
               for b in boot] for i in range(len(CONC))]
def concepts(X):
    if CNN: return X.mean(-1), X
    P = np.stack([np.stack([p.predict_proba(X)[:, 1] if p is not None else np.zeros(len(X)) for p in probes[i]], 1)
                  for i in range(len(CONC))], 1)                    # (n, conc, boot)
    return P.mean(-1), P
Xm, Xs = concepts(X0)
cauc = {c: roc_auc_score(C[te, i], Xm[te, i]) for i, c in enumerate(CONC) if C[te, i].std() > 0}
summary["concept_auc"] = float(np.mean(list(cauc.values())))
print(f"\nconcept AUROC (test) mean {np.mean(list(cauc.values())):.3f}: " + ", ".join(f"{k} {v:.2f}" for k, v in cauc.items()))
t3 = te[N.n_readers.values[te] >= 3]
fr = N[CONC].values
rho = [spearmanr(Xm[t3, i], fr[t3, i]).correlation for i in range(len(CONC)) if fr[t3, i].std() > 0]
print(f"predicted prob vs reader fraction (test, >=3 readers): mean Spearman over findings {np.nanmean(rho):.3f}")
print(f"predicted finding ambiguity 4p(1-p) vs reader finding disagreement: Spearman "
      f"{spearmanr((4 * Xm * (1 - Xm)).mean(1)[t3], N.attr_dis.values[t3]).correlation:.3f}")

# ---------------- reasoning (2 calibration params per diagnosis)
def fit_cal(Z): return [LogisticRegression(max_iter=1000).fit(Z[fit, j:j + 1], (y[fit] == (1 if d == "MAL" else 0))) for j, d in enumerate(DIAGS)]
def prob(cal, Z): return np.stack([cal[j].predict_proba(Z[:, j:j + 1])[:, 1] for j in range(len(DIAGS))], 1)
def pmal(P): return P[:, 0] / (P[:, 0] + P[:, 1])
cal = fit_cal(Xm @ W_mid); P = prob(cal, Xm @ W_mid); pm_ = pmal(P); k = (pm_ > 0.5).astype(int)
cal_or = fit_cal(C @ W_mid); k_or = (pmal(prob(cal_or, C @ W_mid)) > 0.5).astype(int)
print("\nDiagnosis (test AUROC / acc):")
def show(name, s):
    summary["auc: " + name] = float(roc_auc_score(y[te], s[te]))
    print(f"   {name:42s} {roc_auc_score(y[te], s[te]):.3f} / {np.mean((s[te] > 0.5) == y[te]):.3f}")
show("true findings + LLM panel edges", pmal(prob(cal_or, C @ W_mid)))
show("predicted findings + LLM panel edges [OURS]", pm_)
for name, Xf in [("true findings + learned edges", C), ("predicted findings + learned edges", Xm)] + ([] if CNN else [("black-box probe on embedding", X0)]):
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xf[fit], y[fit]); show(name, lr.predict_proba(Xf)[:, 1])
if CNN: show("black-box CNN trained on malignancy", MBB["tta0"])

# ---------------- hesitancy components + confidence
#   *_flip / perc_bern: how often the decision flips;  *_soft: how much the malignancy probability moves (std)
pred = lambda Xx, W: (pmal(prob(cal, Xx @ W)) > 0.5).astype(int)
pprob = lambda Xx, W: pmal(prob(cal, Xx @ W))
W_scaled = [Wm * np.linalg.norm(W_mid) / np.linalg.norm(Wm) for Wm in W_models]
def hesitancy(xm, xs, views):
    kk = pred(xm, W_mid)
    bern = [(rng_soft.random(xm.shape) < xm).astype(float) for _ in range(N_DRAW)]
    out = {"perc_bern": exact_flip_probabilities(xm, lambda binary: pred(binary, W_mid), kk),
           "perc_boot": np.mean([pred(xs[:, :, b], W_mid) != kk for b in range(xs.shape[2])], 0),
           "know_flip": np.mean([pred(xm, Ws) != kk for Ws in W_scaled], 0),
           "perc_soft": np.std([pprob(xb, W_mid) for xb in bern], 0),
           "perc_seed_soft": np.std([pprob(xs[:, :, b], W_mid) for b in range(xs.shape[2])], 0),
           "know_soft": np.std([pprob(xm, Ws) for Ws in W_scaled], 0),
           "1-conf": 1 - np.abs(2 * pprob(xm, W_mid) - 1)}
    if views is not None:
        out["perc_tta"] = np.mean([pred(concepts(views[v])[0], W_mid) != kk for v in ["tta1", "tta2", "tta3"]], 0)
    return out
H = hesitancy(Xm, Xs, V)
know_flip, unc = H["know_flip"], H["1-conf"]

# ---------------- test 1+2: match to radiologist disagreement (test nodules with >=3 readers)
m3 = te[N.n_readers.values[te] >= 3]
ad, md = N.attr_dis.values[m3], N.mal_std.values[m3]
resid = md - LinearRegression().fit(ad[:, None], md).predict(ad[:, None])     # malignancy disagreement not explained by attributes
print(f"\nMatch to radiologist disagreement ({len(m3)} test nodules with >=3 readers), Spearman:")
print(f"   {'':14s} {'attribute dis.':>13s} {'malignancy dis.':>16s} {'mal. dis. | attr':>17s}")
for kk, v in H.items():
    summary[f"readers: {kk}"] = [float(spearmanr(v[m3], ad).correlation), float(spearmanr(v[m3], md).correlation), float(spearmanr(v[m3], resid).correlation)]
    print(f"   {kk:14s} {spearmanr(v[m3], ad).correlation:13.3f} {spearmanr(v[m3], md).correlation:16.3f} {spearmanr(v[m3], resid).correlation:17.3f}")

# ---------------- test 3: error-type attribution
wrong, wrong_or = k[te] != y[te], k_or[te] != y[te]
perc_err = wrong & ~wrong_or
rk = lambda v: pd.Series(v).rank().values / len(v)
print(f"\nError-type attribution (test errors {wrong.sum()} = perception {perc_err.sum()} + knowledge {(wrong & wrong_or).sum()}):")
def attribution_auc(score, indices):
    errors = wrong[indices]
    labels = perc_err[indices][errors]
    if labels.size == 0 or labels.min() == labels.max():
        return np.nan
    return roc_auc_score(labels, score[indices][errors])

bootstrap_rng = np.random.default_rng(2)
resamples = bootstrap_rng.integers(len(te), size=(N_RESAMPLES, len(te)))
summary["attribution_bootstrap"] = {"unit": "nodule", "resamples": N_RESAMPLES, "seed": 2}

def bootstrap_auc(score):
    values = np.array([attribution_auc(score, indices) for indices in resamples])
    return values[np.isfinite(values)]

PAIRS = [("perc_bern", "know_flip"), ("perc_boot", "know_flip"), ("perc_tta", "know_flip"),
         ("perc_soft", "know_soft"), ("perc_seed_soft", "know_soft")]
for pn, kn in PAIRS:
    s = rk(H[pn][te]) - rk(H[kn][te])
    A = bootstrap_auc(s)
    point = attribution_auc(s, np.arange(len(te)))
    summary[f"attr: {pn} - {kn}"] = float(point)
    print(f"   {pn:14s} - {kn:9s} AUROC {point:.3f}  95% CI [{np.percentile(A, 2.5):.3f}, {np.percentile(A, 97.5):.3f}]")
def ci(score):
    A = bootstrap_auc(score)
    point = attribution_auc(score, np.arange(len(te)))
    return f"{point:.3f} [{np.percentile(A, 2.5):.3f}, {np.percentile(A, 97.5):.3f}]"
summary["attr: 1-conf"] = float(roc_auc_score(perc_err[wrong], unc[te][wrong]))
summary["n_errors"], summary["n_perception_errors"] = int(wrong.sum()), int(perc_err.sum())
print(f"   uncertainty (1-conf) alone           AUROC {ci(unc[te])}")
print(f"   1-conf + (perc_bern - know_flip)     AUROC {ci(rk(unc[te]) + rk(H['perc_bern'][te]) - rk(know_flip[te]))}")

# ---------------- degradation sweep
print("\nDegradation sweep (test means):")
for v in [v for v in ["tta0", "blur1", "blur2", "blur4", "blur8", "noise100", "noise200"] if v in V]:
    xm, _ = concepts(V[v]); kv = pred(xm, W_mid)
    pb = exact_flip_probabilities(xm, lambda binary: pred(binary, W_mid), kv)
    kf = np.mean([pred(xm, Wm * np.linalg.norm(W_mid) / np.linalg.norm(Wm)) != kv for Wm in W_models], 0)
    ca = np.mean([roc_auc_score(C[te, i], xm[te, i]) for i in range(len(CONC)) if C[te, i].std() > 0])
    print(f"   {v:9s} perc_bern {pb[te].mean():.3f}  know_flip {kf[te].mean():.3f}  concept AUROC {ca:.3f}  acc {np.mean(kv[te] == y[te]):.3f}")

# ---------------- extra checks
# paired bootstrap: is (our split) - (plain uncertainty) > 0 for error-type attribution?
s_ours = rk(H["perc_bern"][te]) - rk(know_flip[te]); s_unc = unc[te]
D = []
for indices in resamples:
    a = attribution_auc(s_ours, indices)
    b = attribution_auc(s_unc, indices)
    if np.isfinite(a) and np.isfinite(b):
        D.append(a - b)
D = np.asarray(D)
point_diff = attribution_auc(s_ours, np.arange(len(te))) - attribution_auc(s_unc, np.arange(len(te)))
summary["attribution_bootstrap"]["paired_difference"] = {
    "point": float(point_diff), "ci95": np.percentile(D, [2.5, 97.5]).tolist(),
    "one_sided_p": float(np.mean(D <= 0))}
print(f"\nours minus plain uncertainty (attribution AUROC): {point_diff:.3f}, 95% CI [{np.percentile(D, 2.5):.3f}, {np.percentile(D, 97.5):.3f}], "
      f"P(diff<=0) = {np.mean(D <= 0):.3f}")

# Patient-clustered resampling accounts for multiple nodules from the same patient.
test_patients = N.patient.values[te]
patient_ids = np.unique(test_patients)
groups = [np.flatnonzero(test_patients == patient) for patient in patient_ids]
cluster_rng = np.random.default_rng(4)
cluster_diff = []
for _ in range(N_RESAMPLES):
    sampled = cluster_rng.integers(len(groups), size=len(groups))
    indices = np.concatenate([groups[j] for j in sampled])
    a = attribution_auc(s_ours, indices)
    b = attribution_auc(s_unc, indices)
    if np.isfinite(a) and np.isfinite(b):
        cluster_diff.append(a - b)
cluster_diff = np.asarray(cluster_diff)
summary["attribution_bootstrap"]["patient_clustered"] = {
    "n_patients": len(patient_ids), "seed": 4,
    "ci95": np.percentile(cluster_diff, [2.5, 97.5]).tolist(),
    "one_sided_p": float(np.mean(cluster_diff <= 0))}
print(f"patient-clustered paired difference: {point_diff:.3f}, 95% CI "
      f"[{np.percentile(cluster_diff, 2.5):.3f}, {np.percentile(cluster_diff, 97.5):.3f}], "
      f"P(diff<=0) = {np.mean(cluster_diff <= 0):.3f} ({len(patient_ids)} test patients)")
# reader disagreement weighted by how much each finding matters for the decision (|MAL edge - BEN edge|)
imp = np.abs(W_mid[:, 0] - W_mid[:, 1]); imp = imp / imp.sum()
ad_w = (4 * N[CONC].values * (1 - N[CONC].values) * imp).sum(1)[m3]
print("decision-weighted finding disagreement, Spearman:", {k: round(spearmanr(v[m3], ad_w).correlation, 3) for k, v in H.items()})

# ---------------- step 2: are indeterminate nodules (mean malignancy rating exactly 3) more hesitant?
ind = [i for i in N_ind.index if i in set(eid)]
if CNN and ind:
    pi = [eid.index(i) for i in ind]
    Xi = E["P_tta0"][pi]
    Hi = hesitancy(Xi.mean(-1), Xi, None)
    print(f"\nIndeterminate nodules ({len(ind)}) vs clear test nodules ({len(te)}): AUROC of each measure for 'indeterminate'")
    for kk in Hi:
        a_ = roc_auc_score(np.r_[np.ones(len(ind)), np.zeros(len(te))], np.r_[Hi[kk], H[kk][te]])
        summary[f"indeterminate: {kk}"] = float(a_)
        print(f"   {kk:14s} {a_:.3f}")

tag = f"{ENC}_{CNN_FILE.split('/')[-1].replace('.npz', '')}_split{SPLIT_SEED}"
json.dump(summary, open(f"{RES}/summary_{tag}.json", "w"), indent=1)
print("summary ->", f"{RES}/summary_{tag}.json")
