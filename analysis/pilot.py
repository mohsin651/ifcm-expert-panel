# Pilot analysis: LLM-panel intuitionistic edges + MONET/BiomedCLIP concept intervals -> interval (IFS) reasoning on Derm7pt.
# IFS <-> interval: a value in [lo, hi] has membership mu = lo, non-membership gamma = 1 - hi, hesitancy pi = hi - lo.
import sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RES = sys.argv[1] if len(sys.argv) > 1 else "results"      # folder with llm_votes.csv, emb_*.npz
META = sys.argv[2] if len(sys.argv) > 2 else "meta"         # folder with meta.csv and *_indexes.csv
ENC = sys.argv[3] if len(sys.argv) > 3 else "monet"            # which image encoder for perception
N_BOOT = 30              # bootstrap probes for perceptual intervals
Q_LO, Q_HI = 10, 90      # percentiles of the bootstrap spread used as the interval
DIAGS = ["MEL", "NEV", "BCC", "SK"]
POINTS7 = {"atypical_pn": 2, "bwv": 2, "atypical_vessels": 2, "irregular_streaks": 1,
           "irregular_pigm": 1, "irregular_dg": 1, "regression": 1}

# ---------------- labels
meta = pd.read_csv(f"{META}/meta.csv")
tr = pd.read_csv(f"{META}/train_indexes.csv")["indexes"].values
va = pd.read_csv(f"{META}/valid_indexes.csv")["indexes"].values
te = pd.read_csv(f"{META}/test_indexes.csv")["indexes"].values
fit_idx = np.concatenate([tr, va])
N_FIT = int(sys.argv[4]) if len(sys.argv) > 4 else 0      # 0 = all 617 train+valid cases; else a random subset (label budget)
if N_FIT: fit_idx = np.random.default_rng(1).choice(fit_idx, N_FIT, replace=False)
PANEL = sys.argv[5].split(",") if len(sys.argv) > 5 else None   # optional subset of panel models

pg = meta.pigmentation
C = pd.DataFrame({
    "atypical_pn": meta.pigment_network == "atypical", "typical_pn": meta.pigment_network == "typical",
    "irregular_streaks": meta.streaks == "irregular", "regular_streaks": meta.streaks == "regular",
    "irregular_pigm": pg.isin(["diffuse irregular", "localized irregular"]),
    "regular_pigm": pg.isin(["diffuse regular", "localized regular"]),
    "regression": meta.regression_structures != "absent",
    "irregular_dg": meta.dots_and_globules == "irregular", "regular_dg": meta.dots_and_globules == "regular",
    "bwv": meta.blue_whitish_veil == "present", "arborizing": meta.vascular_structures == "arborizing",
    "atypical_vessels": meta.vascular_structures.isin(["dotted", "linear irregular"]),
    "comma": meta.vascular_structures == "comma", "hairpin": meta.vascular_structures == "hairpin",
    "vessels_regr": meta.vascular_structures == "within regression",
}).astype(float)
CONC = list(C.columns)
dx = meta.diagnosis
Y = pd.DataFrame({
    "MEL": dx.str.contains("melanoma"),
    "NEV": dx.str.contains("nevus"),
    "BCC": dx == "basal cell carcinoma",
    "SK": dx == "seborrheic keratosis",
}).astype(int)
in4 = Y.sum(1).values == 1                      # cases belonging to one of the 4 modeled diagnoses
ycls = np.where(in4, Y.values.argmax(1), -1)
difficulty = meta.level_of_diagnostic_difficulty.map({"low": 0, "medium": 1, "high": 2}).values
report = {}

# ---------------- knowledge side: LLM panel -> interval edges
import glob
EDGE_SRC = sys.argv[6] if len(sys.argv) > 6 else "orig"     # orig | comp | choice | comp+choice
def load_votes(style):
    """long format (model, concept, diag, w, p_unsure) averaged over prompt templates and option orders"""
    if style == "choice":
        v = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/llm_choice_*.csv"))])
        v = v.groupby(["model", "concept"])[[f"p_{d}" for d in DIAGS] + ["p_unsure"]].mean().reset_index()
        long = v.melt(id_vars=["model", "concept", "p_unsure"], value_vars=[f"p_{d}" for d in DIAGS], var_name="diag", value_name="q")
        long["diag"] = long.diag.str[2:]
        qn = long.q / (1 - long.p_unsure + 1e-6)                  # renormalise over the 4 diagnoses
        long["w"] = np.where(qn >= 0.25, (qn - 0.25) / 0.75, (qn - 0.25) / 0.25)
        return long[["model", "concept", "diag", "w", "p_unsure"]]
    files = {"orig": "llm_votes*.csv", "comp": "llm_comp_*.csv"}[style]
    v = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/{files}"))])
    v = v.groupby(["model", "concept", "diag"])[["p_up", "p_down", "p_none", "p_unsure"]].mean().reset_index()
    v["w"] = v.p_up - v.p_down
    return v[["model", "concept", "diag", "w", "p_unsure"]]

pm = []
for style in EDGE_SRC.split("+"):
    v = load_votes(style)
    v = v[v.model.isin(PANEL)] if PANEL else v
    v["model"] = v.model + "|" + style                           # each (model, question style) is one panel member
    pm.append(v)
pm = pd.concat(pm)
models = sorted(pm.model.unique())
print("panel members:", len(models), "| edge source:", EDGE_SRC)

def edge_matrix(df):
    """interval edges [lo, hi] in [-1,1]: within-model 'unsure' mass + between-model spread"""
    g = df.groupby(["concept", "diag"])
    w = g.w.mean(); q = g.p_unsure.mean(); sd = g.w.std().fillna(0)
    W = w.unstack().loc[CONC, DIAGS].values
    H = (q + sd).unstack().loc[CONC, DIAGS].values
    return W, np.clip(W - H, -1, 1), np.clip(W + H, -1, 1)

W_mid, W_lo, W_hi = edge_matrix(pm)
edge_hes = (W_hi - W_lo) / 2

# guideline recovery: melanoma column vs 7-point checklist points
pts = np.array([POINTS7.get(c, 0) for c in CONC])
r_panel = spearmanr(W_mid[:, 0], pts).correlation
report["guideline_spearman_panel"] = r_panel
report["guideline_spearman_per_model"] = {m: spearmanr(edge_matrix(pm[pm.model == m])[0][:, 0], pts).correlation for m in models}
print("\n7-point recovery (Spearman of LLM melanoma edge vs checklist points):", round(r_panel, 3))
for m, r in report["guideline_spearman_per_model"].items(): print(f"   {m:45s} {r:.3f}")
print(pd.DataFrame({"points": pts, "panel_w": W_mid[:, 0].round(2), "hesitancy": edge_hes[:, 0].round(2)}, index=CONC).to_string())

# data association (phi coefficient on the fit split) vs LLM edges
phi = np.array([[np.corrcoef(C.values[fit_idx, i], Y[d].values[fit_idx])[0, 1] if C.values[fit_idx, i].std() > 0 else 0
                 for d in DIAGS] for i in range(len(CONC))])
report["edge_vs_data_spearman"] = spearmanr(W_mid.ravel(), phi.ravel()).correlation
strong = np.abs(phi.ravel()) > 0.1
report["sign_agreement_on_strong_edges"] = float(np.mean(np.sign(W_mid.ravel()[strong]) == np.sign(phi.ravel()[strong])))
wrong = (np.sign(W_mid.ravel()) != np.sign(phi.ravel())) & strong
report["edge_hesitancy_wrong_vs_right"] = [float(edge_hes.ravel()[wrong].mean()), float(edge_hes.ravel()[strong & ~wrong].mean())]
print("\nLLM edge vs data phi, Spearman over 60 edges:", round(report["edge_vs_data_spearman"], 3),
      "| sign agreement on |phi|>0.1:", round(report["sign_agreement_on_strong_edges"], 3),
      "| mean edge hesitancy wrong/right:", np.round(report["edge_hesitancy_wrong_vs_right"], 3))
report["edge_hes_auroc_wrong_data"] = roc_auc_score(wrong[strong], edge_hes.ravel()[strong])
wrong7 = (pts > 0) != (W_mid[:, 0] > 0.5)
report["edge_hes_auroc_wrong_guideline"] = roc_auc_score(wrong7, edge_hes[:, 0]) if 0 < wrong7.sum() < len(wrong7) else float("nan")
print("edge hesitancy -> wrong edge AUROC: vs data", round(report["edge_hes_auroc_wrong_data"], 3),
      "| vs 7-point (MEL col)", round(report["edge_hes_auroc_wrong_guideline"], 3), "| wrong MEL edges:", list(np.array(CONC)[wrong7]))

# ---------------- perception side: concept intervals
E = np.load(f"{RES}/emb_{ENC}.npz")
tk = list(E["text_keys"]); T = E["text"]
def tvec(key): return T[tk.index(key)]
corr_keys = [k for k in E.files if k.startswith("img_")]

def zs_scores(X):
    """zero-shot score per concept and paraphrase: cos(pos) - cos(neg); shape (n, conc, 3)"""
    return np.stack([np.stack([X @ tvec(f"pos|{c}|{k}") - X @ tvec(f"neg|{c}|{k}") for k in range(3)], -1) for c in CONC], 1)

Xc = E["img_clean_0"]
S_clean = zs_scores(Xc)
zs_auc = {c: roc_auc_score(C[c].values[te], S_clean[te, i].mean(-1)) for i, c in enumerate(CONC) if C[c].values[te].std() > 0}
print("\nzero-shot concept AUROC (test):", {k: round(v, 2) for k, v in zs_auc.items()}, "mean", round(np.mean(list(zs_auc.values())), 3))
report["concept_auc_zeroshot"] = zs_auc

rng = np.random.default_rng(0)
boot = [rng.choice(fit_idx, len(fit_idx)) for _ in range(N_BOOT)]
probes = {}
for i, c in enumerate(CONC):
    y = C[c].values
    probes[c] = [LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xc[b], y[b]) if y[b].std() > 0 else None for b in boot]

def concept_intervals(X):
    P = np.stack([np.stack([p.predict_proba(X)[:, 1] if p is not None else np.zeros(len(X)) for p in probes[c]], 1) for c in CONC], 1)
    return np.percentile(P, Q_LO, -1), P.mean(-1), np.percentile(P, Q_HI, -1), P

X_lo, X_mid, X_hi, X_samp = concept_intervals(Xc)
pr_auc = {c: roc_auc_score(C[c].values[te], X_mid[te, i]) for i, c in enumerate(CONC) if C[c].values[te].std() > 0}
print("probe concept AUROC (test):", {k: round(v, 2) for k, v in pr_auc.items()}, "mean", round(np.mean(list(pr_auc.values())), 3))
report["concept_auc_probe"] = pr_auc

# ---------------- reasoning: interval aggregation z_d = sum_c x_c w_cd, calibrated per diagnosis with 2 params
def interval_z(xlo, xhi, wlo, whi):
    cands = np.stack([xlo[:, :, None] * wlo[None], xlo[:, :, None] * whi[None], xhi[:, :, None] * wlo[None], xhi[:, :, None] * whi[None]])
    return cands.min(0).sum(1), cands.max(0).sum(1)

def fit_calib(z_fit, idx):
    return [LogisticRegression(max_iter=1000).fit(z_fit[idx, j:j + 1], Y[d].values[idx]) for j, d in enumerate(DIAGS)]

sig = lambda v: 1 / (1 + np.exp(-v))
def apply_calib(cal, z):
    return np.stack([sig(cal[j].coef_[0, 0] * z[:, j] + cal[j].intercept_[0]) for j in range(len(DIAGS))], 1)

def evaluate(name, x_mid, W, idx_eval, x_fit=None):
    x_fit = x_mid if x_fit is None else x_fit
    cal = fit_calib(x_fit @ W, fit_idx)
    p = apply_calib(cal, x_mid @ W)
    auc = roc_auc_score(Y.MEL.values[idx_eval], p[idx_eval, 0])
    m4 = idx_eval[in4[idx_eval]]
    acc = np.mean(p[m4].argmax(1) == ycls[m4])
    print(f"  {name:45s} MEL AUROC {auc:.3f} | 4-class acc {acc:.3f}")
    report.setdefault("diagnosis", {})[name] = {"mel_auc": auc, "acc4": acc}
    return cal, p

print("\nDiagnosis on test (only 2 calibration params per diagnosis are fit):")
Cv = C.values
W7 = np.zeros_like(W_mid); W7[:, 0] = pts / 2
evaluate("oracle concepts + 7-point guideline (MEL only)", Cv, W7, te)
evaluate("oracle concepts + LLM panel edges", Cv, W_mid, te)
for m in models:
    evaluate(f"oracle concepts + {m.split('/')[-1]}", Cv, edge_matrix(pm[pm.model == m])[0], te)
evaluate("predicted concepts + 7-point guideline (MEL only)", X_mid, W7, te)
cal, P_ours = evaluate("predicted concepts + LLM panel edges [OURS]", X_mid, W_mid, te)
# learned-edge baselines (use labels to fit 15x4 weights)
for name, Xf in [("oracle concepts + learned edges (CBM)", Cv), ("predicted concepts + learned edges (CBM)", X_mid)]:
    lr = LogisticRegression(max_iter=3000).fit(Xf[fit_idx][in4[fit_idx]], ycls[fit_idx][in4[fit_idx]])
    lm = LogisticRegression(max_iter=3000).fit(Xf[fit_idx], Y.MEL.values[fit_idx])
    auc = roc_auc_score(Y.MEL.values[te], lm.predict_proba(Xf[te])[:, 1]); acc = lr.score(Xf[te][in4[te]], ycls[te][in4[te]])
    print(f"  {name:45s} MEL AUROC {auc:.3f} | 4-class acc {acc:.3f}"); report["diagnosis"][name] = {"mel_auc": auc, "acc4": acc}
# black box: linear probe on image embedding; zero-shot diagnosis
lm = LogisticRegression(max_iter=3000, C=1.0).fit(Xc[fit_idx], Y.MEL.values[fit_idx])
lr = LogisticRegression(max_iter=3000, C=1.0).fit(Xc[fit_idx][in4[fit_idx]], ycls[fit_idx][in4[fit_idx]])
auc = roc_auc_score(Y.MEL.values[te], lm.predict_proba(Xc[te])[:, 1]); acc = lr.score(Xc[te][in4[te]], ycls[te][in4[te]])
print(f"  {'black-box linear probe on image embedding':45s} MEL AUROC {auc:.3f} | 4-class acc {acc:.3f}"); report["diagnosis"]["blackbox_probe"] = {"mel_auc": auc, "acc4": acc}
D = np.stack([np.mean([Xc @ tvec(f"dx|{d}|{k}") for k in range(3)], 0) for d in DIAGS], 1)
auc = roc_auc_score(Y.MEL.values[te], D[te, 0] - D[te, 1:].max(1)); acc = np.mean(D[te][in4[te]].argmax(1) == ycls[te][in4[te]])
print(f"  {'zero-shot diagnosis prompts':45s} MEL AUROC {auc:.3f} | 4-class acc {acc:.3f}"); report["diagnosis"]["zeroshot_dx"] = {"mel_auc": auc, "acc4": acc}

# ---------------- hesitancy. Main prediction = panel-mean edges with one shared calibration ([OURS] above).
# spread measures (q90 - q10 of the predicted-class prob over MC draws; each panel member calibrated on its own edges):
#   perc / know / total
# decision-level measures (new):
#   perc_flip   : fraction of bootstrap concept probes that change the predicted diagnosis
#   know_flip   : fraction of panel members (edges rescaled to the panel-mean norm, shared calibration) that change it
#   know_contest: share of the evidence separating top-1 from top-2 diagnosis that rests on contested edges
N_MC = 300
W_models = [edge_matrix(pm[pm.model == m])[0] for m in models]
W_scaled = [Wm * np.linalg.norm(W_mid) / max(np.linalg.norm(Wm), 1e-9) for Wm in W_models]
def rankn(v): return pd.Series(v).rank().values / len(v)

def hesitancy(x_mid, x_samp, x_fit, seed=0):
    r = np.random.default_rng(seed)
    cal_s = fit_calib(x_fit @ W_mid, fit_idx)
    p0 = apply_calib(cal_s, x_mid @ W_mid)
    order = np.argsort(-p0, 1); k, k2 = order[:, 0], order[:, 1]; rows = np.arange(len(k))
    cals = [fit_calib(x_fit @ Wm, fit_idx) for Wm in W_models]
    out = {}
    for mode in ["total", "perc", "know"]:
        sims = []
        for _ in range(N_MC):
            x = x_samp[:, :, r.integers(x_samp.shape[2])] if (mode != "know" and x_samp is not None) else x_mid
            mi = r.integers(len(models)) if mode != "perc" else None
            p = apply_calib(cals[mi], x @ W_models[mi]) if mi is not None else apply_calib(cal_s, x @ W_mid)
            sims.append(p[rows, k])
        sims = np.stack(sims, 1)
        out[mode] = np.percentile(sims, Q_HI, 1) - np.percentile(sims, Q_LO, 1)
    out["know_flip"] = np.mean([apply_calib(cal_s, x_mid @ Ws).argmax(1) != k for Ws in W_scaled], 0)
    out["perc_flip"] = (np.mean([apply_calib(cal_s, x_samp[:, :, b] @ W_mid).argmax(1) != k for b in range(x_samp.shape[2])], 0)
                        if x_samp is not None else np.zeros(len(k)))
    d = np.abs(x_mid * (W_mid[:, k].T - W_mid[:, k2].T))
    h = (edge_hes[:, k].T + edge_hes[:, k2].T) / 2
    out["know_contest"] = (d * h).sum(1) / (d.sum(1) + 1e-9)
    return p0, out

P, H = hesitancy(X_mid, X_samp, X_mid)
m4 = te[in4[te]]
err = (P[m4].argmax(1) != ycls[m4]).astype(int)
conf = P[m4].max(1)
report["hesitancy"] = {f"error_auroc_{k}": roc_auc_score(err, v[m4]) for k, v in H.items()}
report["hesitancy"]["error_auroc_1-conf"] = roc_auc_score(err, -conf)
for kk in ["know_flip", "know_contest", "perc_flip"]:
    report["hesitancy"][f"error_auroc_1-conf+{kk}"] = roc_auc_score(err, rankn(-conf) + rankn(H[kk][m4]))
for kk, v in H.items(): report["hesitancy"][f"spearman_difficulty_{kk}"] = spearmanr(v[te], difficulty[te]).correlation
report["hesitancy"]["spearman_difficulty_1-conf"] = spearmanr(-P[te].max(1), difficulty[te]).correlation
print("\nHesitancy, predicted concepts (test, 4-class cases; error rate %.2f):" % err.mean())
for kk, v in report["hesitancy"].items(): print(f"   {kk:40s} {v:.3f}")

# knowledge-only test: with oracle concepts every error is a knowledge error -> does knowledge hesitancy predict it?
P_or, H_or = hesitancy(Cv, None, Cv)
err_or = (P_or[m4].argmax(1) != ycls[m4]).astype(int)
report["hesitancy_oracle"] = {f"error_auroc_{kk}": roc_auc_score(err_or, H_or[kk][m4]) for kk in ["know", "know_flip", "know_contest"]}
report["hesitancy_oracle"]["error_auroc_1-conf"] = roc_auc_score(err_or, -P_or[m4].max(1))
for kk in ["know_flip", "know_contest"]:
    report["hesitancy_oracle"][f"error_auroc_1-conf+{kk}"] = roc_auc_score(err_or, rankn(-P_or[m4].max(1)) + rankn(H_or[kk][m4]))
print("Hesitancy, oracle concepts (knowledge errors only; error rate %.2f):" % err_or.mean())
for kk, v in report["hesitancy_oracle"].items(): print(f"   {kk:40s} {v:.3f}")

print("\nCorruption sweep (mean over test): perceptual vs knowledge hesitancy, concept AUROC, 4-class acc")
sweep = []
for key in corr_keys:
    lo, mid, hi, samp = concept_intervals(E[key])
    Pk, Hk = hesitancy(mid, samp, X_mid)
    cauc = np.mean([roc_auc_score(C[c].values[te], mid[te, i]) for i, c in enumerate(CONC) if C[c].values[te].std() > 0])
    acc = np.mean(Pk[m4].argmax(1) == ycls[m4])
    sweep.append({"corruption": key[4:], "H_perc": Hk["perc"][te].mean(), "H_know": Hk["know"][te].mean(), "H_total": Hk["total"][te].mean(),
                  "perc_flip": Hk["perc_flip"][te].mean(), "know_flip": Hk["know_flip"][te].mean(), "know_contest": Hk["know_contest"][te].mean(), "concept_auc": cauc, "acc4": acc})
sweep = pd.DataFrame(sweep); print(sweep.round(3).to_string(index=False))
report["corruption_sweep"] = sweep.to_dict("records")

json.dump(report, open(f"{RES}/report_{ENC}_{EDGE_SRC}.json", "w"), indent=1, default=float)
pd.DataFrame(W_mid, index=CONC, columns=DIAGS).round(3).to_csv(f"{RES}/edges_mid.csv")
pd.DataFrame(edge_hes, index=CONC, columns=DIAGS).round(3).to_csv(f"{RES}/edges_hesitancy.csv")
