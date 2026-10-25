# Does the perceptual/knowledge split tell us WHY a case is uncertain?
# A) error-type attribution: perception errors (wrong with predicted concepts, right with true concepts)
#    vs knowledge errors (wrong even with true concepts)
# B) actionability: gain from "human checks the findings" (true concepts) vs "expert fixes the rules" (data-derived edges)
#    in cases flagged perception-dominant vs knowledge-dominant
import sys, io, contextlib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ENC_ = sys.argv[3] if len(sys.argv) > 3 else "monet"
SRC = sys.argv[2] if len(sys.argv) > 2 else "orig+comp+choice"
N_DRAW = 200
sys.argv = [sys.argv[0], "results", "meta", ENC_, "0", sys.argv[1], SRC]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/pilot.py").read().split("# ---------------- hesitancy.")[0])

rng = np.random.default_rng(0)
cal = fit_calib(X_mid @ W_mid, fit_idx)
pred = lambda X, W: apply_calib(cal, X @ W).argmax(1)
k = pred(X_mid, W_mid)
cal_or = fit_calib(Cv @ W_mid, fit_idx)
k_or = apply_calib(cal_or, Cv @ W_mid).argmax(1)

# uncertainty sources (decision flip rates)
perc_bern = np.mean([pred((rng.random(X_mid.shape) < X_mid).astype(float), W_mid) != k for _ in range(N_DRAW)], 0)
perc_boot = np.mean([pred(X_samp[:, :, b], W_mid) != k for b in range(X_samp.shape[2])], 0)
W_models = [edge_matrix(pm[pm.model == m])[0] for m in models]
know_flip = np.mean([pred(X_mid, Wm * np.linalg.norm(W_mid) / np.linalg.norm(Wm)) != k for Wm in W_models], 0)
conf = apply_calib(cal, X_mid @ W_mid).max(1)

m4 = te[in4[te]]
y = ycls[m4]
wrong, wrong_or = k[m4] != y, k_or[m4] != y
perc_err = wrong & ~wrong_or          # fixing the findings fixes it
know_err = wrong & wrong_or           # wrong even with true findings
print(f"test 4-class cases {len(m4)}: errors {wrong.sum()} = perception {perc_err.sum()} + knowledge {know_err.sum()}")

rk = lambda v: pd.Series(v).rank().values / len(v)
print("\nA) among errors, AUROC for 'this is a perception error' (0.5 = no attribution ability)")
E = wrong
for pn, pv in [("perc_bern", perc_bern), ("perc_boot", perc_boot)]:
    s = rk(pv[m4]) - rk(know_flip[m4])
    print(f"   {pn:10s} - know_flip : {roc_auc_score(perc_err[E], s[E]):.3f}   | {pn} alone {roc_auc_score(perc_err[E], pv[m4][E]):.3f}"
          f" | know_flip alone (reversed) {roc_auc_score(perc_err[E], -know_flip[m4][E]):.3f}")
print(f"   confidence alone (reference)  : {roc_auc_score(perc_err[E], conf[m4][E]):.3f}")

print("\nB) actionability: accuracy gain from each fix, split by the dominant uncertainty source (perc_bern vs know_flip)")
cal_fix = fit_calib(X_mid @ (phi / np.abs(phi).max() * np.abs(W_mid).max()), fit_idx)
W_exp = phi / np.abs(phi).max() * np.abs(W_mid).max()
k_fixC = k_or                                              # human verifies the findings
k_fixW = apply_calib(cal_fix, X_mid @ W_exp).argmax(1)     # rules replaced by expert (data-derived) edges
dom = rk(perc_bern[m4]) > rk(know_flip[m4])
uncertain = rk(-conf[m4]) > 0.5                           # restrict to the less-confident half, where review matters
rows = []
for name, g in [("perception-dominant", dom & uncertain), ("knowledge-dominant", ~dom & uncertain)]:
    base = np.mean(k[m4][g] == y[g])
    rows.append({"group": name, "n": g.sum(), "acc": base,
                 "gain fix findings": np.mean(k_fixC[m4][g] == y[g]) - base,
                 "gain fix rules": np.mean(k_fixW[m4][g] == y[g]) - base})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# bootstrap 95% CIs over test cases
B = 2000; br = np.random.default_rng(1)
s = rk(perc_bern[m4]) - rk(know_flip[m4])
okC, okW, ok0 = (k_fixC[m4] == y).astype(float), (k_fixW[m4] == y).astype(float), (k[m4] == y).astype(float)
pg, kg = dom & uncertain, ~dom & uncertain
A, I = [], []
for _ in range(B):
    i = br.integers(len(m4), size=len(m4))
    e = E[i]
    if 0 < perc_err[i][e].sum() < e.sum(): A.append(roc_auc_score(perc_err[i][e], s[i][e]))
    a, b = pg[i], kg[i]
    I.append(((okC[i][a] - okW[i][a]).mean()) - ((okC[i][b] - okW[i][b]).mean()))
print(f"\nA) attribution AUROC 95% CI: [{np.percentile(A, 2.5):.3f}, {np.percentile(A, 97.5):.3f}]")
print(f"B) interaction (fix-findings minus fix-rules advantage, perc-group minus know-group): "
      f"{np.mean(I):.3f}, 95% CI [{np.percentile(I, 2.5):.3f}, {np.percentile(I, 97.5):.3f}]")
U = []
for _ in range(B):
    i = br.integers(len(m4), size=len(m4)); e = E[i]
    if 0 < perc_err[i][e].sum() < e.sum(): U.append(roc_auc_score(perc_err[i][e], -conf[m4][i][e]))
print(f"uncertainty (1-conf) attribution AUROC {roc_auc_score(perc_err[E], -conf[m4][E]):.3f} 95% CI [{np.percentile(U, 2.5):.3f}, {np.percentile(U, 97.5):.3f}]")
