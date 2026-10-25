# Hesitancy-guided expert review: a simulated expert corrects k edges (replaced by the data-derived phi association,
# scaled to the panel's edge range). Which k edges to show the expert: most hesitant vs random vs least hesitant.
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.argv = [sys.argv[0], "results", "meta", "monet", "0", sys.argv[1], sys.argv[2]]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/pilot.py").read().split("# ---------------- perception side")[0])   # labels + edges + phi
E = np.load("results/emb_monet.npz"); Xc = E["img_clean_0"]
probe = {c: LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xc[fit_idx], C[c].values[fit_idx]) for c in CONC}
Xp = np.stack([probe[c].predict_proba(Xc)[:, 1] for c in CONC], 1)
expert = phi / np.abs(phi).max() * np.abs(W_mid).max()          # simulated expert opinion per edge
sig = lambda v: 1 / (1 + np.exp(-v))
def score(W, X):
    cal = [LogisticRegression(max_iter=1000).fit((X @ W)[fit_idx, j:j+1], Y[d].values[fit_idx]) for j, d in enumerate(DIAGS)]
    p = np.stack([cal[j].predict_proba((X @ W)[:, j:j+1])[:, 1] for j in range(4)], 1)
    m4 = te[in4[te]]
    return roc_auc_score(Y.MEL.values[te], p[te, 0]), np.mean(p[m4].argmax(1) == ycls[m4])
rng = np.random.default_rng(0)
order_h = np.argsort(-edge_hes.ravel())
rows = []
for k in [0, 3, 6, 10, 15, 20, 30]:
    for name, sel in [("most hesitant", [order_h[:k]]), ("least hesitant", [order_h[::-1][:k]]),
                      ("random", [rng.choice(60, k, replace=False) for _ in range(30)])]:
        r = []
        for s in sel:
            W = W_mid.copy().ravel(); W[s] = expert.ravel()[s]; W = W.reshape(W_mid.shape)
            r.append(score(W, C.values) + score(W, Xp))
        r = np.mean(r, 0)
        rows.append({"k": k, "edges shown": name, "oracle MEL": r[0], "oracle acc4": r[1], "pred MEL": r[2], "pred acc4": r[3]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
