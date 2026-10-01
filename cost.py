"""Does sequencing predictability cost anything? Pitch level, within pitcher-season, controlling for pitch type x count and velocity."""
import numpy as np, pandas as pd, json
from common import load
df = load(); pp = pd.read_csv("out/pitch_probs.csv").set_index("row"); df = df.join(pp)
SW = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt", "missed_bunt", "bunt_foul_tip"}
WH = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
df["swing"] = df.description.isin(SW).astype(float); df["whiff"] = df.description.isin(WH).astype(float)
df["rv"] = -df.delta_run_exp                                  # runs saved by the pitcher on this pitch
df["seqpred"] = (df.logp_seq - df.logp_sit) / np.log(2)       # bits by which the previous pitch made this pitch more predictable
df["pred"] = df.logp_seq / np.log(2)
df["cell"] = df.x.astype(int) * 12 + df["count"].astype(int)
df["ps"] = df.pitcher.astype(np.int64) * 10000 + df.season
df["velo_z"] = df.groupby(["ps", "x"]).release_speed.transform(lambda v: (v - v.mean()) / (v.std() + 1e-9)).fillna(0)
def fe2(d, cols, g1="ps", g2="cell", it=8):
    X = d[cols].to_numpy(float).copy(); a = d[g1].to_numpy(); b = d[g2].to_numpy()
    ia = pd.factorize(a)[0]; ib = pd.factorize(b)[0]
    for _ in range(it):
        for idx in (ia, ib):
            cnt = np.bincount(idx); 
            for j in range(X.shape[1]): X[:, j] -= (np.bincount(idx, weights=X[:, j]) / cnt)[idx]
    return X
def reg(d, y, xs, cl="pitcher"):
    d = d.dropna(subset=[y] + xs)
    Z = fe2(d, [y] + xs); Y = Z[:, 0]; X = Z[:, 1:]
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ beta
    XtXi = np.linalg.inv(X.T @ X); g = pd.factorize(d[cl])[0]
    S = np.zeros((X.shape[1], X.shape[1]))
    for k in range(g.max() + 1):
        m = g == k; u = X[m].T @ e[m]; S += np.outer(u, u)
    V = XtXi @ S @ XtXi
    return dict(n=len(d), beta=beta.tolist(), se=np.sqrt(np.diag(V)).tolist(), ymean=float(d[y].mean()))
out = {}
d = df[df.season.isin([2025, 2026])]
out["swing"] = reg(d, "swing", ["seqpred", "velo_z"])
out["whiff_given_swing"] = reg(d[d.swing == 1], "whiff", ["seqpred", "velo_z"])
out["rv"] = reg(d, "rv", ["seqpred", "velo_z"])
c = d[d.description == "hit_into_play"]
out["xwoba_contact"] = reg(c, "estimated_woba_using_speedangle", ["seqpred", "velo_z"])
# sanity: situational predictability itself (the count and hand), which a hitter knows
out["rv_on_pred"] = reg(d, "rv", ["pred", "velo_z"])
json.dump(out, open("out/cost.json", "w"), indent=1)
for k, v in out.items():
    print(k, "n", v["n"], "beta per bit", round(v["beta"][0], 5), "se", round(v["se"][0], 5), "t", round(v["beta"][0] / v["se"][0], 2), "mean y", round(v["ymean"], 4))
