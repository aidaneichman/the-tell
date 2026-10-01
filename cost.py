"""Is sequencing predictability associated with worse pitcher outcomes? Pitch level, 2025-26, pitches with a
previous pitch in the PA. seqpred = out-of-sample bits by which the previous pitch made this pitch more predictable
(leave-one-game-out). Two-way fixed effects: pitcher-season, and pitch type x count x batter hand; velocity
(z-scored within pitcher-season x type) as a control. Variant 'prevout' adds the previous pitch's outcome
(ball / called strike / swinging strike / foul). SEs clustered by pitcher. Writes out/cost.json."""
import numpy as np, pandas as pd, json
from common import load
df = load(); pp = pd.read_csv("out/pitch_probs.csv").set_index("row"); df = df.join(pp)
SW = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt", "missed_bunt", "bunt_foul_tip"}
WH = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
df["swing"] = df.description.isin(SW).astype(float); df["whiff"] = df.description.isin(WH).astype(float)
df["rv"] = -df.delta_run_exp
df["seqpred"] = (df.logp_seq - df.logp_sit) / np.log(2)
df["cell"] = ((df.x.astype(int) * 12 + df["count"].astype(int)) * 2 + df.hand.astype(int))
df["ps"] = df.pitcher.astype(np.int64) * 10000 + df.season
df["velo_z"] = df.groupby(["ps", "x"]).release_speed.transform(lambda v: (v - v.mean()) / (v.std() + 1e-9)).fillna(0)
po = df.description.shift(1)
cat = np.select([po.isin(["called_strike"]), po.isin(["swinging_strike", "swinging_strike_blocked", "missed_bunt"]),
                 po.isin(["foul", "foul_tip", "foul_bunt", "bunt_foul_tip"])], [1, 2, 3], 0)          # 0 = ball
for k in (1, 2, 3): df[f"po{k}"] = (cat == k).astype(float)
def fe2(d, cols, g1="ps", g2="cell", it=10):
    X = d[cols].to_numpy(float).copy(); ia = pd.factorize(d[g1])[0]; ib = pd.factorize(d[g2])[0]
    for _ in range(it):
        for idx in (ia, ib):
            cnt = np.bincount(idx)
            for j in range(X.shape[1]): X[:, j] -= (np.bincount(idx, weights=X[:, j]) / cnt)[idx]
    return X
def reg(d, y, xs, cl="pitcher"):
    d = d.dropna(subset=[y] + xs)
    Z = fe2(d, [y] + xs); Y = Z[:, 0]; X = Z[:, 1:]
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]; e = Y - X @ beta
    XtXi = np.linalg.inv(X.T @ X); g = pd.factorize(d[cl])[0]
    U = np.vstack([np.bincount(g, weights=X[:, j] * e) for j in range(X.shape[1])]).T
    V = XtXi @ (U.T @ U) @ XtXi
    return dict(n=len(d), beta=beta.tolist(), se=np.sqrt(np.diag(V)).tolist(), ymean=float(d[y].mean()))
d = df[df.season.isin([2025, 2026]) & (df.prev < 8)].dropna(subset=["seqpred"])
lo, hi = d.seqpred.quantile([.1, .9]); out = {"seqpred_p10_p90": [float(lo), float(hi)], "span": float(hi - lo)}
c = d[d.description == "hit_into_play"]
for tag, xs in (("base", ["seqpred", "velo_z"]), ("prevout", ["seqpred", "velo_z", "po1", "po2", "po3"])):
    out[tag] = {"swing": reg(d, "swing", xs), "whiff_given_swing": reg(d[d.swing == 1], "whiff", xs), "rv": reg(d, "rv", xs),
                "xwoba_contact": reg(c, "estimated_woba_using_speedangle", xs)}
json.dump(out, open("out/cost.json", "w"), indent=1)
for tag in ("base", "prevout"):
    for k, v in out[tag].items():
        b, se = v["beta"][0], v["se"][0]
        print(tag, k, "n", v["n"], "per bit %.5f t %.2f  p10->p90 %.5f" % (b, b / se, b * out["span"]))
