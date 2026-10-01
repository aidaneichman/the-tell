"""Binned partial-residual plots for the cost regressions (two-way FE: pitcher-season, pitch type x count)."""
import numpy as np, pandas as pd, os
def fe2(d, cols, g1="ps", g2="cell", it=8):
    X = d[cols].to_numpy(float).copy(); ia = pd.factorize(d[g1])[0]; ib = pd.factorize(d[g2])[0]
    for _ in range(it):
        for idx in (ia, ib):
            cnt = np.bincount(idx)
            for j in range(X.shape[1]): X[:, j] -= (np.bincount(idx, weights=X[:, j]) / cnt)[idx]
    return X
def binned():
    if os.path.exists("out/cost_bins.pkl"): return pd.read_pickle("out/cost_bins.pkl")
    from common import load
    df = load(); df = df.join(pd.read_csv("out/pitch_probs.csv").set_index("row"))
    df = df[df.season.isin([2025, 2026])]
    SW = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt", "missed_bunt", "bunt_foul_tip"}
    WH = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
    df["swing"] = df.description.isin(SW).astype(float); df["whiff"] = df.description.isin(WH).astype(float)
    df["rv"] = -df.delta_run_exp; df["xw"] = df.estimated_woba_using_speedangle
    df["seqpred"] = (df.logp_seq - df.logp_sit) / np.log(2)
    df["cell"] = df.x.astype(int) * 12 + df["count"].astype(int); df["ps"] = df.pitcher.astype(np.int64) * 10000 + df.season
    out = {}
    for y, sub in (("whiff", df[df.swing == 1]), ("rv", df), ("xw", df[df.description == "hit_into_play"])):
        sub = sub.dropna(subset=[y, "seqpred"])
        Z = fe2(sub, [y, "seqpred"]); raw = sub.seqpred.to_numpy()
        q = pd.qcut(raw, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "x": raw, "y": Z[:, 0]}).groupby("q")
        out[y] = pd.DataFrame({"x": g.x.mean(), "y": g.y.mean(), "se": g.y.std() / np.sqrt(g.y.size())})
        out[y]["x"] = out[y]["x"] - raw.mean()
    pd.to_pickle(out, "out/cost_bins.pkl"); return out
if __name__ == "__main__": print(binned())
