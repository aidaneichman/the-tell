"""Memoryless given the situation. (1) Asymptotic conditional G-test of X_t indep X_{t-1} | (hand, count), with its level
checked by a simulator in which the count is endogenous. (2) A universal-inference e-process that is valid even though the
count depends on earlier pitches. Within-PA previous pitch only."""
import numpy as np, pandas as pd, json
from scipy.stats import chi2
from common import load
D = 8
def gstat(h, c, pv, x):
    s = h * 12 + c
    N = np.zeros((24, 9, D)); np.add.at(N, (s, pv, x), 1)
    G, df = 0.0, 0
    for k in range(24):
        T = N[k]; r = T.sum(1); q = T.sum(0); tot = T.sum()
        if tot == 0: continue
        with np.errstate(divide="ignore", invalid="ignore"):
            G += 2 * np.nansum(np.where(T > 0, T * np.log(T * tot / np.outer(r, q)), 0))
        df += max(0, ((r > 0).sum() - 1) * ((q > 0).sum() - 1))
    return G, max(df, 1)
def e_process(h, c, pv, x, lam=0.5):
    """log E = log prod_t Q_t(x_t | s_t, prev_t) - log sup_p prod_t p_{s_t}(x_t); Q_t fit on the past only (KT per cell,
    backed off to the pitcher's past per-state mix). E[Q_t/p_{s_t} | past] = 1 because s_t is fixed by the past."""
    s = h * 12 + c
    Nsx = np.zeros((24, D)); Nspx = np.zeros((24, 9, D)); logQ = 0.0
    for t in range(len(x)):
        base = (Nsx[s[t]] + lam) / (Nsx[s[t]].sum() + lam * D)
        q = (Nspx[s[t], pv[t]] + 4 * base) / (Nspx[s[t], pv[t]].sum() + 4)
        logQ += np.log(q[x[t]]); Nsx[s[t], x[t]] += 1; Nspx[s[t], pv[t], x[t]] += 1
    den = sum((T[T > 0] * np.log(T[T > 0] / T.sum())).sum() for T in Nsx)
    return logQ - den
if __name__ == "__main__":
    df = load(); d = df[df.season == 2025]
    # league outcome model for the simulator: P(ball, strike, foul-with-2-strikes-stays, ends PA | count, pitch type)
    desc = d.description
    ball = desc.isin(["ball", "blocked_ball", "pitchout", "hit_by_pitch"]); inplay = desc.eq("hit_into_play")
    strike = ~ball & ~inplay
    out_ev = d.events.notna()
    ok = np.zeros((12, D, 3))            # ball, strike, PA ends in play
    for (cc, xx), g in d.groupby(["count", "x"]):
        b = ball[g.index].mean(); p = inplay[g.index].mean(); ok[cc, xx] = [b, 1 - b - p, p]
    rows = []; rng = np.random.default_rng(5)
    keys = d.groupby("pitcher").size(); keys = keys[keys >= 600].sample(40, random_state=2).index
    for pid, g in d.groupby("pitcher"):
        h = g.hand.to_numpy(int); c = g["count"].to_numpy(int); pv = g.prev.to_numpy(int); x = g.x.to_numpy(int)
        G, df_ = gstat(h, c, pv, x)
        r = dict(pitcher=pid, n=len(x), G=G, df=df_, p_asym=chi2.sf(G, df_), loge=e_process(h, c, pv, x))
        if pid in keys:                   # simulate the null: same PA count and handedness, x ~ p_{hand,count}, count evolves
            s = h * 12 + c; pst = np.zeros((24, D)); np.add.at(pst, (s, x), 1); pst = (pst + .5) / (pst.sum(1, keepdims=True) + 4)
            npa = int((pv == 8).sum()); hands = h[pv == 8]
            rej = []
            for _ in range(200):
                H, C, PV, X = [], [], [], []
                for a in range(npa):
                    b_, k_ = 0, 0; prev = 8
                    while True:
                        cc = b_ * 3 + k_; xx = rng.choice(D, p=pst[hands[a] * 12 + cc])
                        H.append(hands[a]); C.append(cc); PV.append(prev); X.append(xx)
                        pb, ps, pi = ok[cc, xx]; u = rng.random()
                        if u < pb:
                            b_ += 1
                            if b_ == 4: break
                        elif u < pb + ps:
                            if k_ < 2: k_ += 1
                            elif rng.random() < 0.55: break          # strikeout; else a two-strike foul
                        else: break
                        prev = xx
                Gs, dfs = gstat(np.array(H), np.array(C), np.array(PV), np.array(X))
                rej.append(chi2.sf(Gs, dfs) <= 0.05)
            r["sim_level"] = float(np.mean(rej))
        rows.append(r)
    R = pd.DataFrame(rows); R.to_csv("out/conditional_2025.csv", index=False)
    print("conditional G-test rejects:", round((R.p_asym <= .05).mean(), 3), " e-process E>=20:", round((R.loge >= np.log(20)).mean(), 3))
    print("simulated level of the asymptotic conditional test (endogenous count), 40 pitchers x 200:",
          round(R.sim_level.mean(), 3), "range", round(R.sim_level.min(), 3), round(R.sim_level.max(), 3))
