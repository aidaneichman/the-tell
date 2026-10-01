"""Reliability of TELL. Within season: independent split-half (models refit on odd games and on even games
separately, Spearman-Brown corrected). Across seasons: correlation of full-season values for pitchers in both.
Also empirical-Bayes shrinkage. Writes out/reliability.json, out/splithalf_indep.csv, out/tell_eb.csv."""
import numpy as np, pandas as pd, json
from common import load
from model import ladder
hp = json.load(open("out/hp.json")); R = pd.read_csv("out/tell.csv"); out = {}
df = load(); rows = []
for (pid, yr), g in df.groupby(["pitcher", "season"], sort=False):
    gi = pd.factorize(g.game_pk)[0]; r = dict(pitcher=pid, season=yr, n=len(g))
    for par in (0, 1):
        h = g[(gi % 2) == par]
        if h.game_pk.nunique() < 3: r[f"t{par}"] = np.nan; continue
        P, x, gm, G = ladder(h, hp); b = -np.log2(P[np.arange(len(x)), :, x]); k = h.prev.to_numpy() < 8
        r[f"t{par}"] = (b[k, 2] - b[k, 3]).mean()
    rows.append(r)
S = pd.DataFrame(rows).dropna(); S.to_csv("out/splithalf_indep.csv", index=False)
for yr in (2024, 2025, 2026):
    s = S[S.season == yr]; rr = np.corrcoef(s.t0, s.t1)[0, 1]; out[f"split_indep_{yr}"] = dict(r_half=rr, sb=2 * rr / (1 + rr), n=len(s))
for a, b in ((2024, 2025), (2025, 2026)):
    x = R[R.season == a].set_index("pitcher"); y = R[R.season == b].set_index("pitcher"); j = x.index.intersection(y.index)
    for col in ("tell", "count", "hand"):
        out[f"yoy_{col}_{a}_{b}"] = dict(r=np.corrcoef(x.loc[j, col], y.loc[j, col])[0, 1], n=len(j))
for yr in (2024, 2025, 2026):
    s = R[R.season == yr]; mu = s.tell.mean(); tau2 = max(1e-9, s.tell.var() - (s.tell_se ** 2).mean())
    w = tau2 / (tau2 + s.tell_se ** 2)
    R.loc[s.index, "eb"] = mu + w * (s.tell - mu); R.loc[s.index, "eb_sd"] = np.sqrt(w * s.tell_se ** 2)
    out[f"eb_{yr}"] = dict(mu=mu, tau=np.sqrt(tau2), mean_shrink=float(w.mean()))
R.to_csv("out/tell_eb.csv", index=False)
json.dump(out, open("out/reliability.json", "w"), indent=1, default=float)
for k, v in out.items(): print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
