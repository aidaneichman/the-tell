import numpy as np, pandas as pd, json
R = pd.read_csv("out/tell.csv"); out = {}
for yr in (2024, 2025, 2026):
    s = R[R.season == yr]
    r = np.corrcoef(s.tell_h0, s.tell_h1)[0, 1]; out[f"split_{yr}"] = dict(r_half=r, spearman_brown=2 * r / (1 + r), n=len(s))
for a, b in ((2024, 2025), (2025, 2026)):
    x = R[R.season == a].set_index("pitcher"); y = R[R.season == b].set_index("pitcher"); j = x.index.intersection(y.index)
    for col in ("tell", "count", "hand"):
        out[f"yoy_{col}_{a}_{b}"] = dict(r=np.corrcoef(x.loc[j, col], y.loc[j, col])[0, 1], n=len(j))
# empirical Bayes shrinkage per season
for yr in (2024, 2025, 2026):
    s = R[R.season == yr]; mu = s.tell.mean(); tau2 = max(1e-9, s.tell.var() - (s.tell_se ** 2).mean())
    w = tau2 / (tau2 + s.tell_se ** 2)
    R.loc[s.index, "eb"] = mu + w * (s.tell - mu); R.loc[s.index, "eb_sd"] = np.sqrt(w * s.tell_se ** 2)
    out[f"eb_{yr}"] = dict(mu=mu, tau=np.sqrt(tau2), mean_shrink=float(w.mean()))
R.to_csv("out/tell_eb.csv", index=False)
json.dump(out, open("out/reliability.json", "w"), indent=1, default=float)
for k, v in out.items(): print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
L = R[(R.season == 2025) & (R.n >= 1500)].sort_values("eb", ascending=False)
print("\nmost predictable 2025 (>=1500 pitches)"); print(L.head(10)[["name", "n", "tell", "eb", "eb_sd"]].round(4).to_string(index=False))
print("\nleast predictable 2025"); print(L.tail(10)[["name", "n", "tell", "eb", "eb_sd"]].round(4).to_string(index=False))
