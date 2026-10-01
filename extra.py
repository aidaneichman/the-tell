"""Paired season-to-season change in TELL (pitchers in both seasons, normal 95% CI). 2025->2026 spans the first
ABS-challenge season; 2024->2025 is the comparison. Writes out/extra.json."""
import numpy as np, pandas as pd, json
T = pd.read_csv("out/tell.csv"); out = {}
for a, b in ((2024, 2025), (2025, 2026)):
    x = T[T.season == a].set_index("pitcher"); y = T[T.season == b].set_index("pitcher"); j = x.index.intersection(y.index)
    d = y.loc[j, "tell"] - x.loc[j, "tell"]; se = d.std(ddof=1) / np.sqrt(len(d))
    out[f"tell_{a}_{b}"] = dict(n=len(j), base=float(x.loc[j, "tell"].mean()), diff=float(d.mean()), lo=float(d.mean() - 1.96 * se), hi=float(d.mean() + 1.96 * se))
json.dump(out, open("out/extra.json", "w"), indent=1); print(json.dumps(out, indent=1))
