"""Table 1: the ten largest 2025 tells (at least 50 pitches after the cue) re-scored on 2026 games."""
import pandas as pd, numpy as np
from common import load, NAME
FULL = {"FF": "4-seam", "SI": "sinker", "FC": "cutter", "SL": "slider", "ST": "sweeper", "CU": "curve", "CH": "change", "FS": "splitter"}
df = load(); d = df[(df.season == 2026) & (df.prev < 8)]
t = pd.read_csv("out/tells_2025_2026.csv"); t = t[t.n1 >= 50].reindex(t[t.n1 >= 50].d.abs().sort_values(ascending=False).index).head(30)
rows = []
for r in t.itertuples():
    s = d[(d.pitcher == r.pitcher) & (d["count"] == r.count) & (d.hand == r.hand)]
    a = s[s.prev == r.prev]; b = s[s.prev != r.prev]
    nm = r.name.split(", "); rows.append(dict(pitcher=f"{nm[1]} {nm[0]}", situation=f"{r.count//3}-{r.count%3} vs {'LHB' if r.hand else 'RHB'}, after {FULL[NAME[r.prev]]}",
        next=FULL[NAME[r.typ]], d25=round(100 * r.f1) - round(100 * r.f0), d26=round(100 * (a.x == r.typ).mean()) - round(100 * (b.x == r.typ).mean()),
        n26=len(a), kept=np.sign(r.d) == np.sign((a.x == r.typ).mean() - (b.x == r.typ).mean())))
T = pd.DataFrame(rows); T = T[T.n26 >= 25].head(10); T.to_csv("out/table_tells.csv", index=False); print(T.to_string())
with open("table_tells.tex", "w") as f:
    for r in T.itertuples(): f.write(f"{r.pitcher} & {r.situation} & {r.next} & ${r.d25:+d}$ & ${r.d26:+d}$ & {r.n26} \\\\\n")
sz = T.d25.abs(); kept = T.d26 * np.sign(T.d25)
print("mean found", sz.mean(), "mean held out (same direction)", kept.mean())
