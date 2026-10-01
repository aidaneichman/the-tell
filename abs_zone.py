"""The ABS mechanism: the called zone by season, and a dose-response on edge-dependent pitchers."""
import numpy as np, pandas as pd, json
from common import load
df = load(min_n=1)
c = df[df.description.isin(["called_strike", "ball", "blocked_ball"])].dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot"]).copy()
c["cs"] = (c.description == "called_strike").astype(float)
c["zn"] = (c.plate_z - c.sz_bot) / (c.sz_top - c.sz_bot)                  # 0 = bottom of the batter's zone, 1 = top
xb = np.linspace(-1.5, 1.5, 31); zb = np.linspace(-0.6, 1.6, 23)
out, maps = {}, {}
for yr in (2024, 2025, 2026):
    s = c[c.season == yr]
    H = np.histogram2d(s.plate_x, s.zn, [xb, zb])[0]; S = np.histogram2d(s.plate_x, s.zn, [xb, zb], weights=s.cs)[0]
    P = np.where(H >= 30, S / np.maximum(H, 1), np.nan); maps[yr] = P
    cellA = (xb[1] - xb[0]) * (zb[1] - zb[0]) * (s.sz_top - s.sz_bot).mean()
    out[f"area_{yr}"] = float(np.nansum(P > 0.5) * cellA)                # square feet called a strike more often than not
    mid = s[s.plate_x.abs() < 0.4]; side = s[(s.zn > 0.3) & (s.zn < 0.7)]
    def edge(v, y, lo, hi):
        g = np.linspace(lo, hi, 400); f = np.array([y[(v > a - .05) & (v < a + .05)].mean() for a in g]); k = np.nanargmin(np.abs(f - .5)); return float(g[k])
    out[f"top_{yr}"] = edge(mid.zn.values, mid.cs.values, 0.7, 1.4); out[f"bot_{yr}"] = edge(mid.zn.values, mid.cs.values, -0.4, 0.4)
    out[f"side_{yr}"] = edge(side.plate_x.abs().values, side.cs.values, 0.6, 1.3)
    out[f"called_strike_rate_{yr}"] = float(s.cs.mean())
np.save("out/zone_maps.npy", np.stack([maps[y] for y in (2024, 2025, 2026)]))
# dose-response: share of a pitcher's 2025 pitches within 2 inches of the zone edge
d = df[df.season == 2025].dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot"])
dx = (d.plate_x.abs() - 0.83).abs(); dz = np.minimum((d.plate_z - d.sz_top).abs(), (d.plate_z - d.sz_bot).abs())
inx = d.plate_x.abs() <= 0.83 + 0.167; inz = (d.plate_z >= d.sz_bot - 0.167) & (d.plate_z <= d.sz_top + 0.167)
edge_ = ((dx <= 0.167) & inz) | ((dz <= 0.167) & inx)
E = edge_.groupby(d.pitcher).mean().rename("edge")
T = pd.read_csv("out/tell.csv"); a = T[T.season == 2025].set_index("pitcher"); b = T[T.season == 2026].set_index("pitcher")
j = a.index.intersection(b.index).intersection(E.index)
dt = b.loc[j, "tell"] - a.loc[j, "tell"]; e = E[j]
X = np.column_stack([np.ones(len(j)), e.values]); beta, *_ = np.linalg.lstsq(X, dt.values, rcond=None)
res = dt.values - X @ beta; se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * res.var(ddof=2)))
q = pd.qcut(e, 4, labels=False)
out["dose"] = dict(n=len(j), slope=float(beta[1]), se=float(se[1]), edge_mean=float(e.mean()),
                   quartile_mean_change=[float(dt[q == k].mean()) for k in range(4)], quartile_se=[float(dt[q == k].std() / np.sqrt((q == k).sum())) for k in range(4)])
json.dump(out, open("out/abs_zone.json", "w"), indent=1)
print(json.dumps(out, indent=1))
