"""The called zone by season in absolute feet (Statcast redefined sz_top/sz_bot in 2026, so normalised
coordinates would fake a shift). Edge = where the called-strike rate crosses 50%: top and bottom on pitches
within 0.4 ft of the plate's centre line, side on pitches 2.0-3.0 ft high. Writes out/abs_zone_abs.json."""
import numpy as np, pandas as pd, json
from common import load
df = load(min_n=1)
c = df[df.description.isin(["called_strike", "ball", "blocked_ball"])].dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot"]).copy()
c["cs"] = (c.description == "called_strike").astype(float)
def edge(v, y, lo, hi):
    g = np.linspace(lo, hi, 500); f = np.array([y[(v > a - .04) & (v < a + .04)].mean() for a in g]); return float(g[np.nanargmin(np.abs(f - .5))])
out = {}
for yr in (2024, 2025, 2026):
    s = c[c.season == yr]; mid = s[s.plate_x.abs() < 0.4]; side = s[(s.plate_z > 2.0) & (s.plate_z < 3.0)]
    out[yr] = dict(top_ft=edge(mid.plate_z.values, mid.cs.values, 2.8, 4.2), bot_ft=edge(mid.plate_z.values, mid.cs.values, 1.0, 2.2),
                   side_ft=edge(side.plate_x.abs().values, side.cs.values, 0.6, 1.3), n_called=int(len(s)))
json.dump(out, open("out/abs_zone_abs.json", "w"), indent=1); print(json.dumps(out, indent=1))
