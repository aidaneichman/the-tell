"""Hitter's-eye units: how often the single most likely pitch is right, count model vs count + previous pitch.
Also the strongest concrete tell per pitcher: the (count, previous pitch) cell where the previous pitch moves
the probability of one pitch type the most, out of sample."""
import numpy as np, pandas as pd, json
from common import load, NAME
from model import ladder
hp = json.load(open("out/hp.json"))
df = load(); df = df[df.season.isin([2024, 2025, 2026])]
rows, cells = [], []
for (pid, yr), g in df.groupby(["pitcher", "season"], sort=False):
    P, x, gm, G = ladder(g, hp)
    a2 = (P[:, 2].argmax(1) == x).mean(); a3 = (P[:, 3].argmax(1) == x).mean()
    big = np.abs(P[:, 3] - P[:, 2]).max(1)          # largest move in any type's probability
    rows.append(dict(pitcher=pid, season=yr, name=g.player_name.iloc[0], n=len(g), acc_count=a2, acc_seq=a3,
                     share_move20=(big >= .20).mean(), share_top60_seq=(P[:, 3].max(1) >= .6).mean(),
                     share_top60_cnt=(P[:, 2].max(1) >= .6).mean()))
    if yr == 2025 and len(g) >= 1500:
        # empirical cells: (count, prev) with >= 40 pitches; compare realized freq vs same count, other prev
        h = pd.DataFrame({"c": g["count"].to_numpy(), "pv": g.prev.to_numpy(), "x": x})
        for (c, pv), s in h[h.pv < 8].groupby(["c", "pv"]):
            if len(s) < 40: continue
            o = h[(h.c == c) & (h.pv != pv)]
            if len(o) < 40: continue
            for k in range(8):
                f1 = (s.x == k).mean(); f0 = (o.x == k).mean()
                cells.append(dict(pitcher=pid, name=g.player_name.iloc[0], count=f"{c//3}-{c%3}", prev=NAME[pv], typ=NAME[k],
                                  n1=len(s), n0=len(o), f1=f1, f0=f0, d=f1 - f0))
R = pd.DataFrame(rows); R.to_csv("out/guess.csv", index=False)
C = pd.DataFrame(cells); C.to_csv("out/cells_2025.csv", index=False)
print(R.groupby("season")[["acc_count", "acc_seq", "share_move20", "share_top60_cnt", "share_top60_seq"]].mean().round(4))
