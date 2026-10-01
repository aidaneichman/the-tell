"""Hitter's-eye units: how often the single most likely pitch is right, count model vs count + previous pitch,
over pitches that have a previous pitch in the plate appearance. Writes out/guess.csv."""
import numpy as np, pandas as pd, json
from common import load
from model import ladder
hp = json.load(open("out/hp.json"))
df = load(); rows = []
for (pid, yr), g in df.groupby(["pitcher", "season"], sort=False):
    P, x, gm, G = ladder(g, hp); k = g.prev.to_numpy() < 8
    rows.append(dict(pitcher=pid, season=yr, name=g.player_name.iloc[0], n=len(g),
                     acc_count=(P[k, 2].argmax(1) == x[k]).mean(), acc_seq=(P[k, 3].argmax(1) == x[k]).mean()))
R = pd.DataFrame(rows); R["gain"] = R.acc_seq - R.acc_count; R.to_csv("out/guess.csv", index=False)
print(R.groupby("season")[["acc_count", "acc_seq", "gain"]].mean().round(4))
s = R[(R.season == 2025) & (R.n >= 1500)]; print(s.nlargest(3, "gain")[["name", "acc_count", "acc_seq", "gain"]])
