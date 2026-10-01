import numpy as np, itertools, json
from common import load
from model import ladder, HP
df = load(); tr = df[df.season == 2024]
keys = tr.groupby("pitcher").size().sample(250, random_state=0).index
sub = [g for p, g in tr[tr.pitcher.isin(keys)].groupby("pitcher")]
def loss(hp):
    tot = np.zeros(5); n = 0
    for g in sub:
        P, x, gm, G = ladder(g, hp); tot += -np.log(P[np.arange(len(x)), :, x]).sum(0); n += len(x)
    return tot / n / np.log(2)
best = dict(HP, a1=20.0, a2=20.0, b3=300.0, a3=100.0)
for key, grid in (("b3", [300, 1000, 3000]), ("a3", [30, 100, 300]), ("b4", [1000, 3000, 1e9])):
    res = {}
    for v in grid:
        hp = dict(best); hp[key] = v; res[v] = loss(hp)
    lvl = {"a1": 1, "a2": 2, "b3": 3, "a3": 3, "b4": 4}[key]
    best[key] = min(res, key=lambda v: res[v][lvl])
    print(key, {v: round(r[lvl], 5) for v, r in res.items()}, "->", best[key], flush=True)
L = loss(best); print("ladder bits:", L.round(5), "gains:", (L[:-1] - L[1:]).round(5))
json.dump(best, open("out/hp.json", "w"))
