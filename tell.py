"""Per pitcher-season: the predictive ladder, with game-bootstrap SEs, split-half halves and per-pitch probabilities.
TELL is the previous-pitch gain averaged over pitches that have a previous pitch in the same plate appearance
(prev < 8). First pitches carry no previous-pitch information and model.py sets their previous-pitch rung equal
to the count rung. Hand and count gains are averaged over all pitches."""
import numpy as np, pandas as pd, json
from common import load
from model import ladder
hp = json.load(open("out/hp.json")); LN2 = np.log(2)
df = load()
rows, pp = [], []
rng = np.random.default_rng(0)
for (pid, yr), g in df.groupby(["pitcher", "season"], sort=False):
    P, x, gm, G = ladder(g, hp)
    pa = P[np.arange(len(x)), :, x]
    bits = -np.log(pa) / LN2
    gain = bits[:, :-1] - bits[:, 1:]                    # hand, count, prev type, prev location
    w = np.column_stack([np.ones(len(x)), np.ones(len(x)), (g.prev.to_numpy() < 8).astype(float)])
    ng = np.vstack([np.bincount(gm, weights=w[:, j], minlength=G) for j in range(3)]).T
    pg = np.vstack([np.bincount(gm, weights=gain[:, j] * w[:, j], minlength=G) for j in range(3)]).T
    Bi = rng.integers(0, G, size=(500, G))
    boot = pg[Bi].sum(1) / ng[Bi].sum(1)
    r = dict(pitcher=pid, season=yr, name=g.player_name.iloc[0], throws=g.p_throws.iloc[0], n=len(g), games=G,
             m=len(np.unique(x)), H=bits[:, 0].mean(), bits_sit=bits[:, 2].mean(), bits_seq=bits[:, 3].mean())
    for j, nm in enumerate(["hand", "count", "tell"]):
        r[nm] = pg[:, j].sum() / ng[:, j].sum(); r[f"{nm}_se"] = boot[:, j].std(ddof=1)
    for par in (0, 1):
        m = (gm % 2) == par; r[f"tell_h{par}"] = (gain[m, 2] * w[m, 2]).sum() / w[m, 2].sum()
    rows.append(r)
    pp.append(pd.DataFrame({"row": g.index.to_numpy(), "logp_sit": np.log(pa[:, 2]), "logp_seq": np.log(pa[:, 3])}))
R = pd.DataFrame(rows); R.to_csv("out/tell.csv", index=False)
pd.concat(pp).to_csv("out/pitch_probs.csv", index=False)
print(R.groupby("season")[["n", "H", "hand", "count", "tell"]].mean().round(4))
