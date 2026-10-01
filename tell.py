"""Per pitcher-season: TELL and the ladder, with game-bootstrap SEs, split-half halves, and per-pitch probabilities."""
import numpy as np, pandas as pd, json
from common import load
from model import ladder
hp = json.load(open("out/hp.json")); LN2 = np.log(2)
df = load(); df.to_parquet("out/pitches.parquet") if False else None
rows, pp = [], []
rng = np.random.default_rng(0)
for (pid, yr), g in df.groupby(["pitcher", "season"], sort=False):
    P, x, gm, G = ladder(g, hp)
    pa = P[np.arange(len(x)), :, x]                      # prob of the pitch thrown, each rung
    bits = -np.log(pa) / LN2
    gain = bits[:, :-1] - bits[:, 1:]                    # hand, count, prev type, prev location
    ng = np.bincount(gm, minlength=G)
    pg = np.vstack([np.bincount(gm, weights=gain[:, j], minlength=G) for j in range(3)]).T
    Bi = rng.integers(0, G, size=(500, G))
    boot = pg[Bi].sum(1) / ng[Bi].sum(1)[:, None]
    r = dict(pitcher=pid, season=yr, name=g.player_name.iloc[0], throws=g.p_throws.iloc[0], n=len(g), games=G,
             m=len(np.unique(x)), H=bits[:, 0].mean(), bits_sit=bits[:, 2].mean(), bits_seq=bits[:, 3].mean())
    for j, nm in enumerate(["hand", "count", "tell"]):
        r[nm] = gain[:, j].mean(); r[f"{nm}_se"] = boot[:, j].std(ddof=1)
    for par in (0, 1): r[f"tell_h{par}"] = gain[(gm % 2) == par, 2].mean()
    rows.append(r)
    pp.append(pd.DataFrame({"row": g.index.to_numpy(), "logp_sit": np.log(pa[:, 2]), "logp_seq": np.log(pa[:, 3]),
                            "top_seq": P[:, 3, :].max(1)}))
R = pd.DataFrame(rows); R.to_csv("out/tell.csv", index=False)
pd.concat(pp).to_csv("out/pitch_probs.csv", index=False)
print(R.groupby("season")[["n", "H", "hand", "count", "tell"]].median().round(4))
print("share of seasons with tell > 2 se:", R.groupby("season").apply(lambda d: (d.tell > 2 * d.tell_se).mean()).round(3).to_dict())
