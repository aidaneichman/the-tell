"""Concrete cues, selected on one window and scored on a later one, so selection cannot inflate the score.
A cue is a (pitcher, count, batter hand, previous pitch) situation; its size is the shift in one next-pitch type's
share against the same pitcher, count and hand after any other previous pitch. Each pitcher's top cue (30+ pitches
each way when found) is re-scored on the later window (10+ each way). League netting: the same shift computed for
every other pitcher with that cue in the later window (10+ each way), averaged weighted by cue pitches, is
subtracted, so a cue that only reflects a league-wide habit (everyone doubles up) scores zero.
Writes out/tells_<window>.csv and out/tells_holdout.json."""
import numpy as np, pandas as pd, json
from common import load
df = load(); df = df[df.prev < 8]

def shifts(d, mn):
    """All (pitcher, count, hand, prev, typ) shifts f1 - f0 with n1, n0 >= mn."""
    N = d.groupby(["pitcher", "count", "hand", "prev", "x"]).size().unstack("x", fill_value=0).reindex(columns=range(8), fill_value=0)
    n1 = N.sum(axis=1); tot = N.groupby(level=[0, 1, 2]).transform("sum"); n0 = tot.sum(axis=1) - n1
    f1 = N.div(n1, axis=0); f0 = (tot - N).div(n0.replace(0, np.nan), axis=0)
    S = (f1 - f0).stack().rename("d").reset_index().rename(columns={"x": "typ"})
    S["n1"] = n1.reindex(S.set_index(["pitcher", "count", "hand", "prev"]).index).to_numpy()
    S["n0"] = n0.reindex(S.set_index(["pitcher", "count", "hand", "prev"]).index).to_numpy()
    S["f1"] = f1.stack().to_numpy(); S["f0"] = f0.stack().to_numpy()
    return S[(S.n1 >= mn) & (S.n0 >= mn)]

out = {}
for name, A, B in (("2025H1_2025H2", (df.season == 2025) & (df.game_date < "2025-07-01"), (df.season == 2025) & (df.game_date >= "2025-07-01")),
                   ("2025_2026", df.season == 2025, df.season == 2026)):
    Sa = shifts(df[A], 30); top = Sa.loc[Sa.groupby("pitcher").d.apply(lambda s: s.abs().idxmax())].reset_index(drop=True)
    Sb = shifts(df[B], 10); key = ["count", "hand", "prev", "typ"]
    top = top.merge(Sb[["pitcher"] + key + ["d"]].rename(columns={"d": "d_out"}), on=["pitcher"] + key, how="inner")
    Sb["wd"] = Sb.d * Sb.n1; L = Sb.groupby(key)[["wd", "n1"]].sum()
    own = Sb.set_index(["pitcher"] + key)[["wd", "n1"]]
    lk = top.set_index(key).index; ok = top.set_index(["pitcher"] + key).index
    lw = L.reindex(lk).to_numpy() - own.reindex(ok).to_numpy()                # leave this pitcher out
    top["league"] = np.where(lw[:, 1] > 0, lw[:, 0] / np.maximum(lw[:, 1], 1), np.nan)
    sg = np.sign(top.d); kept = top.d_out * sg; net = (top.d_out - top.league) * sg
    rep = top.prev == top.typ
    rng = np.random.default_rng(7); kv = kept.to_numpy(); nv = net.to_numpy()
    Bi = rng.integers(0, len(kv), size=(2000, len(kv)))                       # pitcher-level bootstrap of the held-out means
    ci_out = np.percentile(kv[Bi].mean(1), [2.5, 97.5]); ci_net = np.percentile(np.nanmean(nv[Bi], 1), [2.5, 97.5])
    out[name] = dict(n=len(top), d_in=float(top.d.abs().mean()), d_out=float(kept.mean()), same_sign=float((kept > 0).mean()),
                     d_out_lo=float(ci_out[0]), d_out_hi=float(ci_out[1]),
                     net_out=float(net.mean()), net_same_sign=float((net > 0).mean()), league_mean=float((top.league * sg).mean()),
                     net_out_lo=float(ci_net[0]), net_out_hi=float(ci_net[1]),
                     d_out_repeat=float(kept[rep & (top.d > 0)].mean()), d_out_other=float(kept[~(rep & (top.d > 0))].mean()))
    top["name"] = top.pitcher.map(df.groupby("pitcher").player_name.first())
    top.to_csv(f"out/tells_{name}.csv", index=False); print(name, {k: round(v, 3) for k, v in out[name].items()})
json.dump(out, open("out/tells_holdout.json", "w"), indent=1)
