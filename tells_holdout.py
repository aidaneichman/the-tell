"""Concrete tells, selected on one window and scored on a later one, so selection bias cannot inflate them.
A tell is a (pitcher, count, previous pitch, next type) cell; its size is the shift in the next type's
frequency against the same pitcher, count and batter hand after any other previous pitch."""
import numpy as np, pandas as pd, json
from common import load, NAME
df = load(); df = df[df.prev < 8]
def cells(d, mn):
    out = []
    for (pid, c, h), s in d.groupby(["pitcher", "count", "hand"]):
        tot = np.bincount(s.x, minlength=8); ntot = len(s)
        for pv, t in s.groupby("prev"):
            n1 = len(t); n0 = ntot - n1
            if n1 < mn or n0 < mn: continue
            f1 = np.bincount(t.x, minlength=8) / n1; f0 = (tot - np.bincount(t.x, minlength=8)) / n0
            k = np.argmax(np.abs(f1 - f0))
            out.append((pid, c, h, pv, k, n1, f1[k], f0[k], f1[k] - f0[k]))
    return pd.DataFrame(out, columns=["pitcher", "count", "hand", "prev", "typ", "n1", "f1", "f0", "d"])
def score(d, sel):
    r = []
    for row in sel.itertuples():
        s = d[(d.pitcher == row.pitcher) & (d["count"] == row.count) & (d.hand == row.hand)]
        a = s[s.prev == row.prev]; b = s[s.prev != row.prev]
        r.append(((a.x == row.typ).mean() - (b.x == row.typ).mean()) if len(a) >= 10 and len(b) >= 10 else np.nan)
    return np.array(r)
out = {}
for name, A, B in (("2025H1->2025H2", (df.season == 2025) & (df.game_date < "2025-07-01"), (df.season == 2025) & (df.game_date >= "2025-07-01")),
                   ("2025->2026", df.season == 2025, df.season == 2026)):
    a = df[A]; b = df[B]
    C = cells(a, 30); top = C.loc[C.groupby("pitcher").d.apply(lambda s: s.abs().idxmax())]
    top["d_out"] = score(b, top); top = top.dropna(subset=["d_out"])
    same = np.sign(top.d) == np.sign(top.d_out)
    out[name] = dict(n=len(top), d_in=float(top.d.abs().mean()), d_out=float((top.d_out * np.sign(top.d)).mean()),
                     same_sign=float(same.mean()), half_kept=float(((top.d_out * np.sign(top.d)) >= top.d.abs() / 2).mean()))
    top["name"] = [df.loc[df.pitcher == p, "player_name"].iloc[0] for p in top.pitcher]
    top.to_csv(f"out/tells_{name.replace('>', '').replace('-', '_')}.csv", index=False)
    print(name, out[name])
json.dump(out, open("out/tells_holdout.json", "w"), indent=1)
# league repeat rate: P(same type as previous) versus what the count-and-hand mix alone implies
d = df[df.season == 2025]
rep = (d.x == d.prev).mean()
exp = []
for _, s in d.groupby(["pitcher", "count", "hand"]):
    f = np.bincount(s.x, minlength=8) / len(s); exp.append((f[s.prev.to_numpy()]).sum())
print("2025 repeat rate", round(rep, 4), "expected from count mix", round(sum(exp) / len(d), 4))
