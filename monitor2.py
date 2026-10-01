"""Alarm audit. The original alarm permutes whole outings, so count structure alone can fire it.
Variants: original vs stratified (permute within count x hand inside the outing, score only within-PA
pairs by a previous-pitch tilt fit on earlier outings), each on real data and on a negative control
in which pitch types are shuffled within game x count x hand (no sequencing beyond count and hand)."""
import numpy as np, pandas as pd, json
from multiprocessing import Pool
from common import load
from eprocess import game_eprocess
D, B, A, LOG20 = 8, 199, 20.0, np.log(20)

def strat_eprocess(x, games, strata, newpa, B, rng):
    logE, path = 0.0, []
    N = np.zeros((D, D)); m = np.ones(D)
    gs = np.unique(games)
    for k, g in enumerate(gs):
        idx = np.nonzero(games == g)[0]; blk = x[idx]; st = strata[idx]; np_ = newpa[idx]
        n = len(blk); pair = ~np_; pair[0] = False
        if k > 0 and pair.sum() >= 2:
            mm = m / m.sum(); Q = (N + A * mm) / (N.sum(1, keepdims=True) + A); lt = np.log(Q / mm)
            key = st[None, :] + rng.random((B, n)); order = np.argsort(key, 1, kind="stable"); pos = np.argsort(st, kind="stable")
            Xb = np.empty((B + 1, n), dtype=blk.dtype); Xb[0] = blk; Xb[1:, pos] = blk[order]
            ll = (lt[Xb[:, :-1], Xb[:, 1:]] * pair[None, 1:]).sum(1)
            mx = ll.max(); logE += np.log(B + 1) + ll[0] - (mx + np.log(np.exp(ll - mx).sum()))
        path.append(logE)
        pr = np.nonzero(pair)[0]; np.add.at(N, (blk[pr - 1], blk[pr]), 1); np.add.at(m, blk, 1)
    return np.array(path)

def control(x, games, strata, rng):
    key = (games.astype(np.int64) * 100 + strata).astype(float) + rng.random(len(x))
    blocks = games.astype(np.int64) * 100 + strata
    out = np.empty_like(x); out[np.argsort(blocks, kind="stable")] = x[np.argsort(key, kind="stable")]
    return out

def run(args):
    pid, x, games, strata, newpa, seed = args
    rng = np.random.default_rng(seed); xc = control(x, games, strata, rng)
    r = {"pitcher": pid, "n": len(x)}
    for tag, xx in (("real", x), ("ctrl", xc)):
        for meth in ("orig", "strat"):
            p = game_eprocess(xx, games, B, rng) if meth == "orig" else strat_eprocess(xx, games, strata, newpa, B, rng)
            hit = np.nonzero(p >= LOG20)[0]
            r[f"{meth}_{tag}_fired"] = len(hit) > 0
            r[f"{meth}_{tag}_pitches"] = int(np.isin(games, np.unique(games)[: hit[0] + 1]).sum()) if len(hit) else -1
            if tag == "real": r[f"{meth}_path"] = p.round(4).tolist()
    return r

if __name__ == "__main__":
    df = load(); d = df[df.season == 2025]
    jobs = []
    for i, (pid, g) in enumerate(d.groupby("pitcher", sort=False)):
        gm = pd.factorize(g.game_pk)[0]
        jobs.append((int(pid), g.x.to_numpy(np.int64), gm, (g["count"].to_numpy() * 2 + g.hand.to_numpy()).astype(np.int64),
                     (g.prev.to_numpy() == 8), 1000 + i))
    with Pool(8) as P: R = P.map(run, jobs, chunksize=4)
    names = d.groupby("pitcher").player_name.first(); dates = d.groupby(["pitcher"]).apply(lambda g: g.groupby(pd.factorize(g.game_pk)[0]).game_date.first().tolist())
    paths = {r["pitcher"]: dict(dates=dates[r["pitcher"]], orig=r.pop("orig_path"), strat=r.pop("strat_path")) for r in R}
    M = pd.DataFrame(R); M["name"] = M.pitcher.map(names); M.to_csv("out/monitor_audit_2025.csv", index=False)
    json.dump(paths, open("out/monitor_audit_paths.json", "w"))
    S = {c: float(M[c].mean()) for c in M.columns if c.endswith("fired")}
    for meth in ("orig", "strat"):
        f = M[M[f"{meth}_real_fired"]]; S[f"{meth}_real_median_pitches"] = float(f[f"{meth}_real_pitches"].median()) if len(f) else None
    S["n"] = len(M); json.dump(S, open("out/monitor_audit.json", "w"), indent=1); print(json.dumps(S, indent=1))
