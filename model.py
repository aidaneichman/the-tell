"""Leave-one-game-out predictive distributions for a pitcher-season, as a ladder of nested models.
Every level backs off to the one below; the previous-pitch and location effects are pooled across
counts as multiplicative tilts, so they are estimable from a season of data."""
import numpy as np, pandas as pd
D = 8
def logo_tables(keys, x, gm, K, G):
    T = np.zeros((K, D)); np.add.at(T, (keys, x), 1)
    Gt = np.zeros((G, K, D)); np.add.at(Gt, (gm, keys, x), 1)
    return T[keys] - Gt[gm, keys]                                  # (n, D) leave-one-game-out counts
def blend(counts, prior, a):
    return (counts + a * prior) / (counts.sum(1, keepdims=True) + a)
def normalize(v): return v / v.sum(1, keepdims=True)
def ladder(g, hp):
    x = g.x.to_numpy().astype(int); gm = pd.factorize(g.game_pk)[0]; G = gm.max() + 1; n = len(x)
    h = g.hand.to_numpy().astype(int); c = g["count"].to_numpy().astype(int)
    pv = g.prev.to_numpy().astype(int); pz = g.prevzone.to_numpy().astype(int)
    unif = np.full((n, D), 1 / D)
    p0 = blend(logo_tables(np.zeros(n, int), x, gm, 1, G), unif, hp["a0"])
    p1 = blend(logo_tables(h, x, gm, 2, G), p0, hp["a1"])
    p2 = blend(logo_tables(h * 12 + c, x, gm, 24, G), p1, hp["a2"])
    tilt = blend(logo_tables(pv, x, gm, 9, G), p0, hp["b3"]) / p0          # previous-pitch tilt, pooled over counts
    p3 = blend(logo_tables((h * 12 + c) * 9 + pv, x, gm, 24 * 9, G), normalize(p2 * tilt), hp["a3"])
    pvz = logo_tables(pv * 3 + pz, x, gm, 27, G); pvonly = logo_tables(pv, x, gm, 9, G)
    tilt4 = blend(pvz, blend(pvonly, p0, hp["b3"]), hp["b4"]) / blend(pvonly, p0, hp["b3"])
    p4 = normalize(p3 * tilt4)
    P = np.stack([p0, p1, p2, p3, p4], 1)                                 # (n, 5, D)
    return P, x, gm, G
HP = dict(a0=2.0, a1=20.0, a2=20.0, b3=30.0, a3=30.0, b4=30.0)
