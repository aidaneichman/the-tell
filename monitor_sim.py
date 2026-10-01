"""Endogenous-count check of the stratified alarm. For each 2025 pitcher, simulate a season under the null
(pitch type drawn from that pitcher's P(type | count, hand), with no dependence on the previous pitch) in which
the count responds to each pitch: the outcome is drawn from the league's 2025 P(outcome | type, count, hand),
outcome in {ball, strike, foul, ball in play}. Real game lengths (plate appearances per outing) and real
batter-hand sequences are kept. The stratified alarm is then run on each simulated season. Writes out/monitor_sim.json."""
import numpy as np, pandas as pd, json
from multiprocessing import Pool
from common import load
from monitor import strat_eprocess, B, LOG20
K = 4                                                     # 0 ball, 1 strike, 2 foul, 3 in play
def outcome_cat(desc):
    return np.select([desc.isin(["ball", "blocked_ball", "pitchout", "intent_ball"]),
                      desc.isin(["called_strike", "swinging_strike", "swinging_strike_blocked", "missed_bunt"]),
                      desc.isin(["foul", "foul_tip", "foul_bunt", "bunt_foul_tip"])], [0, 1, 2], 3)
def simulate(ptype, pas, rng, OUT):
    xs, cs, hs, npa, gs = [], [], [], [], []
    for gi, hands in enumerate(pas):
        for h in hands:
            b = s = 0; first = True
            while True:
                c = b * 3 + s; t = rng.choice(8, p=ptype[c, h]); o = rng.choice(K, p=OUT[t, c, h])
                xs.append(t); cs.append(c); hs.append(h); npa.append(first); gs.append(gi); first = False
                if o == 0:
                    b += 1
                    if b == 4: break
                elif o == 1:
                    s += 1
                    if s == 3: break
                elif o == 2: s = min(s + 1, 2)
                else: break
    x = np.array(xs, np.int64); return x, np.array(gs), (np.array(cs) * 2 + np.array(hs)).astype(np.int64), np.array(npa)
def run(args):
    pid, ptype, pas, OUT, seed, reps = args
    rng = np.random.default_rng(seed); fired = 0
    for _ in range(reps):
        x, g, st, npa = simulate(ptype, pas, rng, OUT)
        fired += int((strat_eprocess(x, g, st, npa, B, rng) >= LOG20).any())
    return fired, reps
if __name__ == "__main__":
    df = load(); d = df[df.season == 2025].copy(); d["o"] = outcome_cat(d.description)
    C = np.zeros((8, 12, 2, K)); np.add.at(C, (d.x, d["count"], d.hand, d.o), 1)
    OUT = (C + 1) / (C + 1).sum(-1, keepdims=True)
    jobs = []
    for i, (pid, g) in enumerate(d.groupby("pitcher", sort=False)):
        T = np.zeros((12, 2, 8)); np.add.at(T, (g["count"], g.hand, g.x), 1)
        m = np.bincount(g.x, minlength=8) + 0.5; m = m / m.sum()
        ptype = (T + 20 * m) / (T.sum(-1, keepdims=True) + 20)
        first = g.prev.to_numpy() == 8
        pas = [g.hand.to_numpy()[first & (g.game_pk.to_numpy() == gp)] for gp in pd.unique(g.game_pk)]
        jobs.append((int(pid), ptype, pas, OUT, 5000 + i, 2))
    with Pool(8) as P: R = P.map(run, jobs, chunksize=4)
    f = sum(r[0] for r in R); n = sum(r[1] for r in R)
    out = dict(sims=n, pitchers=len(R), fired=f, rate=f / n, se=float(np.sqrt(f / n * (1 - f / n) / n)))
    json.dump(out, open("out/monitor_sim.json", "w"), indent=1); print(out)
