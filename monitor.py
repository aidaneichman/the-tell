"""The in-season monitor: game-block permutation e-process (valid under within-game exchangeability), 2025."""
import numpy as np, pandas as pd, json, sys
from eprocess import game_eprocess
from common import load
df = load(); d = df[df.season == 2025]
rng = np.random.default_rng(11); rows, paths = [], {}
for pid, g in d.groupby("pitcher", sort=False):
    x = g.x.to_numpy(np.int64); games = pd.factorize(g.game_pk)[0]
    path = game_eprocess(x, games, 199, rng)
    dates = g.groupby(games).game_date.first().to_numpy()
    hit = np.nonzero(path >= np.log(20))[0]
    rows.append(dict(pitcher=pid, name=g.player_name.iloc[0], n=len(x), games=len(path), fired=len(hit) > 0,
                     fire_game=int(hit[0]) + 1 if len(hit) else -1, fire_date=dates[hit[0]] if len(hit) else "",
                     pitches_at_fire=int((games <= hit[0]).sum()) if len(hit) else -1, logE_final=path[-1]))
    paths[int(pid)] = dict(dates=list(dates), logE=path.round(4).tolist())
M = pd.DataFrame(rows); M.to_csv("out/monitor_2025.csv", index=False); json.dump(paths, open("out/monitor_paths.json", "w"))
print("fired:", M.fired.mean().round(3), "median pitches at fire:", M.pitches_at_fire[M.fired].median(), "median game:", M.fire_game[M.fired].median())
print(M[M.fired].sort_values("pitches_at_fire").head(10)[["name", "fire_date", "pitches_at_fire", "n"]].to_string(index=False))
