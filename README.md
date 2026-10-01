# The Tell

What a pitcher's previous pitch gives away about the next one, measured out of sample, re-scored on
new games, and monitored in season with a false-alarm guarantee. MLB Statcast, 2024 to 2026.

Submitted to the MIT Sloan Sports Analytics Conference 2027 research paper competition (baseball).

## Reproduce

Requires Python 3.11+ with numpy, pandas, scipy and matplotlib.

```
python fetch_full.py            # public Baseball Savant CSV endpoint, about 2.1M pitches, writes data/
python tune.py                  # smoothing tuned on 2024 only
python tell.py                  # TELL per pitcher-season, leave-one-game-out
python reliability.py
python tells_holdout.py         # concrete tells: select on one window, score on a later one
python guess.py                 # hitter guess accuracy
python table_tells.py           # Table 1
python cost.py                  # whiff, run value and xwOBA regressions with two-way fixed effects
python monitor.py               # original whole-outing alarm (reads count structure; kept for the audit)
python monitor2.py              # stratified alarm (within count x hand, within-PA pairs) + count-only negative control
python abs_zone.py              # ABS called-zone change, 2024-25 placebo
python fig_abstract.py          # Figure 1
```

Statcast redefined `sz_top`/`sz_bot` in 2026. Zone analyses here use absolute feet, not the
normalised bounds.

| File | What it does |
|---|---|
| `common.py` | loads pitches, pitch-type classes, previous pitch within the plate appearance |
| `model.py` | nested hierarchical Dirichlet predictive ladder: hand, count, previous pitch |
| `eprocess.py` | game-block permutation e-process |
| `out/` | small result files used by the figures |
