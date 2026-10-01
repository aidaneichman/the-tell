# The Tell

What a pitcher's previous pitch gives away about the next one: measured out of sample, re-scored on
new games, and monitored in season. MLB Statcast, 2024 to 2026. Submitted to the MIT Sloan Sports
Analytics Conference 2027 research paper competition (baseball track).

The abstract as submitted on 1 October 2026 is tag `ssac27-abstract`. Later commits may rebuild the analysis for the full paper.

## Reproduce

```
python -m pip install -r requirements.txt
./reproduce.sh
```

`reproduce.sh` runs the whole chain from the committed data to `numbers.tex` (every number in the
abstract), `table_tells.tex` (Table 1), `fig_abstract.pdf/png` (Figure 1) and `abstract.pdf`. It takes
about 4 minutes on a laptop and is deterministic (fixed seeds).

## Data

The exact pitch snapshot used is committed, one gzipped CSV per season, with the columns listed in
`fetch_full.py` (`KEEP`). Pulled from Baseball Savant's public CSV endpoint on 30 September 2026,
regular season, 20 Mar 2024 to 28 Sep 2026.

| File | Bytes | sha256 |
|---|---|---|
| `data/statcast_2024.csv.gz` | 43,494,952 | `54abb08b0dc61d4b6d2b084c2aca5b9726f8707a2ed4ba274fba509338d18e9f` |
| `data/statcast_2025.csv.gz` | 43,850,235 | `c1a90241f79a652830c172eeedfa1f4ae9322d372c44eaecfeaf56c8c307da06` |
| `data/statcast_2026.csv.gz` | 39,957,370 | `c7add063183c4354feb9fc28e42979e443454caf023a368e57fc80adb694d03d` |

Statcast reclassifies pitch types retroactively, so a fresh pull may differ slightly. To re-pull
(optional):

```
mkdir -p data
python fetch_full.py data/pitches_full.csv
python snapshot.py data/pitches_full.csv
```

Statcast redefined `sz_top`/`sz_bot` in 2026, so the zone analysis uses absolute feet.

## License

Code is MIT-licensed (see `LICENSE`). The Statcast snapshots in `data/` are MLB Advanced Media data,
redistributed as published on Baseball Savant for research reproducibility and subject to MLB's terms of use;
the MIT license does not cover them.

## Pipeline

Each file in `out/` is written by exactly one script.

| Script | Writes | What it does |
|---|---|---|
| `common.py` | | loads the snapshot; eight pitch classes; previous pitch within the plate appearance (8 = first pitch) |
| `model.py` | | nested hierarchical Dirichlet ladder: hand, count, previous pitch; leave-one-game-out. First pitches get no previous-pitch rung |
| `tune.py` | `out/hp.json` | smoothing: a0, a1, a2 fixed (2, 20, 20); b3, a3, b4 tuned on 2024 only |
| `tell.py` | `out/tell.csv`, `out/pitch_probs.csv` | TELL per pitcher-season (previous-pitch gain over pitches after the first of a PA) |
| `reliability.py` | `out/reliability.json`, `out/splithalf_indep.csv`, `out/tell_eb.csv` | independent split-half (refit on odd and on even games), year-to-year r, empirical Bayes |
| `extra.py` | `out/extra.json` | paired season-to-season change in TELL |
| `guess.py` | `out/guess.csv` | top-1 guess accuracy, count model vs count + previous pitch |
| `tells_holdout.py` | `out/tells_*.csv`, `out/tells_holdout.json` | top cue per pitcher, found on one window, scored on a later one, and net of the league |
| `table_tells.py` | `table_tells.tex`, `out/table_tells.csv` | Table 1 |
| `cost.py` | `out/cost.json` | outcome regressions, pitcher-season and type x count x hand fixed effects, velocity; variant with the previous pitch's result |
| `eprocess.py` | | the original unstratified alarm, kept only for the audit in `monitor.py` |
| `monitor.py` | `out/monitor_audit*.{json,csv}` | stratified alarm on 2025, plus the audit: unstratified vs stratified, real vs cue-free control |
| `monitor_sim.py` | `out/monitor_sim.json` | stratified alarm on simulated seasons where the count responds to each pitch |
| `abs_zone.py` | `out/abs_zone_abs.json` | called-zone edges in absolute feet by season |
| `mknumbers.py` | `numbers.tex` | every number in the abstract |
| `fig_abstract.py` | `fig_abstract.pdf/png` | Figure 1 |

## Where each number in the abstract comes from

| Abstract | Value | Source |
|---|---|---|
| pitches, pitcher-seasons | 2,049,352; 1,793 | `out/tell.csv` (sum of `n`, rows) |
| hand, count, previous pitch (millibits, 2025 mean over pitcher-seasons) | 102, 58, 12.8 | `out/tell.csv` `hand`, `count`, `tell` |
| guess accuracy gain; top pitcher | 0.8 pts; Dylan Cease 7 | `out/guess.csv` (2025; top among 1,500+ pitches) |
| top cue found, held out, net of league, same direction | 26, 10, 7, 78% | `out/tells_holdout.json["2025H1_2025H2"]` |
| held-out 95% CI (pitcher bootstrap); repeat vs other cues | 7–13; 14, 8 | `out/tells_holdout.json["2025H1_2025H2"]` (`d_out_lo/hi`, `d_out_repeat/other`) |
| Table 1 mean size found, held out | 44, 29 | `out/table_tells.csv` |
| split-half reliability; year-to-year r; count usage | 0.75; 0.22; 0.62 | `out/reliability.json` `split_indep_2025.sb`, `yoy_tell_2025_2026`, `yoy_count_2025_2026` |
| whiffs, runs per 100 pitches (10th to 90th percentile) | 1.1 (t 6.8); 0.26 (t 4.7) | `out/cost.json` `base`, scaled by `span` |
| same, previous pitch's result controlled | 0.9 (t 6.2); 0.24 (t 4.4) | `out/cost.json` `prevout` |
| alarm fired; median pitches; cue-free controls | 25%; 436; 1% | `out/monitor_audit.json` `strat_*` |
| alarm, endogenous-count simulations | 1.6% | `out/monitor_sim.json` `rate` |
| unstratified alarm on cue-free controls (audit) | 15% | `out/monitor_audit.json` `orig_ctrl_fired` |
| zone top change 2025 to 2026; 2024 to 2025 | 1.2 in; 0.2 in | `out/abs_zone_abs.json` `top_ft` |
| TELL change 2025 to 2026 (95% CI); 2024 to 2025 | +0.01 (-1.86, +1.88); -0.49 | `out/extra.json` |

The macros in `numbers.tex` carry these values into `abstract.tex`; no number in the abstract is typed by hand.
