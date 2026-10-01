#!/usr/bin/env bash
# Regenerates every number in abstract.tex, Table 1 and Figure 1 from the committed data snapshot.
# Usage: ./reproduce.sh            (uses python3; set PYTHON=/path/to/python to override)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p out
step() { echo "== $1"; shift; "$PY" "$@"; }
step "tune (2024 only)"            tune.py
step "TELL ladder"                 tell.py
step "reliability"                 reliability.py
step "paired season changes"       extra.py
step "guess accuracy"              guess.py
step "held-out cues"               tells_holdout.py
step "Table 1"                     table_tells.py
step "outcome regressions"         cost.py
step "alarm + audit"               monitor.py
step "alarm, endogenous count"     monitor_sim.py
step "called zone, absolute feet"  abs_zone.py
step "numbers.tex"                 mknumbers.py
step "Figure 1"                    fig_abstract.py
if command -v tectonic >/dev/null; then SOURCE_DATE_EPOCH=1790812800 tectonic -X compile -Z deterministic-mode abstract.tex; else echo "tectonic not found: compile abstract.tex with any LaTeX engine"; fi
echo "done"
