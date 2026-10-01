#!/usr/bin/env python3
"""Pull regular-season Statcast pitches 2024-2026 with location, game state and outcome columns."""
import csv, io, sys, time, urllib.request, datetime as dt
KEEP = ["game_pk","game_date","pitcher","player_name","batter","at_bat_number","pitch_number",
        "pitch_type","balls","strikes","outs_when_up","on_1b","on_2b","on_3b","inning","inning_topbot",
        "p_throws","stand","release_speed","release_spin_rate","pfx_x","pfx_z","plate_x","plate_z",
        "sz_top","sz_bot","zone","type","description","events","launch_speed","launch_angle",
        "estimated_woba_using_speedangle","woba_value","woba_denom","delta_run_exp","bat_speed",
        "home_score","away_score","fielder_2"]
SEASONS = {2024: ("2024-03-20","2024-09-30"), 2025: ("2025-03-18","2025-09-28"), 2026: ("2026-03-25","2026-09-28")}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
URL = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=R%7C&type=details"
       "&player_type=pitcher&game_date_gt={a}&game_date_lt={b}")
def chunk(a, b, days=3):
    a = dt.date.fromisoformat(a); b = dt.date.fromisoformat(b)
    while a <= b:
        e = min(a + dt.timedelta(days=days-1), b); yield a.isoformat(), e.isoformat(); a = e + dt.timedelta(days=1)
out = open(sys.argv[1], "w", newline=""); w = csv.writer(out); w.writerow(KEEP); total = 0
for yr, (a, b) in SEASONS.items():
    for lo, hi in chunk(a, b):
        for attempt in range(6):
            try:
                raw = urllib.request.urlopen(urllib.request.Request(URL.format(a=lo, b=hi), headers=UA), timeout=240).read().decode("utf-8-sig"); break
            except Exception:
                if attempt == 5: raise
                time.sleep(5 * (attempt + 1))
        rd = list(csv.DictReader(io.StringIO(raw)))
        if len(rd) >= 24900: print(f"WARN {lo}..{hi} hit row cap", flush=True)
        for r in rd:
            if r.get("pitch_type"): w.writerow([r.get(k, "") for k in KEEP])
        total += len(rd); out.flush(); print(f"{lo}..{hi} {len(rd)} total {total}", flush=True)
print(f"DONE {total}", flush=True)
