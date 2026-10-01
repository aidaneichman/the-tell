"""Every number in abstract.tex, as LaTeX macros, read from the output files. Writes numbers.tex."""
import numpy as np, pandas as pd, json
out = []
def put(k, v): out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
def com(x): return f"{int(round(x)):,}".replace(",", "{,}")
def sgn(x, f="{:+.2f}"): return f.format(x).replace("-", "$-$").replace("+", "$+$")
T = pd.read_csv("out/tell.csv"); R = json.load(open("out/reliability.json")); X = json.load(open("out/extra.json"))
C = json.load(open("out/cost.json")); Z = json.load(open("out/abs_zone_abs.json")); H = json.load(open("out/tells_holdout.json"))
M = json.load(open("out/monitor_audit.json")); S = json.load(open("out/monitor_sim.json"))
G = pd.read_csv("out/guess.csv"); TB = pd.read_csv("out/table_tells.csv")
put("nPitches", com(T.n.sum())); put("nPS", com(len(T))); put("minN", "250")
s = T[T.season == 2025]
put("mbHand", f"{1000*s.hand.mean():.0f}"); put("mbCount", f"{1000*s['count'].mean():.0f}"); put("mbTell", f"{1000*s.tell.mean():.1f}")
g = G[G.season == 2025]; put("guessGain", f"{100*g.gain.mean():.1f}")
gt = g[g.n >= 1500].nlargest(1, "gain").iloc[0]; put("guessTopName", gt["name"].split(", ")[1] + " " + gt["name"].split(",")[0]); put("guessTopGain", f"{100*gt.gain:.0f}")
h = H["2025H1_2025H2"]; put("hoIn", f"{100*h['d_in']:.0f}"); put("hoOut", f"{100*h['d_out']:.0f}"); put("hoSign", f"{100*h['same_sign']:.0f}")
put("hoNet", f"{100*h['net_out']:.0f}"); put("hoNetSign", f"{100*h['net_same_sign']:.0f}"); put("hoN", f"{h['n']}")
put("tabIn", f"{TB.d25.abs().mean():.0f}"); put("tabOut", f"{(TB.d26*np.sign(TB.d25)).mean():.0f}"); put("tabKept", f"{int(TB.kept.sum())}"); put("tabN", f"{len(TB)}")
put("relHalf", f"{R['split_indep_2025']['sb']:.2f}"); put("yoyTell", f"{R['yoy_tell_2025_2026']['r']:.2f}"); put("yoyCount", f"{R['yoy_count_2025_2026']['r']:.2f}")
sp = C["span"]
for tag, suf in (("base", ""), ("prevout", "Po")):
    for k, nm, sc, fm in (("whiff_given_swing", "Whiff", 100, "{:.1f}"), ("rv", "Rv", 100, "{:.2f}")):
        b, se = C[tag][k]["beta"][0], C[tag][k]["se"][0]
        put(f"eff{nm}{suf}", fm.format(b * sp * sc)); put(f"t{nm}{suf}", f"{b/se:.1f}")
put("almFire", f"{100*M['strat_real_fired']:.0f}"); put("almMed", com(M["strat_real_median_pitches"]))
put("almCtrl", f"{100*M['strat_ctrl_fired']:.0f}"); put("almSim", f"{100*S['rate']:.1f}"); put("almOrigCtrl", f"{100*M['orig_ctrl_fired']:.0f}")
put("topAbs", f"{12*(Z['2025']['top_ft']-Z['2026']['top_ft']):.1f}"); put("topPlac", f"{12*(Z['2024']['top_ft']-Z['2025']['top_ft']):.1f}")
for a, b, t in ((2024, 2025, "Plac"), (2025, 2026, "Abs")):
    r = X[f"tell_{a}_{b}"]; put(f"tell{t}Diff", sgn(1000*r["diff"])); put(f"tell{t}Lo", sgn(1000*r["lo"])); put(f"tell{t}Hi", sgn(1000*r["hi"]))
open("numbers.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out))
