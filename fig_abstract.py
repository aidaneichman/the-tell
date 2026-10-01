"""Figure 1 of the SSAC abstract: one cue held out, every cue held out, and the stratified alarm."""
import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import Patch
from common import load
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"], "font.size": 9,
    "axes.labelsize": 9, "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
INK, ACC, ACC2, RED, GRY, LGT = "#1b1f24", "#0f6e8c", "#e07b00", "#c0392b", "#8f969e", "#d9dde1"
TYPE_COL = {"SL": "#0f6e8c", "FF": "#c0392b", "CU": "#5b4b8a", "other": LGT}
def head(x, y, letter, title, sub=None):
    fig.text(x, y, letter, fontweight="bold", fontsize=12, va="baseline")
    fig.text(x + 0.032, y, title, fontweight="bold", fontsize=9.5, va="baseline")
    if sub: fig.text(x + 0.032, y - 0.028, sub, fontsize=8.5, color=INK, va="baseline")

fig = plt.figure(figsize=(7.2, 6.4))
# ---- A: one cue, found Apr-Jun, held out Jul-Sep
df = load(); d = df[(df.season == 2025) & (df.player_name == "Kikuchi, Yusei") & (df["count"] == 3) & (df.hand == 0) & (df.prev < 8)]
a = fig.add_axes([0.205, 0.585, 0.25, 0.30])
rows = []
for m in (d.game_date < "2025-07-01", d.game_date >= "2025-07-01"):
    for lab, s in (("after a slider", d[m & (d.prev == 3)]), ("after anything else", d[m & (d.prev != 3)])):
        f = np.bincount(s.x, minlength=8) / len(s); rows.append((lab, len(s), {"SL": f[3], "FF": f[0], "CU": f[5], "other": 1 - f[0] - f[3] - f[5]}))
ys = [4.1, 3.2, 1.2, 0.3]
for y, (lab, n, f) in zip(ys, rows):
    left = 0
    for k in ("SL", "FF", "CU", "other"):
        a.barh(y, f[k], left=left, height=0.74, color=TYPE_COL[k], edgecolor="white", linewidth=0.7)
        if f[k] >= 0.10: a.text(left + f[k] / 2, y, f"{100*f[k]:.0f}", ha="center", va="center", color=INK if k == "other" else "white", fontsize=8.5, fontweight="bold")
        left += f[k]
    a.text(-0.03, y + 0.1, lab, ha="right", va="center", fontsize=8.5)
    a.text(-0.03, y - 0.27, f"n = {n}", ha="right", va="center", fontsize=7.8, color=GRY)
for (y0, y1, hdr), (i, j) in zip(((4.95, 0, "Found: Apr–Jun"), (2.05, 0, "Held out: Jul–Sep")), ((0, 1), (2, 3))):
    a.text(0, y0, hdr, fontsize=9, fontweight="bold", va="center")
    gap = round(100 * rows[i][2]["SL"]) - round(100 * rows[j][2]["SL"])
    a.text(1.04, (ys[i] + ys[j]) / 2, f"+{gap:.0f}\npts", fontsize=10, fontweight="bold", color=ACC, va="center", ha="left")
a.set_xlim(0, 1); a.set_ylim(-0.25, 5.35); a.set_yticks([]); a.spines["left"].set_visible(False)
a.set_xticks([0, .5, 1]); a.set_xticklabels(["0", "50", "100%"])
a.legend(handles=[Patch(color=TYPE_COL[k], label=l) for k, l in (("SL", "slider"), ("FF", "4-seam"), ("CU", "curve"), ("other", "other"))],
         ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.32, -0.11), handlelength=0.9, columnspacing=0.8, handletextpad=0.3)
head(0.0, 0.955, "A", "One cue: Yusei Kikuchi, 2025, 1–0 count vs RHB", "next-pitch mix after a slider vs after anything else")

# ---- B: every pitcher's top cue, found Apr-Jun, scored Jul-Sep
t = pd.read_csv("out/tells_2025H1_2025H2.csv"); sg = np.sign(t.d); xin = 100 * t.d.abs(); xout = 100 * t.d_out * sg
b = fig.add_axes([0.625, 0.585, 0.355, 0.30])
xx = np.array([0, 75]); b.fill_between(xx, 0, xx, color=ACC, alpha=0.06, lw=0)
b.plot(xx, xx, color=GRY, lw=0.9, ls="--"); b.axhline(0, color=GRY, lw=0.8); b.axhline(xout.mean(), color=RED, lw=1.3)
b.scatter(xin, xout, s=18, color=INK, alpha=0.75, lw=0)
k = t.name == "Kikuchi, Yusei"; b.scatter(xin[k], xout[k], s=60, facecolor="none", edgecolor=ACC, lw=1.4)
b.annotate("Kikuchi", (xin[k].iloc[0], xout[k].iloc[0]), xytext=(-4, 7), textcoords="offset points", fontsize=8.5, color=ACC, ha="right", fontweight="bold")
b.text(50, 44, "no shrinkage", fontsize=8, color=GRY, rotation=35, ha="center", va="bottom")
b.text(74, xout.mean() + 1.5, f"average: {xout.mean():.0f} pts", fontsize=8.5, color=RED, ha="right", va="bottom", fontweight="bold")
b.text(74, -1.5, "no tell", fontsize=8, color=GRY, ha="right", va="top")
b.text(65, 36, "shrinkage", fontsize=8.5, color=ACC, ha="center", style="italic")
b.set_xlim(0, 75); b.set_ylim(-36, 66)
b.set_xlabel("shift when found, Apr–Jun (points)"); b.set_ylabel("same cue, held out Jul–Sep (points)")
head(0.535, 0.955, "B", f"Every pitcher's top cue, held out (n = {len(t)})",
     f"found {xin.mean():.0f} pts, held out {xout.mean():.0f}; same sign {100*(xout>0).mean():.0f}% (no tell: 0, 50%)")

# ---- C: stratified alarm
M = pd.read_csv("out/monitor_audit_2025.csv"); P = json.load(open("out/monitor_audit_paths.json")); S = json.load(open("out/monitor_audit.json")); SIM = json.load(open("out/monitor_sim.json"))
e = fig.add_axes([0.085, 0.075, 0.895, 0.33])
for nm, col in (("Brown, Hunter", ACC), ("Sale, Chris", ACC2), ("Skubal, Tarik", GRY)):
    r = M[M.name == nm].iloc[0]; p = P[str(int(r.pitcher))]; dd = pd.to_datetime(p["dates"]); y = np.array(p["strat"]) / np.log(10)
    first = nm.split(", ")[1] + " " + nm.split(",")[0]
    if r.strat_real_fired:
        i = int(np.argmax(y >= np.log10(20)))
        e.plot(dd[:i + 1], y[:i + 1], color=col, lw=1.8, label=f"{first}: fired at {int(r.strat_real_pitches):,} pitches")
        e.plot(dd[i:], y[i:], color=col, lw=1.0, alpha=0.35)
        e.plot(dd[i], y[i], "o", ms=6, color=col, mec="white", mew=0.8)
    else:
        e.plot(dd, y, color=col, lw=1.8, label=f"{first}: never fired")
e.axhline(np.log10(20), color=RED, lw=1.0, ls="--")
e.text(pd.Timestamp("2025-09-29"), np.log10(20) + 0.25, "alarm: E = 20 threshold", color=RED, fontsize=8.5, ha="right", va="bottom")
e.xaxis.set_major_locator(mdates.MonthLocator()); e.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
e.set_ylabel("evidence against no-cue null\n(log10 e-value)"); e.legend(frameon=False, loc="upper left")
e.text(0.995, 0.21, f"Fired for {100*S['strat_real_fired']:.0f}% of 2025 pitchers (median {S['strat_real_median_pitches']:.0f} pitches)\n"
       f"{100*S['strat_ctrl_fired']:.0f}% of cue-free controls, {100*SIM['rate']:.1f}% of endogenous-count simulations", transform=e.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)
head(0.0, 0.47, "C", "An in-season alarm", "permutes pitch order within count and batter hand, game by game")
fig.savefig("fig_abstract.pdf", bbox_inches="tight", metadata={"CreationDate": None}); fig.savefig("fig_abstract.png", dpi=300, bbox_inches="tight")
