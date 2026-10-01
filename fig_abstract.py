"""Figure 1 of the SSAC abstract: one tell, every tell held out, the cost, and the alarm."""
import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from common import load, NAME
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"], "font.size": 7.6,
    "axes.labelsize": 7.6, "axes.titlesize": 8.2, "axes.titleweight": "bold", "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
INK, ACC, ACC2, RED, GRY, LGT = "#1b1f24", "#0f6e8c", "#e07b00", "#c0392b", "#8f969e", "#d9dde1"
TYPE_COL = {"FF": "#c0392b", "SI": "#e07b00", "SL": "#0f6e8c", "CU": "#5b4b8a", "CH": "#3a9d5d", "other": LGT}
from matplotlib.patches import Patch
def head(fig, x, y, letter, title):
    fig.text(x, y, letter, fontweight="bold", fontsize=10.5, va="baseline")
    fig.text(x + 0.03, y, title, fontweight="bold", fontsize=8.2, va="baseline")

fig = plt.figure(figsize=(7.1, 5.2))
L, R2, TOP, BOT = 0.0, 0.53, 0.955, 0.465

# ---- A: one tell, found in April-June, checked in July-September
df = load(); d = df[(df.season == 2025) & (df.player_name == "Kikuchi, Yusei") & (df["count"] == 3) & (df.hand == 0) & (df.prev < 8)]
a = fig.add_axes([0.235, 0.60, 0.24, 0.30])
rows = []
for half, m in (("Apr–Jun", d.game_date < "2025-07-01"), ("Jul–Sep", d.game_date >= "2025-07-01")):
    for lab, s in (("after a slider", d[m & (d.prev == 3)]), ("after anything else", d[m & (d.prev != 3)])):
        f = np.bincount(s.x, minlength=8) / len(s)
        rows.append((half, lab, len(s), {"FF": f[0], "SL": f[3], "CU": f[5], "other": 1 - f[0] - f[3] - f[5]}))
ys = [4.1, 3.2, 1.3, 0.4]
for y, (half, lab, n, f) in zip(ys, rows):
    left = 0
    for k in ("SL", "FF", "CU", "other"):
        a.barh(y, f[k], left=left, height=0.72, color=TYPE_COL[k], edgecolor="white", linewidth=0.6)
        if f[k] >= 0.12: a.text(left + f[k] / 2, y, f"{100*f[k]:.0f}%", ha="center", va="center", color=INK if k == "other" else "white", fontsize=6.8, fontweight="bold")
        left += f[k]
    a.text(-0.02, y, f"{lab}  (n={n})", ha="right", va="center", fontsize=6.8)
for y, h in ((4.95, "Apr–Jun 2025: found"), (2.15, "Jul–Sep 2025: checked, never used to find it")):
    a.text(0, y, h, fontsize=7, color=INK, fontweight="bold", va="center")
a.set_xlim(0, 1); a.set_ylim(-0.1, 5.3); a.set_yticks([]); a.spines["left"].set_visible(False)
a.set_xticks([0, .5, 1]); a.set_xticklabels(["0", "50%", "100%"])
a.legend(handles=[Patch(color=TYPE_COL[k], label=l) for k, l in (("SL", "slider"), ("FF", "4-seam"), ("CU", "curve"), ("other", "other"))],
         ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), handlelength=0.9, columnspacing=0.9, handletextpad=0.35)
head(fig, L, TOP, "A", "One tell: Yusei Kikuchi, 1–0 count, right-handed batter")

# ---- B: every pitcher's biggest tell, chosen in H1, scored in H2
t = pd.read_csv("out/tells_2025H1_2025H2.csv"); sg = np.sign(t.d)
xin = 100 * t.d.abs(); xout = 100 * t.d_out * sg; rep = (t.prev == t.typ) & (t.d > 0)
b = fig.add_axes([0.605, 0.60, 0.385, 0.30])
b.axhline(0, color=GRY, lw=0.6); b.plot([10, 70], [10, 70], color=GRY, lw=0.7, ls="--")
b.text(52, 47, "held out = selected", fontsize=6.2, color=GRY, rotation=31, ha="center", va="bottom")
b.scatter(xin[~rep], xout[~rep], s=11, color=GRY, alpha=0.7, lw=0, label="other tells")
b.scatter(xin[rep], xout[rep], s=13, color=ACC, alpha=0.9, lw=0, label="doubling up")
k = t.name == "Kikuchi, Yusei"
b.scatter(xin[k], xout[k], s=40, facecolor="none", edgecolor=INK, lw=0.9); b.annotate("Kikuchi", (xin[k].iloc[0], xout[k].iloc[0]), xytext=(6, 2), textcoords="offset points", fontsize=6.6)
b.axhline(xout.mean(), color=RED, lw=1.0)
b.text(69, -31, f"selected {xin.mean():.0f} pts, held out {xout.mean():.0f} pts\nsame direction {100*(xout>0).mean():.0f}%", color=RED, fontsize=6.5, ha="right", va="bottom")
b.set_xlim(10, 70); b.set_ylim(-35, 65)
b.set_xlabel("shift in next-pitch share, Apr–Jun (points)"); b.set_ylabel("same tell, Jul–Sep (points)")
b.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="upper left", handletextpad=0.2, borderaxespad=0.1)
head(fig, R2, TOP, "B", f"Each pitcher's biggest tell, re-scored on new games (n = {len(t)})")

# ---- C: predictability is not punished
from cost_bins import binned
B = binned(); C = json.load(open("out/cost.json"))
c = fig.add_axes([0.075, 0.075, 0.40, 0.33]); w = B["whiff"]
c.axhline(0, color=GRY, lw=0.6)
c.errorbar(w.x, 100 * w.y, yerr=196 * w.se, fmt="o", color=INK, ms=3, capsize=0, lw=0.8)
bt = C["whiff_given_swing"]["beta"][0]; xs = np.linspace(w.x.min(), w.x.max(), 10)
c.plot(xs, 100 * bt * xs, color=ACC, lw=1.4)
c.set_xlabel("how much the previous pitch gave this pitch away (bits)"); c.set_ylabel("whiff rate on swings, relative (pp)")
head(fig, L, BOT, "C", "Hitters do not cash in: predictable pitches miss more bats")
c.text(0.98, 0.05, f"{100*bt:+.1f} pp per bit, t = {bt/C['whiff_given_swing']['se'][0]:+.1f}\nn = {C['whiff_given_swing']['n']:,} swings, 2025–26\npitcher-season and type×count FE",
       transform=c.transAxes, ha="right", fontsize=6.3, color=INK)

# ---- D: the alarm
FIRE_OFF = {"Kikuchi, Yusei": (14, -11), "Severino, Luis": (2, 6), "Wheeler, Zack": (26, -5)}
M = pd.read_csv("out/monitor_2025.csv"); P = json.load(open("out/monitor_paths.json"))
e = fig.add_axes([0.605, 0.075, 0.385, 0.33])
for nm, col in (("Kikuchi, Yusei", ACC), ("Severino, Luis", ACC2), ("Wheeler, Zack", GRY)):
    r = M[M.name == nm]
    if not len(r): continue
    p = P[str(int(r.pitcher.iloc[0]))]; dd = pd.to_datetime(p["dates"]); y = np.array(p["logE"]) / np.log(10)
    e.plot(dd, y, color=col, lw=1.3, label=f"{nm.split(', ')[1]} {nm.split(',')[0]}: fired at {int(r.pitches_at_fire.iloc[0]):,} pitches")
    if r.fired.iloc[0]:
        i = int(r.fire_game.iloc[0]) - 1; e.plot(dd[i], y[i], "o", ms=4.5, color=col, mec="white", mew=0.6)
e.axhline(np.log10(20), color=RED, lw=0.9, ls="--", label="alarm line, E = 20")
e.set_yticks([-5, 0, 5, 10]); e.xaxis.set_major_locator(mdates.MonthLocator()); e.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
e.set_ylabel("evidence of sequencing, log10 e-value"); e.legend(frameon=False, loc="upper left", fontsize=6.3, borderaxespad=0.1); e.set_ylim(-6, 17)
f = M[M.fired]
head(fig, R2, BOT, "D", "An in-season alarm with a false-alarm guarantee")
fig.savefig("fig_abstract.pdf", bbox_inches="tight"); fig.savefig("fig_abstract.png", dpi=220, bbox_inches="tight")
print(M[M.name.isin(["Kikuchi, Yusei", "Severino, Luis", "Wheeler, Zack"])][["name", "fired", "fire_date", "pitches_at_fire"]])
