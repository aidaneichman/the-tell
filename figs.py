import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8.4,
    "axes.labelsize": 8.6, "axes.titlesize": 9.2, "legend.fontsize": 7.4, "axes.linewidth": 0.7, "axes.grid": True,
    "grid.alpha": 0.15, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
INK, ACC, RED, GRY, BLU = "#1b1f24", "#1f6f8b", "#c0392b", "#9aa1a9", "#2e86de"
T = pd.read_csv("out/tell_eb.csv"); R = json.load(open("out/reliability.json")); X = json.load(open("out/extra.json"))
def tag(ax, t): ax.text(-0.13, 1.08, t, transform=ax.transAxes, fontweight="bold", fontsize=10)
# ---------- Fig 1: the ladder and the plane
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
s = T[T.season == 2025]
a = ax[0]
labs = ["batter hand", "count", "previous pitch\n(TELL)"]; vals = [s.hand.mean(), s["count"].mean(), s.tell.mean()]
a.barh(range(3)[::-1], vals, color=[GRY, GRY, ACC], height=.6)
for i, v in enumerate(vals): a.text(v + .002, 2 - i, f"{v:.3f} bits", va="center", fontsize=7.8)
a.set_yticks(range(3)[::-1]); a.set_yticklabels(labs); a.set_xlim(0, .115); a.grid(axis="y", visible=False)
a.set_xlabel("out-of-sample information about the next pitch type\n(bits per pitch, mean over 2025 pitcher-seasons)")
a.set_title("What a hitter learns, in order", loc="left"); tag(a, "A")
a = ax[1]; b = s[s.n >= 1000]
a.scatter(b.eb * 1000, b.rv100, s=np.clip(b.n / 90, 4, 34), c=INK, alpha=.35, linewidths=0)
for _, r in pd.concat([b.nlargest(4, "eb"), b.nsmallest(3, "eb"), b.nlargest(3, "rv100")]).drop_duplicates("pitcher").iterrows():
    nm = r["name"].split(",")[0]; a.annotate(nm, (r.eb * 1000, r.rv100), xytext=(3, 3), textcoords="offset points", fontsize=7, color=ACC)
a.axvline(0, color=GRY, lw=.8); a.axhline(0, color=GRY, lw=.8)
a.set_xlabel("TELL, shrunk (millibits per pitch)"); a.set_ylabel("runs saved per 100 pitches")
a.set_title(f"Predictable is not the same as hittable (r = {np.corrcoef(b.eb, b.rv100)[0,1]:.2f})", loc="left"); tag(a, "B")
fig.tight_layout(w_pad=3); fig.savefig("fig1_hero.pdf", bbox_inches="tight"); plt.close(fig)
# ---------- Fig 2: reliability
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
a = ax[0]; s = pd.read_csv("out/splithalf_indep.csv"); s = s[s.season == 2025]
a.scatter(s.t0 * 1000, s.t1 * 1000, s=5, c=INK, alpha=.35, linewidths=0)
lim = [-20, 70]; a.plot(lim, lim, color=RED, lw=1, ls="--"); a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("TELL from odd games only (mbits)"); a.set_ylabel("TELL from even games only (mbits)")
a.set_title(f"Within a season: reliability {R['split_indep_2025']['sb']:.2f}", loc="left"); tag(a, "A")
for k, (col, lab) in enumerate((("tell", "TELL"), ("count", "count usage"))):
    a = ax[k + 1]; x = T[T.season == 2025].set_index("pitcher"); y = T[T.season == 2026].set_index("pitcher"); j = x.index.intersection(y.index)
    a.scatter(x.loc[j, col] * 1000, y.loc[j, col] * 1000, s=5, c=ACC if col == "tell" else GRY, alpha=.5, linewidths=0)
    lo = min(x.loc[j, col].min(), y.loc[j, col].min()) * 1000; hi = max(x.loc[j, col].max(), y.loc[j, col].max()) * 1000
    a.plot([lo, hi], [lo, hi], color=RED, lw=1, ls="--")
    a.set_xlabel(f"{lab}, 2025 (mbits)"); a.set_ylabel(f"{lab}, 2026 (mbits)")
    a.set_title(f"{lab} across seasons: r = {R[f'yoy_{col}_2025_2026']['r']:.2f}", loc="left"); tag(a, "BC"[k])
fig.tight_layout(w_pad=2.2); fig.savefig("fig2_reliability.pdf", bbox_inches="tight"); plt.close(fig)
# ---------- Fig 3: leaderboard
L = T[(T.season == 2025) & (T.n >= 1500)].sort_values("eb")
sel = pd.concat([L.head(12), L.tail(12)])
fig, ax = plt.subplots(figsize=(10.5, 3.9))
yy = np.arange(len(sel))
ax.errorbar(sel.eb * 1000, yy, xerr=1.645 * sel.eb_sd * 1000, fmt="o", color=INK, ms=3.5, lw=1, capsize=0)
ax.scatter(sel.tell * 1000, yy, marker="x", s=14, color=RED, zorder=5, label="raw (unshrunk)")
ax.set_yticks(yy); ax.set_yticklabels([n.split(",")[0] + ", " + n.split(",")[1].strip()[0] + "." for n in sel.name], fontsize=6.8)
ax.axvline(0, color=GRY, lw=.8); ax.axhline(11.5, color=GRY, lw=.6, ls=":")
ax.set_xlabel("TELL, empirical-Bayes estimate with 90% interval (millibits per pitch)")
ax.text(.99, .03, "least predictable 12", transform=ax.transAxes, ha="right", fontsize=7.4, color=GRY)
ax.text(.99, .95, "most predictable 12", transform=ax.transAxes, ha="right", fontsize=7.4, color=GRY)
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.99, 0.08)); ax.set_title("2025 starters and bulk arms (at least 1,500 pitches)", loc="left")
fig.tight_layout(); fig.savefig("fig3_leaderboard.pdf", bbox_inches="tight"); plt.close(fig)
# ---------- Fig 4: cost, binned partial residuals
from cost_bins import binned
B = binned(); C = json.load(open("out/cost.json"))
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
for k, (yv, lab, key, scale) in enumerate((("whiff", "whiff rate on swings (pp)", "whiff_given_swing", 100), ("rv", "runs saved per 100 pitches", "rv", 100),
                                          ("xw", "xwOBA on contact (points)", "xwoba_contact", 1000))):
    a = ax[k]; d = B[yv]
    a.errorbar(d.x, d.y * scale, yerr=1.96 * d.se * scale, fmt="o", color=INK, ms=3.5, capsize=0)
    bt = C[key]["beta"][0]; xs = np.linspace(d.x.min(), d.x.max(), 10); a.plot(xs, bt * xs * scale, color=ACC, lw=1.5)
    a.axhline(0, color=GRY, lw=.7); a.set_xlabel("sequencing predictability of the pitch (bits)")
    a.set_ylabel(lab + ", relative"); tag(a, "ABC"[k])
    a.set_title(f"slope {bt*scale:+.2f} per bit (t = {bt/C[key]['se'][0]:+.1f})", loc="left")
fig.tight_layout(w_pad=2.2); fig.savefig("fig4_cost.pdf", bbox_inches="tight"); plt.close(fig)
# ---------- Fig 5: the monitor
M = pd.read_csv("out/monitor_2025.csv"); P = json.load(open("out/monitor_paths.json"))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.1), gridspec_kw=dict(width_ratios=[1.5, 1]))
a = ax[0]
names = ["Severino, Luis", "Brown, Hunter", "Skubal, Tarik", "Wheeler, Zack"]
cols = [ACC, BLU, GRY, "#7f8c8d"]
for nm, c in zip(names, cols):
    r = M[M.name == nm]
    if not len(r): continue
    p = P[str(int(r.pitcher.iloc[0]))]; d = pd.to_datetime(p["dates"])
    a.plot(d, np.array(p["logE"]) / np.log(10), color=c, lw=1.6, label=nm.split(",")[0])
a.axhline(np.log10(20), color=RED, lw=1, ls="--"); a.text(pd.Timestamp("2025-09-10"), np.log10(20) - 2.6, "alarm: E = 20", color=RED, fontsize=7.4)
a.set_ylabel("log10 evidence against memorylessness"); a.legend(frameon=False, loc="upper left")
a.set_title("The in-season monitor, 2025 (valid at every outing)", loc="left"); tag(a, "A")
import matplotlib.dates as mdates; a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
a = ax[1]; f = M[M.fired].pitches_at_fire.sort_values()
a.step(f, np.arange(1, len(f) + 1) / len(M), color=INK, where="post")
a.set_xscale("log"); a.set_xlabel("pitches thrown when the alarm fired"); a.set_ylabel("share of 2025 pitchers")
a.set_title(f"{100*M.fired.mean():.0f}% fire; median at {int(f.median())} pitches", loc="left"); tag(a, "B")
fig.tight_layout(w_pad=2.5); fig.savefig("fig5_monitor.pdf", bbox_inches="tight"); plt.close(fig)
# ---------- Fig 6: ABS
Z = json.load(open("out/abs_zone_abs.json"))
df = load(min_n=1)
c = df[df.description.isin(["called_strike", "ball", "blocked_ball"]) & df.season.isin([2025, 2026])].dropna(subset=["plate_x", "plate_z"])
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw=dict(width_ratios=[1, 1.3]))
a = ax[0]; xb = np.arange(-1.6, 1.61, .08); zb = np.arange(0.6, 4.4, .08)
for yr, col, ls in ((2025, GRY, "--"), (2026, RED, "-")):
    s = c[c.season == yr]; H = np.histogram2d(s.plate_x, s.plate_z, [xb, zb])[0]
    S = np.histogram2d(s.plate_x, s.plate_z, [xb, zb], weights=(s.description == "called_strike"))[0]
    Pm = np.where(H >= 25, S / np.maximum(H, 1), np.nan)
    from scipy.ndimage import gaussian_filter
    Pm = gaussian_filter(np.nan_to_num(Pm, nan=0), 1)
    a.contour((xb[:-1] + xb[1:]) / 2, (zb[:-1] + zb[1:]) / 2, Pm.T, levels=[.5], colors=[col], linestyles=[ls], linewidths=1.8)
    a.plot([], [], color=col, ls=ls, label=f"{yr}: 50% called-strike contour")
a.set_aspect("equal"); a.set_xlim(-1.4, 1.4); a.set_ylim(0.9, 4.1); a.set_xlabel("plate x (ft, catcher's view)"); a.set_ylabel("plate z (ft)")
a.legend(frameon=False, loc="lower center", fontsize=7)
a.set_title(f"Top of the called zone: {12*(Z['2026']['top_ft']-Z['2025']['top_ft']):+.1f} in", loc="left"); tag(a, "A")
a = ax[1]; A = json.load(open("out/abs2.json"))
rows = [("zone top, placebo\n2024 to 2025", 12 * (Z["2025"]["top_ft"] - Z["2024"]["top_ft"]), None, GRY, "in"),
        ("zone top, ABS\n2025 to 2026", 12 * (Z["2026"]["top_ft"] - Z["2025"]["top_ft"]), None, RED, "in")]
a2 = a.twinx(); a2.grid(False)
for i, (lb, v, _, col, u) in enumerate(rows): a.bar(i, v, color=col, width=.55)
a.set_ylabel("change in the called zone's top edge (inches)"); a.axhline(0, color=GRY, lw=.7)
for i, (key, lb, col) in enumerate((("tell_2024_2025", "TELL, placebo\n2024 to 2025", GRY), ("tell_2025_2026", "TELL, ABS\n2025 to 2026", RED))):
    r = X[key]; a2.errorbar(i + 2.3, r["diff"] * 1000, yerr=[[(r["diff"] - r["lo"]) * 1000], [(r["hi"] - r["diff"]) * 1000]], fmt="o", color=col, capsize=4, ms=5)
a2.set_ylabel("paired change in TELL (mbits), 95% CI"); a2.set_ylim(-3, 3); a.set_ylim(-1.5, 1.5)
a.set_xticks([0, 1, 2.3, 3.3]); a.set_xticklabels([r[0] for r in rows] + ["TELL, placebo\n2024 to 2025", "TELL, ABS\n2025 to 2026"], fontsize=6.8)
a.grid(axis="x", visible=False); a.set_title("The zone moved; sequencing did not", loc="left"); tag(a, "B")
fig.tight_layout(w_pad=2.5); fig.savefig("fig6_abs.pdf", bbox_inches="tight"); plt.close(fig)
print("figures written")
