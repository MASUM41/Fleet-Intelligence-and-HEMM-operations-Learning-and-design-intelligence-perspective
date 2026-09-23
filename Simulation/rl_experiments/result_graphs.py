"""
rl_experiments/result_graphs.py — 6 report-grade graphs for the
agent-architecture shootout (A/B/C/D + hand-policy baselines).

Data source (nothing hardcoded):
    analysis/rl/{a,b,c,d}_s{0..4}_curve.csv   -> per-seed finals @ 100k trips
    analysis/rl/baselines_eval.csv            -> 20-config eval per baseline

Output: analysis/rl/graphs/*.png

Run (from Simulation/):
    python -m rl_experiments.result_graphs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIM_DIR = Path(__file__).resolve().parents[1]
RL_DIR = SIM_DIR / "analysis" / "rl"
OUT = RL_DIR / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
METHOD_COLOR = {                       # same colours as learning_curves.png
    "A per-truck":  "#D62728",
    "B one-for-all": "#1F77B4",
    "C per-mine":   "#F39C12",
    "D universal":  "#9467BD",
}
BASE_COLOR = {
    "GREEDY_NEAREST": "#2C3E50", "ETA_MIN": "#7F8C8D", "ETA_AWARE": "#95A5A6",
    "FIFO": "#BDC3C7", "LOAD_BALANCE": "#A6ACAF", "ROUND_ROBIN": "#C9CDD0",
}
SHORT = {                              # compact tick labels (avoid collisions)
    "GREEDY_NEAREST": "GREEDY", "ETA_MIN": "ETA\nMIN", "ETA_AWARE": "ETA\nAWARE",
    "FIFO": "FIFO", "LOAD_BALANCE": "LOAD\nBAL", "ROUND_ROBIN": "ROUND\nROBIN",
    "A per-truck": "A\nper-truck", "B one-for-all": "B\none-for-all",
    "C per-mine": "C\nper-mine", "D universal": "D\nuniversal",
}
RL_KEYS = {"a": "A per-truck", "b": "B one-for-all", "c": "C per-mine", "d": "D universal"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.8,
    "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.frameon": False,
})

METRICS = [                          # (column, display, unit, higher_is_better)
    ("reward", "Reward", "", True),
    ("mean_duration", "Trip duration", "s", False),
    ("throughput", "Throughput", "trips/h", True),
    ("queue_wait", "Queue wait", "s", False),
    ("fuel", "Fuel", "L/trip", False),
    ("balance", "Mine balance (std)", "", False),
]


# ---------------------------------------------------------------- data
def load_rl_finals() -> pd.DataFrame:
    rows = []
    for key, name in RL_KEYS.items():
        for seed in range(5):
            df = pd.read_csv(RL_DIR / f"{key}_s{seed}_curve.csv")
            last = df.iloc[-1].to_dict()
            last.update(method=name, seed=seed)
            rows.append(last)
    return pd.DataFrame(rows)


def load_baselines() -> pd.DataFrame:
    df = pd.read_csv(RL_DIR / "baselines_eval.csv")
    g = df.groupby("baseline")
    out = g[["reward", "mean_duration", "throughput", "queue_wait", "fuel", "balance"]].mean()
    out["reward_std"] = g["reward"].std()
    return out.reset_index().rename(columns={"baseline": "method"})


RL = load_rl_finals()
BASE = load_baselines()

RL_MEAN = RL.groupby("method")[["reward", "mean_duration", "throughput",
                                "queue_wait", "fuel", "balance", "wall_s"]].mean()
RL_STD = RL.groupby("method")[["reward", "mean_duration", "throughput",
                               "queue_wait", "fuel", "balance", "wall_s"]].std()

ALL = pd.concat([
    RL_MEAN.assign(reward_std=RL_STD["reward"]).reset_index(),
    BASE.assign(wall_s=np.nan),
], ignore_index=True)
ALL = ALL.set_index("method")


def save(fig, name):
    import time
    for attempt in range(5):                       # viewer/AV can lock files briefly
        try:
            fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.6)
    plt.close(fig)
    print(f"  -> {name}")


# ================================================================ 1. reward bars
def graph1_reward_bar():
    order = ALL["reward"].sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(order))
    for i, m in enumerate(order):
        is_rl = m in RL_KEYS.values()
        col = METHOD_COLOR.get(m, BASE_COLOR.get(m, "#999999"))
        vals = ALL.loc[m, "reward"]
        std = ALL.loc[m, "reward_std"] if not np.isnan(ALL.loc[m, "reward_std"]) else 0
        # RL whiskers = seed spread (dark); baseline whiskers = config spread (light)
        ax.bar(x[i], vals, color=col, width=0.68, zorder=3,
               yerr=std, capsize=3.5,
               error_kw=dict(ecolor="#222222" if is_rl else "#AAAAAA",
                             lw=1.4 if is_rl else 0.9))
        ax.text(x[i], vals + 14, f"{vals:.1f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="white", zorder=5)
        if is_rl:                                        # overlay the 5 seeds
            seeds = RL.loc[RL.method == m, "reward"]
            ax.scatter(np.full(len(seeds), x[i]), seeds, color="white",
                       edgecolor="#222222", s=26, zorder=4, linewidths=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(m, m) for m in order], fontsize=8.5)
    ax.set_ylabel("mean episode reward  (higher = better)")
    ax.set_title("Final evaluation reward after 100k training trips\n"
                 "bars = mean · dark whiskers = 5-seed spread (white dots = seeds) · "
                 "light whiskers = spread across the 20 scenarios")
    ax.margins(y=0.08)
    save(fig, "1_reward_bar.png")


# ================================================================ 2. radar profile
def graph2_radar():
    shown = ["C per-mine", "D universal", "B one-for-all", "A per-truck", "ETA_MIN"]
    axes_metrics = METRICS[:4]          # fuel & balance: identical (±<0.5%) -> omitted
    labels = [d for _, d, _, _ in axes_metrics]
    norm = {}
    for col, _, _, hib in axes_metrics:
        vals = ALL.loc[shown, col]
        lo, hi = vals.min(), vals.max()
        span = (hi - lo) or 1.0
        norm[col] = ((vals - lo) / span if hib else (hi - vals) / span).to_dict()

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw=dict(polar=True))
    for m in shown:
        col = METHOD_COLOR.get(m, BASE_COLOR.get(m))
        pts = [norm[c][m] for c, _, _, _ in axes_metrics]
        pts += pts[:1]
        ax.plot(angles, pts, color=col, lw=2.2, label=m, zorder=3)
        ax.fill(angles, pts, color=col, alpha=0.10, zorder=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8, color="#666666")
    ax.set_ylim(0, 1.05)
    ax.set_title("Metric profile (0–100 scale, 100 = best of the five shown)\n"
                 "fuel & mine-balance omitted: identical (±0.5%) for all five", pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
    save(fig, "2_radar_profile.png")


# ================================================================ 3. trade-off scatter
def graph3_scatter():
    # per-method label placement (dx, dy in points, ha) — tuned to avoid overlaps
    place = {
        "A per-truck":    (0, 13, "center"),
        "B one-for-all":  (0, 13, "center"),
        "D universal":    (-14, -6, "right"),
        "C per-mine":     (0, 13, "center"),
        "GREEDY_NEAREST": (0, -17, "center"),
        "ETA_MIN":        (0, 13, "center"),
        "FIFO":           (0, 13, "center"),
        "ROUND_ROBIN":    (-12, -2, "right"),
        "LOAD_BALANCE":   (0, -17, "center"),
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for m, row in ALL.iterrows():
        if m == "ETA_AWARE":
            continue                                   # identical to ETA_MIN
        col = METHOD_COLOR.get(m, BASE_COLOR.get(m, "#999999"))
        ax.scatter(row["mean_duration"], row["queue_wait"],
                   s=row["throughput"] * 7, color=col, alpha=0.85,
                   edgecolor="white", linewidths=1.2, zorder=3)
        label = "ETA_MIN = ETA_AWARE" if m == "ETA_MIN" else m
        dx, dy, ha = place.get(m, (0, 13, "center"))
        ax.annotate(label, (row["mean_duration"], row["queue_wait"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=9, fontweight="bold", color=col)
    ax.annotate("←  better", xy=(0, 1), xycoords="axes fraction",
                ha="left", va="top", fontsize=10, color="#555555")
    ax.annotate("better  →", xy=(1, 0), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=10, color="#555555")
    ax.set_xlabel("mean trip duration, s   (lower = better)")
    ax.set_ylabel("queue wait per trip, s   (lower = better)")
    ax.set_title("Speed vs congestion trade-off  (bubble size = throughput)")
    ax.set_xlim(240, 415)
    ax.set_ylim(-0.6, 9.9)
    save(fig, "3_tradeoff_scatter.png")


# ================================================================ 4. metric heatmap
def graph4_heatmap():
    order = ["C per-mine", "D universal", "B one-for-all", "A per-truck",
             "GREEDY_NEAREST", "ETA_MIN", "ETA_AWARE", "FIFO", "LOAD_BALANCE", "ROUND_ROBIN"]
    score = np.zeros((len(order), len(METRICS)))
    flat_cols = []
    for j, (col, _, _, hib) in enumerate(METRICS):
        vals = ALL.loc[order, col]
        lo, hi = vals.min(), vals.max()
        rel_spread = (hi - lo) / max(abs(vals.mean()), 1e-9)
        if rel_spread < 0.02:                      # no real difference -> neutral
            score[:, j] = 50.0
            flat_cols.append(j)
            continue
        span = (hi - lo) or 1.0
        score[:, j] = ((vals - lo) / span if hib else (hi - vals) / span) * 100

    xlabels = []
    for j, (_, d, u, _) in enumerate(METRICS):
        lbl = f"{d}\n({u})" if u else d
        if j in flat_cols:
            lbl += "  †"
        xlabels.append(lbl)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    im = ax.imshow(score, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(xlabels, fontsize=9.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9.5)
    for i, m in enumerate(order):
        for j, (col, _, u, _) in enumerate(METRICS):
            v = ALL.loc[m, col]
            txt = f"{v:.3f}" if col == "fuel" else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="#1a1a1a", fontweight="bold")
    ax.set_title("Scorecard: colour = 0–100 vs best/worst method per column\n"
                 "(number in cell = actual value;   † = differs by <2% across ALL "
                 "methods → no differentiation, shown neutral)")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("100 = best column value", fontsize=9)
    save(fig, "4_metric_heatmap.png")


# ================================================================ 5. seed spread
def graph5_seed_spread():
    order = ["C per-mine", "D universal", "B one-for-all", "A per-truck"]
    fig, ax = plt.subplots(figsize=(8, 5))
    xlabels = []
    for i, m in enumerate(order):
        seeds = RL.loc[RL.method == m, "reward"].to_numpy()
        jitter = np.linspace(-0.13, 0.13, len(seeds))
        ax.scatter(np.full(len(seeds), i) + jitter, seeds, s=90,
                   color=METHOD_COLOR[m], edgecolor="white", linewidths=1.0, zorder=3)
        mean = seeds.mean()
        ax.hlines(mean, i - 0.28, i + 0.28, color="#222222", lw=2.2, zorder=4)
        ax.vlines(i, seeds.min(), seeds.max(), color=METHOD_COLOR[m],
                  lw=1.2, alpha=0.5, zorder=2)
        xlabels.append(f"{m}\n(mean {mean:.1f})")
    # Reference lines: ETA_MIN = primary baseline; GREEDY = oracle ceiling; FIFO = floor
    for label, name, color, ls in [("ETA_MIN (baseline)", "ETA_MIN", "#C0392B", "-"),
                                   ("GREEDY (oracle ceiling)", "GREEDY_NEAREST", "#2C3E50", "--"),
                                   ("FIFO (floor)", "FIFO", "#888888", ":")]:
        y = ALL.loc[name, "reward"]
        ax.axhline(y, color=color, ls=ls, lw=1.2, zorder=1)
        ax.text(3.58, y + 1.0, label, va="bottom", ha="right", fontsize=8.5,
                color=color)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("final reward  (higher = better)")
    ax.set_title("Seed-to-seed stability: every dot is one full 100k-trip training run\n"
                 "(black tick = mean; note B's collapsed seed at −287)")
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylim(-415, -255)
    save(fig, "5_seed_spread.png")


# ================================================================ 6. vs ETA_MIN (primary baseline) %
def graph6_vs_baseline():
    g = ALL.loc["ETA_MIN"]           # primary baseline: strongest beatable hand rule
    rows = ["C per-mine", "D universal", "B one-for-all", "A per-truck"]
    specs = [("reward", True), ("throughput", True),
             ("mean_duration", False), ("queue_share", False)]

    def pct(m, col, hib):
        # queue wait scales with trip length -> compare the *share* of the
        # trip spent queuing (wait / duration), not the raw seconds
        if col == "queue_share":
            v = ALL.loc[m, "queue_wait"] / ALL.loc[m, "mean_duration"]
            gv = g["queue_wait"] / g["mean_duration"]
            return (gv / v - 1) * 100
        v, gv = ALL.loc[m, col], g[col]
        if col == "reward":                # both negative: use signed difference
            return (v - gv) / abs(gv) * 100
        return (v / gv - 1) * 100 if hib else (gv / v - 1) * 100

    fig, ax = plt.subplots(figsize=(8.8, 5))
    height, gap = 0.18, 0.055
    ypos = 0
    yticks, ylabels = [], []
    for m in rows:
        for k, (col, hib) in enumerate(specs):
            p = pct(m, col, hib)
            y = ypos - k * (height + gap)
            ax.barh(y, p, height=height, color=METHOD_COLOR[m],
                    alpha=1.0 - 0.18 * k, zorder=3)
            ax.text(p + (0.5 if p >= 0 else -0.5), y, f"{p:+.1f}%",
                    va="center", ha="left" if p >= 0 else "right", fontsize=8.5)
        yticks.append(ypos - 1.5 * (height + gap))
        ylabels.append(m)
        ypos -= 4 * (height + gap) + 0.14
    ax.axvline(0, color="#222222", lw=1.2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("% better (+)  or  worse (−)  than ETA_MIN (primary baseline)")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#666666", alpha=1.0 - 0.18 * k)
               for k in range(4)]
    ax.legend(handles, ["reward", "throughput", "trip duration", "queue share"],
              fontsize=9, loc="upper left")
    ax.set_title("Each RL design measured against ETA_MIN — the strongest beatable hand rule\n"
                 "(darkest = reward … lightest = queue share; queue = wait/duration, distance-normalised)")
    ax.set_xlim(-65, 36)
    save(fig, "6_vs_eta_min.png")


# ================================================================ 7. bar panels
def graph7_bar_panels():
    order = ALL["reward"].sort_values(ascending=False).index.tolist()
    panels = [("reward", "Episode reward", "higher = better", "{:.0f}"),
              ("mean_duration", "Trip duration (s)", "lower = better", "{:.0f}"),
              ("throughput", "Throughput (trips/h)", "higher = better", "{:.0f}"),
              ("queue_wait", "Queue wait (s)", "lower = better", "{:.1f}")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (col, title, hint, fmt) in zip(axs.flat, panels):
        x = np.arange(len(order))
        for i, m in enumerate(order):
            col_c = METHOD_COLOR.get(m, BASE_COLOR.get(m, "#999999"))
            v = ALL.loc[m, col]
            ax.bar(x[i], v, color=col_c, width=0.68, zorder=3)
            off = (ALL[col].max() - ALL[col].min()) * 0.02
            va, ytxt = ("bottom", v + off) if v >= 0 else ("top", v - off)
            ax.text(x[i], ytxt, fmt.format(v), ha="center", va=va, fontsize=7.5,
                    fontweight="bold", color="#333333")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        names = {"GREEDY_NEAREST": "GREEDY", "ETA_MIN": "ETA_MIN",
                 "ETA_AWARE": "ETA_AWR", "FIFO": "FIFO",
                 "LOAD_BALANCE": "LOAD_BAL", "ROUND_ROBIN": "R_ROBIN",
                 "A per-truck": "A", "B one-for-all": "B",
                 "C per-mine": "C", "D universal": "D"}
        ax.set_xticklabels([names.get(m, m) for m in order], fontsize=8,
                           rotation=35, ha="right")
        ax.set_title(f"{title}   ({hint})", fontsize=11)
        ax.margins(y=0.15)
    fig.suptitle("The four key metrics as bar charts — same method order everywhere",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "7_bar_panels.png")


# ================================================================ 8. scatter
def graph8_scatter():
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for m, col in METHOD_COLOR.items():
        sub = RL[RL.method == m]
        ax.scatter(sub["mean_duration"], sub["reward"], s=64, color=col,
                   label=m, alpha=0.9, edgecolor="white", linewidths=0.8, zorder=3)
    for name, col in BASE_COLOR.items():
        if name not in ("GREEDY_NEAREST", "ETA_MIN", "FIFO"):
            continue
        ax.scatter(ALL.loc[name, "mean_duration"], ALL.loc[name, "reward"],
                   marker="s", s=80, color=col, zorder=3)
        off = {"GREEDY_NEAREST": (-10, 0, "right"), "ETA_MIN": (10, -2, "left"),
               "FIFO": (-10, 0, "right")}[name]
        ax.annotate(name, (ALL.loc[name, "mean_duration"], ALL.loc[name, "reward"]),
                    textcoords="offset points", xytext=(off[0], off[1]),
                    ha=off[2], va="center", fontsize=8.5,
                    color=col, fontweight="bold")
    ax.set_xlabel("mean trip duration of the run, s   (lower = better)")
    ax.set_ylabel("episode reward   (higher = better)")
    ax.set_title("Every one of the 20 training runs: faster trips → higher reward\n"
                 "(dots = RL seeds, squares = hand-rule baselines)")
    ax.legend(fontsize=9, loc="upper right")
    ax.margins(0.12)
    save(fig, "8_scatter_duration_reward.png")


# ================================================================ 9. pies
def graph9_pies():
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.4))
    for ax, (m, col) in zip(axs, METHOD_COLOR.items()):
        dur = ALL.loc[m, "mean_duration"]
        wait = ALL.loc[m, "queue_wait"]
        share = wait / dur * 100
        ax.pie([wait, dur - wait], colors=["#C0392B", "#D5D8DC"], startangle=90,
               counterclock=False,
               wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5))
        ax.text(0, 0.08, f"{share:.1f}%", ha="center", fontsize=13,
                fontweight="bold", color="#C0392B")
        ax.text(0, -0.24, "queuing", ha="center", fontsize=8, color="#666666")
        ax.set_title(m, fontsize=11)
    fig.suptitle("Share of an average trip wasted standing in queue (red slice)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "9_queue_pies.png")


# ================================================================ 10. bubble
def graph10_bubble():
    place = {
        "A per-truck":    (-14, -2, "right"),
        "D universal":    (12, -2, "left"),
        "C per-mine":     (0, 13, "center"),
        "GREEDY_NEAREST": (-12, -2, "right"),
        "ETA_MIN":        (0, -18, "center"),
        "FIFO":           (0, 13, "center"),
        "ROUND_ROBIN":    (10, -2, "left"),
        "LOAD_BALANCE":   (0, -17, "center"),
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for m, row in ALL.iterrows():
        if m == "ETA_AWARE":
            continue
        col = METHOD_COLOR.get(m, BASE_COLOR.get(m, "#999999"))
        ax.scatter(row["throughput"], row["reward"],
                   s=row["queue_wait"] * 160 + 60, color=col, alpha=0.8,
                   edgecolor="white", linewidths=1.2, zorder=3)
        label = "ETA_MIN = ETA_AWARE" if m == "ETA_MIN" else m
        if m == "B one-for-all":          # crowded cluster -> leader-line label
            ax.annotate(label, (row["throughput"], row["reward"]),
                        textcoords="offset points", xytext=(26, -26),
                        ha="left", fontsize=9, fontweight="bold", color=col,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.9))
            continue
        dx, dy, ha = place.get(m, (0, 13, "center"))
        ax.annotate(label, (row["throughput"], row["reward"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=9, fontweight="bold", color=col)
    ax.annotate("↑  better", xy=(0, 1), xycoords="axes fraction",
                ha="left", va="top", fontsize=10, color="#555555")
    ax.annotate("better  →", xy=(1, 0), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=10, color="#555555")
    ax.set_xlabel("throughput, trips/h   (higher = better)")
    ax.set_ylabel("episode reward   (higher = better)")
    ax.set_title("Productivity vs reward  (bubble size = congestion: big = more queueing)")
    ax.set_xlim(55, 112)
    ax.set_ylim(-430, -245)
    save(fig, "10_bubble_throughput.png")


if __name__ == "__main__":
    print(f"reading from {RL_DIR}")
    graph1_reward_bar()
    graph2_radar()
    graph3_scatter()
    graph4_heatmap()
    graph5_seed_spread()
    graph6_vs_baseline()
    graph7_bar_panels()
    graph8_scatter()
    graph9_pies()
    graph10_bubble()
    print(f"done -> {OUT}")
