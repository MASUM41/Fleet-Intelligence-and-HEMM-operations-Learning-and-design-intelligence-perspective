"""
rl_experiments/report.py — aggregate shootout results into tables + curves.

Reads analysis/rl/*.csv  ->  prints the final comparison table,
saves learning-curve PNGs, and checks hypotheses H1-H4.

Usage (from Simulation/):
    python -m rl_experiments.report
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PKG_DIR = Path(__file__).resolve().parent
_SIM_DIR = _PKG_DIR.parent
RL_DIR = _SIM_DIR / "analysis" / "rl"

METHOD_LABEL = {"b": "B one-for-all", "a": "A per-truck", "c": "C per-mine", "d": "D universal"}
METHOD_COLOR = {"b": "#2E86AB", "a": "#E63946", "c": "#F4A261", "d": "#9B59B6"}


def load_curves():
    frames = []
    for f in sorted(RL_DIR.glob("*_s*_curve.csv")):
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_baselines():
    p = RL_DIR / "baselines_eval.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main():
    curves = load_curves()
    base = load_baselines()

    if curves.empty and base.empty:
        print("No results yet. Run:  python -m rl_experiments.run --method all --workers 6")
        return

    # ---------------- final table (last eval point per method/seed) ----------
    print("=" * 78)
    print("FINAL EVAL (last point, mean +/- std over seeds)")
    print("=" * 78)
    rows = []
    if not curves.empty:
        last = curves.sort_values("trips").groupby(["method", "seed"]).tail(1)
        for m, g in last.groupby("method"):
            rows.append(dict(
                method=METHOD_LABEL.get(m, m),
                seeds=g["seed"].nunique(),
                reward=f"{g['reward'].mean():.1f} +/- {g['reward'].std():.1f}",
                duration_s=f"{g['mean_duration'].mean():.1f} +/- {g['mean_duration'].std():.1f}",
                throughput=f"{g['throughput'].mean():.1f} +/- {g['throughput'].std():.1f}",
                queue_s=f"{g['queue_wait'].mean():.2f}",
                fuel_L=f"{g['fuel'].mean():.3f}",
                balance=f"{g['balance'].mean():.2f}",
            ))
    tbl = pd.DataFrame(rows)
    if not tbl.empty:
        print(tbl.to_string(index=False))

    if not base.empty:
        agg = base.groupby("baseline").agg(
            reward=("reward", "mean"), duration_s=("mean_duration", "mean"),
            throughput=("throughput", "mean"), queue_s=("queue_wait", "mean"),
            fuel_L=("fuel", "mean"), balance=("balance", "mean")).round(2)
        print("\nBaselines (0 training trips, same eval suite):")
        print(agg.to_string())

    # ---------------- learning curves --------------------------------------
    if not curves.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for m, g in curves.groupby("method"):
            g = g.sort_values("trips")
            piv_r = g.pivot_table(index="trips", columns="seed", values="reward")
            piv_d = g.pivot_table(index="trips", columns="seed", values="mean_duration")
            col = METHOD_COLOR.get(m, None)
            lab = METHOD_LABEL.get(m, m)
            axes[0].plot(piv_r.index, piv_r.mean(axis=1), color=col, lw=2, label=lab)
            axes[0].fill_between(piv_r.index, piv_r.min(axis=1), piv_r.max(axis=1),
                                 color=col, alpha=0.15)
            axes[1].plot(piv_d.index, piv_d.mean(axis=1), color=col, lw=2, label=lab)
            axes[1].fill_between(piv_d.index, piv_d.min(axis=1), piv_d.max(axis=1),
                                 color=col, alpha=0.15)
        if not base.empty:
            agg = base.groupby("baseline")["reward"].mean()
            for name, v in agg.items():
                axes[0].axhline(v, ls="--", lw=1, alpha=0.5)
                axes[0].text(curves["trips"].max(), v, f" {name}", fontsize=7,
                             va="center", alpha=0.7)
        axes[0].set_title("Eval reward vs training trips")
        axes[0].set_xlabel("trips consumed"); axes[0].set_ylabel("mean episode reward")
        axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
        axes[1].set_title("Eval mean trip duration vs training trips")
        axes[1].set_xlabel("trips consumed"); axes[1].set_ylabel("seconds")
        axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
        fig.suptitle("Agent-Architecture Shootout (Double DQN, identical budget)", y=1.02)
        out = RL_DIR / "learning_curves.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved curves -> {out}")

    # ---------------- hypothesis check --------------------------------------
    if not curves.empty:
        last = curves.sort_values("trips").groupby(["method", "seed"]).tail(1)
        fin = last.groupby("method")["reward"].mean()
        print("\nHypothesis check (by final mean eval reward):")
        order = fin.sort_values(ascending=False)
        for m, v in order.items():
            print(f"  {METHOD_LABEL.get(m, m):<16} {v:8.1f}")
        if set(fin.index) >= {"b", "a"}:
            print(f"  H2 (A oscillates): per-truck curve std across eval points = "
                  f"{curves[curves.method=='a'].groupby('seed')['reward'].std().mean():.2f}"
                  f"  vs central B = "
                  f"{curves[curves.method=='b'].groupby('seed')['reward'].std().mean():.2f}")


if __name__ == "__main__":
    main()
