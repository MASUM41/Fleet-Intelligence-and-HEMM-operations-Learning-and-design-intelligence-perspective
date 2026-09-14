"""
rl_experiments/run.py — orchestrator for the agent-architecture shootout.

Framings (all Double DQN, identical hyperparams & budget):
    b  one-for-all   : 1 central DDQN, action = mine (9), global obs (49)
    a  per-truck     : N independent DDQNs, action = mine (9), local obs (38)
    c  per-mine      : 9 bidder DDQNs, action = bid level (5), buyer rule picks mine
    d  universal     : 1 central DDQN, action = mine x dump (54), global+dump obs (55)
    baselines        : 6 hand policies, 0 training trips (reference floor)

Fairness contract: same env, same reward (-duration/100), same eval suite,
same trip budget, same hyperparams, same seed schedule. Only `act` changes.

Usage (from Simulation/):
    python -m rl_experiments.run --method b --seed 0 --trips 100000
    python -m rl_experiments.run --method all --seeds 5 --workers 6 --trips 100000
    python -m rl_experiments.run --method baselines
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PKG_DIR = Path(__file__).resolve().parent
_SIM_DIR = _PKG_DIR.parent
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from rl_experiments.env import DispatchEnv, BID_GRID      # noqa: E402
from rl_experiments.nets import DDQNCore                   # noqa: E402
from rl_experiments.baselines import BASELINE_FACTORIES    # noqa: E402

OUT_DIR = _SIM_DIR / "analysis" / "rl"

OBS_DIM = {"b": 49, "a": 38, "c": 7, "d": 55}
N_ACT = {"b": 9, "a": 9, "c": len(BID_GRID), "d": 54}
ENV_MODE = {"b": "central", "a": "central", "c": "central", "d": "universal"}

# Fixed eval suite: same 20 configs for every method/seed (paired comparison)
EVAL_CONFIGS = [dict(fleet=f, coal_scale=c, seed=900000 + i)
                for i, (f, c) in enumerate(
                    (f, c) for f in (4, 6, 8, 10, 12) for c in (0.6, 1.0, 1.5, 2.0))]


@dataclass
class Cfg:
    trips: int = 100_000
    eval_every: int = 10_000
    episode_cap: int = 200
    device: str = "cpu"
    eval_configs: int = 20


def eps_at(step, total):
    return max(0.05, 1.0 - 0.95 * step / (0.6 * total))


# ══════════════════════════════════════════════════════════════════
#  Shared episode driver (used for eval of every method + baselines)
# ══════════════════════════════════════════════════════════════════

def run_episode(env, method, decider):
    obs, info = env.reset()
    done = False
    while not done:
        if method in ("b", "d"):
            a = decider(obs, info["mask"])
        elif method == "a":
            a = decider(info["truck_id"], env.observe("local"), info["mask"])
        elif method == "c":
            a = decider(env, env.debug())
        else:                       # baseline name -> mine name
            a = env.mines.index(decider(env, env.debug()))
        obs, r, done, _, info = env.step(a)
    return env.episode_metrics()


def evaluate_configs(method, decider, cfgs):
    rows = []
    for c in cfgs:
        env = DispatchEnv(mode=ENV_MODE[method], fleet=c["fleet"],
                          coal_scale=c["coal_scale"], master_seed=c["seed"])
        m = run_episode(env, method, decider)
        m.update(fleet=c["fleet"], coal_scale=c["coal_scale"])
        rows.append(m)
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("reward", "mean_duration", "throughput", "queue_wait", "fuel", "balance")}
    return agg, rows


# ══════════════════════════════════════════════════════════════════
#  Policies
# ══════════════════════════════════════════════════════════════════

class CentralPolicy:                     # B and D
    def __init__(self, method, device, seed):
        self.core = DDQNCore(OBS_DIM[method], N_ACT[method], device=device, seed=seed)

    def act(self, obs, mask, eps):
        return self.core.act(obs, mask, eps)

    def store_learn(self, s, a, r, s2, done, mask2):
        self.core.store(s, a, r, s2, done, mask2)
        self.core.learn()


class TrucksPolicy:                      # A — independent per-truck learners
    """Proper per-truck MDP: transition completes when the SAME truck pops again."""

    def __init__(self, device, seed):
        self.device, self.seed = device, seed
        self.cores = {}
        self.awaiting = {}               # tid -> (s, a, r) waiting for its next decision

    def _core(self, tid):
        if tid not in self.cores:
            self.cores[tid] = DDQNCore(OBS_DIM["a"], N_ACT["a"], device=self.device,
                                       n_step=1, min_buffer=500, seed=self.seed + tid)
        return self.cores[tid]

    def act(self, tid, obs, mask, eps):
        core = self._core(tid)
        if tid in self.awaiting:                       # complete old transition
            s, a, r = self.awaiting.pop(tid)
            core.store(s, a, r, obs, False, mask)
            core.learn()
        return core.act(obs, mask, eps)

    def attach(self, tid, s, a, r):
        self.awaiting[tid] = (s, a, r)

    def flush(self, done=True):
        for tid, (s, a, r) in self.awaiting.items():   # episode over; terminal
            self._core(tid).store(s, a, r, s, True, np.ones(N_ACT["a"], dtype=np.float32))
        self.awaiting.clear()


class MinesPolicy:                       # C — per-mine bidders + buyer rule
    def __init__(self, n_mines, device, seed):
        self.cores = [DDQNCore(OBS_DIM["c"], N_ACT["c"], device=device,
                               min_buffer=500, seed=seed + 100 * j)
                      for j in range(n_mines)]
        self.prev = None
        self._learn_ptr = 0                # rotate: 1 gradient update per event
                                           # (equalizes update count with B, ~9x faster)

    def act(self, env, dbg, eps):
        obs_j, lvls, bids = [], [], {}
        for j, core in enumerate(self.cores):
            o = env.observe_mine(j)
            lv = core.act(o, np.ones(N_ACT["c"], dtype=np.float32), eps)
            obs_j.append(o)
            lvls.append(lv)
            bids[env.mines[j]] = BID_GRID[lv]
        self.prev = (obs_j, lvls)
        act = {m: v for m, v in dbg["mines"].items() if v["coal"] > 0}
        best = min(act, key=lambda m: act[m]["travel"] + act[m]["wait"]
                   + act[m]["cycle"] + bids[m])
        return env.mines.index(best)

    def store_learn(self, env, r, done):
        obs_j, lvls = self.prev
        for j, core in enumerate(self.cores):
            core.store(obs_j[j], lvls[j], r, env.observe_mine(j), done,
                       np.ones(N_ACT["c"], dtype=np.float32))
        self.cores[self._learn_ptr].learn()          # one learner per event
        self._learn_ptr = (self._learn_ptr + 1) % len(self.cores)


# ══════════════════════════════════════════════════════════════════
#  Training runs
# ══════════════════════════════════════════════════════════════════

def _append_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def train_run(method, seed, cfg: Cfg):
    import torch
    torch.set_num_threads(1)
    t_start = time.perf_counter()
    eval_cfgs = EVAL_CONFIGS[: cfg.eval_configs]

    env = DispatchEnv(mode=ENV_MODE[method], episode_cap=cfg.episode_cap,
                      master_seed=seed * 10_000)
    obs, info = env.reset()

    if method in ("b", "d"):
        policy = CentralPolicy(method, cfg.device, seed)
        decide = lambda o, m, e: policy.act(o, m, e)
    elif method == "a":
        policy = TrucksPolicy(cfg.device, seed)
    else:
        policy = MinesPolicy(len(env.mines), cfg.device, seed)

    curve_path = OUT_DIR / f"{method}_s{seed}_curve.csv"
    if curve_path.exists():
        curve_path.unlink()
    header = ["method", "seed", "trips", "reward", "mean_duration",
              "throughput", "queue_wait", "fuel", "balance", "wall_s"]

    for step in range(cfg.trips):
        eps = eps_at(step, cfg.trips)
        if method in ("b", "d"):
            a = policy.act(obs, info["mask"], eps)
            prev_obs, prev_mask = obs, info["mask"]
            obs, r, term, _, info = env.step(a)
            policy.store_learn(prev_obs, a, r, obs, term,
                               info["mask"] if not term else prev_mask)
        elif method == "a":
            tid = info["truck_id"]
            obs_l = env.observe("local")
            a = policy.act(tid, obs_l, info["mask"], eps)
            obs, r, term, _, info = env.step(a)
            policy.attach(tid, obs_l, a, r)
            if term:
                policy.flush()
        else:                        # c
            a = policy.act(env, env.debug(), eps)
            obs, r, term, _, info = env.step(a)
            policy.store_learn(env, r, term)

        if term:
            obs, info = env.reset()

        if (step + 1) % cfg.eval_every == 0:
            if method in ("b", "d"):
                decider = lambda o, m: policy.act(o, m, 0.0)
            elif method == "a":
                decider = lambda tid, o, m: policy._core(tid).act(o, m, 0.0)
            else:
                decider = lambda e, dbg: policy.act(e, dbg, 0.0)
            agg, _ = evaluate_configs(method, decider, eval_cfgs)
            wall = time.perf_counter() - t_start
            _append_row(curve_path, header,
                        [method, seed, step + 1,
                         round(agg["reward"], 2), round(agg["mean_duration"], 2),
                         round(agg["throughput"], 2), round(agg["queue_wait"], 3),
                         round(agg["fuel"], 4), round(agg["balance"], 3),
                         round(wall, 1)])
            print(f"  [{method}|s{seed}] {step+1:>7,} trips  "
                  f"evalR={agg['reward']:.1f}  dur={agg['mean_duration']:.0f}s  "
                  f"tph={agg['throughput']:.0f}  ({wall:.0f}s)", flush=True)
    return str(curve_path)


def evaluate_baselines(cfg: Cfg):
    t0 = time.perf_counter()
    out = OUT_DIR / "baselines_eval.csv"
    if out.exists():
        out.unlink()
    header = ["baseline", "cfg", "fleet", "coal_scale", "reward",
              "mean_duration", "throughput", "queue_wait", "fuel", "balance"]
    for name, factory in BASELINE_FACTORIES.items():
        for i, c in enumerate(EVAL_CONFIGS[: cfg.eval_configs]):
            env = DispatchEnv(mode="central", fleet=c["fleet"],
                              coal_scale=c["coal_scale"], master_seed=c["seed"])
            m = run_episode(env, "baseline", factory())
            _append_row(out, header,
                        [name, i, c["fleet"], c["coal_scale"], round(m["reward"], 2),
                         round(m["mean_duration"], 2), round(m["throughput"], 2),
                         round(m["queue_wait"], 3), round(m["fuel"], 4),
                         round(m["balance"], 3)])
        print(f"  baseline {name:<15} done", flush=True)
    print(f"baselines -> {out}  ({time.perf_counter()-t0:.0f}s)")
    return str(out)


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def _job(args):
    method, seed, cfg = args
    return train_run(method, seed, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="b",
                    help="b | a | c | d | all | baselines")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--trips", type=int, default=100_000)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-configs", type=int, default=20)
    ap.add_argument("--episode-cap", type=int, default=200)
    ap.add_argument("--device", default="cuda" if _cuda_ok() else "cpu")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Cfg(trips=a.trips, eval_every=a.eval_every, episode_cap=a.episode_cap,
              device=a.device, eval_configs=a.eval_configs)
    print(f"out_dir={OUT_DIR}  device={cfg.device}  trips={cfg.trips:,}")

    if a.method == "baselines":
        evaluate_baselines(cfg)
        return

    methods = ["b", "a", "c", "d"] if a.method == "all" else [a.method]
    jobs = [(m, s, cfg) for m in methods for s in range(a.seed, a.seed + a.seeds)]

    if a.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for path in ex.map(_job, jobs):
                print(f"finished -> {path}")
    else:
        for j in jobs:
            _job(j)


def _cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
