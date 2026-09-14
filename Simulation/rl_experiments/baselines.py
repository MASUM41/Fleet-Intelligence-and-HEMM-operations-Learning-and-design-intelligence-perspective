"""
rl_experiments/baselines.py — the 6 hand-built dispatch policies.

Reference floor for the architecture shootout (0 training trips).
Each baseline is a *factory*: make_policy() -> choose(env, dbg) -> mine_name,
so stateful policies (ROUND_ROBIN) get a fresh counter per episode.

Same policy family as dispatch_comparator, re-expressed against the live
calendar state of DispatchEnv (env.debug()).
"""

from __future__ import annotations


def _active(dbg):
    return {m: v for m, v in dbg["mines"].items() if v["coal"] > 0}


def make_greedy_nearest():
    return lambda env, dbg: min(_active(dbg), key=lambda m: (
        dbg["mines"][m]["travel"] + dbg["mines"][m]["wait"] + dbg["mines"][m]["cycle"]))


def make_eta_min():
    return lambda env, dbg: min(_active(dbg), key=lambda m: (
        dbg["mines"][m]["cycle"] + dbg["mines"][m]["wait"]))


def make_eta_aware():
    return lambda env, dbg: min(_active(dbg), key=lambda m: (
        dbg["mines"][m]["cycle"] + dbg["mines"][m]["wait"] + env.eta_bias()))


def make_fifo():
    return lambda env, dbg: min(_active(dbg), key=lambda m: dbg["mines"][m]["last_assigned"])


def make_load_balance():
    return lambda env, dbg: min(_active(dbg), key=lambda m: (
        dbg["mines"][m]["total_trips"] + dbg["mines"][m]["pending"]))


def make_round_robin():
    state = {"i": 0}

    def choose(env, dbg):
        act = sorted(_active(dbg))
        m = act[state["i"] % len(act)]
        state["i"] += 1
        return m

    return choose


BASELINE_FACTORIES = {
    "GREEDY_NEAREST": make_greedy_nearest,
    "ETA_MIN": make_eta_min,
    "ETA_AWARE": make_eta_aware,
    "FIFO": make_fifo,
    "LOAD_BALANCE": make_load_balance,
    "ROUND_ROBIN": make_round_robin,
}
