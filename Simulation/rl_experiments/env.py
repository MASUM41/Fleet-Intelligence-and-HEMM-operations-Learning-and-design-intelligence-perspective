"""
rl_experiments/env.py — DispatchEnv: event-driven dispatch MDP.

Wraps the "calendar simulator" arithmetic (trip_factory.Timetable) into a
step-based environment shared by ALL agent framings so the architecture
shootout is fair:

    reset() -> obs, info          # fresh episode: coal refilled, fleet sampled
    step(action) -> obs, reward, terminated, truncated, info

Modes:
    central    : action = mine index              (B one-for-all, A per-truck, C per-mine)
    universal  : action = joint (mine, dump)      (D)

Reward (identical for every framing):  r = -trip_duration_s / 100

Observation builders (all float32, ~[0,1] scaled):
    observe("central")   49-dim global fleet state        (B)
    observe("local")     38-dim per-truck view            (A)
    observe_mine(j)       7-dim per-mine view             (C)
    observe("universal") 55-dim global + dump features    (D)

Action masking: depleted mines (coal<=0) are invalid in every framing.
"""

from __future__ import annotations

import heapq
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SIM_DIR = Path(__file__).resolve().parents[1]
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from trip_factory import (  # noqa: E402
    Timetable, LOAD_SERVICE_S, UNLOAD_SERVICE_S, TURN_S, CARGO_KG,
)

# Bid levels (seconds added to a mine's score) for the per-mine auction (C)
BID_GRID = [-80.0, -40.0, 0.0, 40.0, 80.0]


# ═══════════════════════════════════════════════════════════════════════
#  Road network estimator — realistic times for UNSEEN (mine, dump) pairs
# ═══════════════════════════════════════════════════════════════════════
#  Logged pairs keep their real medians. Unseen pairs previously fell back
#  to a flat 42 s — systematically SHORT, which the universal agent (D)
#  learned to exploit. Here we instead compute shortest-path distances on
#  the road graph and divide by calibrated effective speeds.

class RoadNet:
    def __init__(self, sim_dir, mines, dumps, timetable):
        import pandas as pd
        self.tt = timetable
        self.mines, self.dumps = mines, dumps

        md = json.load(open(sim_dir / "Map" / "map_data.json"))
        nodes = md["NODES"]
        adj = {n: [] for n in nodes}
        for a, b in md["EDGES"]:
            if a in nodes and b in nodes:
                w = float(np.hypot(nodes[a][0] - nodes[b][0],
                                   nodes[a][1] - nodes[b][1]))
                adj[a].append((b, w))
                adj[b].append((a, w))
        self._nodes, self._adj = nodes, adj

        # Dijkstra from every zone we care about (9 mines + 6 dumps)
        self.dist = {}
        for src in set(self.mines) | set(self.dumps):
            self.dist[src] = self._dijkstra(src)

        # Calibrate effective speeds from the real logged trips
        tg = pd.read_csv(sim_dir / "analysis" / "trips_graph.csv")
        tg = tg[tg["total_trip_duration_s"] < 600].copy()
        to_mine = (tg["total_trip_duration_s"] - tg["travel_to_dump_s"]
                   - tg["load_wait_s"] - tg["unload_wait_s"]).clip(lower=10.0)
        self.empty_speed = float(np.median(tg["road_dist_to_mine_m"] / to_mine))
        self.loaded_speed = float(np.median(tg["road_dist_to_dump_m"] / tg["travel_to_dump_s"]))
        self.fuel_per_m = float(tg["fuel_L"].sum() / max(tg["total_road_dist_m"].sum(), 1.0))

    def _dijkstra(self, src):
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            for v, w in self._adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def path_dist(self, a, b):
        d = self.dist.get(a, {}).get(b)
        if d is None:  # disconnected fallback: euclidean x detour factor
            pa, pb = self._nodes[a], self._nodes[b]
            d = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) * 1.4
        return d

    def to_mine(self, node, mine):
        r = self.tt.routes.get((mine, node))
        if r is not None:
            return r["to_mine"]                      # real median wins
        return self.path_dist(node, mine) / max(self.empty_speed, 0.3)

    def to_dump(self, mine, dump):
        r = self.tt.routes.get((mine, dump))
        if r is not None:
            return r["to_dump"]                      # real median wins
        return self.path_dist(mine, dump) / max(self.loaded_speed, 0.3)

    def fuel(self, mine, dump):
        r = self.tt.routes.get((mine, dump))
        if r is not None:
            return r["fuel"]
        return self.path_dist(mine, dump) * self.fuel_per_m


@dataclass
class MineSt:
    coal: float
    next_free: float = 0.0
    pending: list = field(default_factory=list)   # heap of arrival times (dispatched, not yet arrived)
    total_trips: int = 0
    last_assigned: float = -1e9


@dataclass
class DumpSt:
    next_free: float = 0.0


class DispatchEnv:
    """One decision per newly-empty truck. The MDP is event-driven (Semi-MDP)."""

    def __init__(self, mode="central", fleet=None, episode_cap=200,
                 coal_scale=1.0, master_seed=0):
        assert mode in ("central", "universal")
        self.mode = mode
        self.tt = Timetable()
        self.mines = list(self.tt.mines)          # 9 logged mines
        self.dumps = list(self.tt.dumps)          # 6 logged dumps
        self.M, self.D = len(self.mines), len(self.dumps)
        self.episode_cap = episode_cap
        self.coal_scale = coal_scale
        self.fleet_fixed = fleet
        self._episode_seed = master_seed

        with open(_SIM_DIR / "Map" / "mine_config.json") as f:
            caps = json.load(f)["coal_capacities"]
        self.coal_caps = {m: float(caps.get(m, 10000.0)) * coal_scale for m in self.mines}

        self.road = RoadNet(_SIM_DIR, self.mines, self.dumps, self.tt)
        self.n_actions = self.M if mode == "central" else self.M * self.D

        # Universal mode: joint actions are restricted to LOGGED (mine, dump)
        # pairs. Graph-estimated pairs are allowed for travel-time lookups but
        # NOT as actions — otherwise D exploits optimistic unseen-pair times.
        self._logged_pair = np.zeros((self.M, self.D), dtype=np.float32)
        for mi, m in enumerate(self.mines):
            for di, d in enumerate(self.dumps):
                if (m, d) in self.tt.routes:
                    self._logged_pair[mi, di] = 1.0
        self.cur = None   # (t, truck_id, node) awaiting a decision

    # ------------------------------------------------------------------ utils
    def _travel(self, from_node, mine):
        """Empty-leg time from a dump node to a mine (real median or graph estimate)."""
        return self.road.to_mine(from_node, mine)

    def _cycle(self, mine):
        return self.tt.route(mine)[1]["total"]

    def _active(self):
        return [m for m in self.mines if self.mine_st[m].coal > 0]

    def _next_pending(self):
        t, _, tid, node = heapq.heappop(self.heap)
        self.sim_time = t
        self.cur = (t, tid, node)
        for m in self.mines:                       # expire arrived trucks
            p = self.mine_st[m].pending
            while p and p[0] <= t:
                heapq.heappop(p)

    # ------------------------------------------------------------------- API
    def reset(self, seed=None):
        self.rng = np.random.default_rng(
            seed if seed is not None else self._episode_seed)
        self._episode_seed += 1

        self.mine_st = {m: MineSt(self.coal_caps[m]) for m in self.mines}
        self.dump_st = {d: DumpSt() for d in self.dumps}
        self.fleet = self.fleet_fixed or int(self.rng.integers(4, 13))

        self.heap = []
        self._ctr = itertools.count()
        for i in range(self.fleet):
            t0 = self.rng.uniform(0, 30) + 5 * i
            node = self.dumps[int(self.rng.integers(0, self.D))]
            heapq.heappush(self.heap, (t0, next(self._ctr), i, node))

        self.trips = 0
        self.sim_time = 0.0
        self.ep_reward = 0.0
        self.trip_log = []        # per-trip dicts (for eval metrics)
        self.expected_log = []    # (expected_duration, actual) for ETA_AWARE bias
        self._next_pending()
        return self.observe(), self.info()

    def step(self, action):
        t, tid, node = self.cur
        if self.mode == "central":
            mine, dump = self.mines[int(action)], None
        else:
            mi, di = divmod(int(action), self.D)
            mine, dump = self.mines[mi], self.dumps[di]
        mst = self.mine_st[mine]

        # expected duration (deterministic part) — logged for ETA_AWARE bias
        travel_est = self._travel(node, mine)
        wait_est = max(0.0, mst.next_free - (t + travel_est))
        expected = travel_est + wait_est + self._cycle(mine)

        # ---- stochastic outcome -------------------------------------------
        jit = lambda v, sg: max(2.5, v * (1.0 + self.rng.normal(0.0, sg)))
        to_mine = jit(self._travel(node, mine), 0.08)

        if dump is None:
            dump = self.tt.route(mine)[0]             # best-pair dump
        to_dump = jit(self.road.to_dump(mine, dump), 0.08)   # real or graph-est.
        sl = jit(LOAD_SERVICE_S, 0.05)
        sd = jit(UNLOAD_SERVICE_S, 0.05)
        fuel = max(0.05, self.road.fuel(mine, dump) * (1.0 + self.rng.normal(0.0, 0.10)))

        # ---- the whole trip as arithmetic ---------------------------------
        arrive = t + to_mine
        start = max(arrive, mst.next_free)
        wait = start - arrive
        load_done = start + sl
        mst.next_free = load_done                    # book the loader (FCFS)
        mst.coal = max(0.0, mst.coal - CARGO_KG)
        mst.total_trips += 1
        mst.last_assigned = t
        heapq.heappush(mst.pending, arrive)

        depart_load = load_done + TURN_S
        arrive_dump = depart_load + to_dump
        dst = self.dump_st[dump]
        dstart = max(arrive_dump, dst.next_free)
        done_t = dstart + sd
        dst.next_free = done_t

        duration = done_t - t
        reward = -duration / 100.0
        self.ep_reward += reward
        self.trip_log.append(dict(trip=self.trips + 1, truck=tid, mine=mine,
                                  dump=dump, depart=t, done=done_t,
                                  duration=duration, wait=wait, fuel=fuel))
        self.expected_log.append((expected, duration))

        heapq.heappush(self.heap, (done_t, next(self._ctr), tid, dump))
        self.trips += 1

        terminated = (self.trips >= self.episode_cap
                      or not self._active())
        if not terminated:
            self._next_pending()
        return self.observe(), reward, terminated, False, self.info()

    # ------------------------------------------------------------ observation
    def observe(self, kind=None):
        kind = kind or ("universal" if self.mode == "universal" else "central")
        if kind == "central":
            return self._obs_central()
        if kind == "universal":
            return np.concatenate([self._obs_central(), self._obs_dumps()])
        if kind == "local":
            return self._obs_local()
        raise ValueError(kind)

    def _obs_central(self):
        t, tid, node = self.cur
        feats = []
        for m in self.mines:
            ms = self.mine_st[m]
            travel = self._travel(node, m)
            wait = max(0.0, ms.next_free - (t + travel))
            feats += [ms.coal / 10000.0,
                      min(wait, 200.0) / 200.0,
                      min(len(ms.pending), 4) / 4.0,
                      self._cycle(m) / 300.0,
                      min(travel, 300.0) / 300.0]
        busy = sum(1 for (_, _, _, _) in self.heap) / 12.0   # trucks mid-trip
        feats += [self.fleet / 12.0, busy,
                  self.trips / self.episode_cap, min(self.sim_time, 2e4) / 2e4]
        return np.asarray(feats, dtype=np.float32)

    def _obs_dumps(self):
        t, _, node = self.cur
        out = []
        for d in self.dumps:
            ds = self.dump_st[d]
            out.append(min(max(0.0, ds.next_free - t), 200.0) / 200.0)
        return np.asarray(out, dtype=np.float32)

    def _obs_local(self):
        """Per-truck view: own node + public signboards (coal & queue counts)."""
        t, tid, node = self.cur
        feats = []
        for m in self.mines:
            ms = self.mine_st[m]
            travel = self._travel(node, m)
            queue_display = len(ms.pending) + (1.0 if ms.next_free > t else 0.0)
            feats += [min(travel, 300.0) / 300.0,
                      ms.coal / 10000.0,
                      min(queue_display, 4) / 4.0,
                      self._cycle(m) / 300.0]
        feats += [self.trips / self.episode_cap, tid / 12.0]
        return np.asarray(feats, dtype=np.float32)

    def observe_mine(self, j):
        """Per-mine view for the auction framing (C)."""
        t, tid, node = self.cur
        m = self.mines[j]
        ms = self.mine_st[m]
        travel = self._travel(node, m)
        wait = max(0.0, ms.next_free - (t + travel))
        total = sum(self.mine_st[x].total_trips for x in self.mines) + 1e-9
        return np.asarray([
            ms.coal / 10000.0,
            min(wait, 200.0) / 200.0,
            min(len(ms.pending), 4) / 4.0,
            ms.total_trips / total,
            min(travel, 300.0) / 300.0,
            self.trips / self.episode_cap,
            min(self.sim_time, 2e4) / 2e4,
        ], dtype=np.float32)

    # ---------------------------------------------------------------- masks
    def action_mask(self):
        act = np.array([1.0 if self.mine_st[m].coal > 0 else 0.0
                        for m in self.mines], dtype=np.float32)
        if self.mode == "central":
            return act
        joint = act[:, None] * self._logged_pair      # (M, D) mine-major
        return joint.reshape(-1).astype(np.float32)

    def info(self):
        t, tid, node = self.cur
        return dict(truck_id=tid, node=node, sim_time=t,
                    mask=self.action_mask(), fleet=self.fleet)

    # ---------------------------------------------------- baselines support
    def debug(self):
        """Deterministic per-mine state for hand policies and the C buyer rule."""
        t, tid, node = self.cur
        out = {"t": t, "truck_id": tid, "node": node, "mines": {}}
        for m in self.mines:
            ms = self.mine_st[m]
            travel = self._travel(node, m)
            wait = max(0.0, ms.next_free - (t + travel))
            out["mines"][m] = dict(coal=ms.coal, travel=travel, wait=wait,
                                   cycle=self._cycle(m),
                                   total_trips=ms.total_trips,
                                   last_assigned=ms.last_assigned,
                                   pending=len(ms.pending))
        return out

    def eta_bias(self):
        """Median (actual - expected) over the episode so far (ETA_AWARE)."""
        if len(self.expected_log) < 5:
            return 0.0
        res = [a - e for e, a in self.expected_log[-50:]]
        return float(np.median(res))

    # ------------------------------------------------------------- metrics
    def episode_metrics(self):
        if not self.trip_log:
            return {}
        durs = [x["duration"] for x in self.trip_log]
        waits = [x["wait"] for x in self.trip_log]
        fuels = [x["fuel"] for x in self.trip_log]
        counts = {}
        for x in self.trip_log:
            counts[x["mine"]] = counts.get(x["mine"], 0) + 1
        makespan = max(x["done"] for x in self.trip_log) - min(x["depart"] for x in self.trip_log)
        return dict(
            trips=len(self.trip_log),
            reward=self.ep_reward,
            mean_duration=float(np.mean(durs)),
            throughput=float(len(self.trip_log) / max(makespan, 1.0) * 3600.0),
            queue_wait=float(np.mean(waits)),
            fuel=float(np.mean(fuels)),
            balance=float(np.std(list(counts.values()))),
        )
