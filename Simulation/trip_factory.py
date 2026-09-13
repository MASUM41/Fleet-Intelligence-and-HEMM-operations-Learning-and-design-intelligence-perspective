"""
trip_factory.py  --  "Calendar simulator": bulk trip generation without physics.

WHY THIS EXISTS
---------------
The Pygame/MPC sim turns a trip into ~14,000 physics steps + rendering
(~77 wall-seconds/trip).  But for DISPATCH learning, a trip is only 5 numbers:

    travel_to_mine + queue_wait + service + travel_to_dump + unload

This module simulates trips with just that arithmetic:
  * a TIMETABLE  (per-route medians, calibrated on analysis/trips_graph.csv)
  * a CALENDAR   (each truck has a "free_at" time; event priority queue)
  * a CLIPBOARD  (each mine/dump: coal left + next-free-service time)

No rendering, no steering, no MPC  ->  ~1 ms per trip instead of ~77 s.
That is the speed class an RL training loop needs (~100k trips).

Usage (from Simulation/):
    python trip_factory.py --trips 10000 --fleet 3
    python trip_factory.py --trips 10000 --fleet 8 --out analysis/synth_f8.csv
"""

import argparse
import heapq
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- service constants (match Simulation/config.py) ---
LOAD_SERVICE_S = 3.0
UNLOAD_SERVICE_S = 3.0
TURN_S = 2.0                      # K-turn bookkeeping observed in real trips
CARGO_KG = 1000.0

TRIPS_GRAPH = "analysis/trips_graph.csv"
TRIPS_REAL = "analysis/trips.csv"
MINE_CFG = "Map/mine_config.json"


# ─────────────────────────────────────────────────────────────────────────────
#  TIMETABLE — per-route medians fitted from the real 182 logged trips
# ─────────────────────────────────────────────────────────────────────────────

class Timetable:
    def __init__(self, path=TRIPS_GRAPH):
        df = pd.read_csv(path).dropna(subset=["total_trip_duration_s"])
        df = df[df["total_trip_duration_s"] < 600]          # drop queue outliers
        self.routes = {}
        for (lz, dz), g in df.groupby(["load_zone", "dump_zone"]):
            total = g["total_trip_duration_s"].median()
            to_dump = g["travel_to_dump_s"].median()
            load_w = g["load_wait_s"].median()
            unl_w = g["unload_wait_s"].median()
            # empty leg is not directly logged (depart_time NaN) -> recover it:
            to_mine = max(10.0, total - to_dump - load_w - unl_w)
            self.routes[(lz, dz)] = dict(
                to_mine=float(to_mine), load_wait=float(load_w),
                to_dump=float(to_dump), unload_wait=float(unl_w),
                fuel=float(g["fuel_L"].median()),
                dist=float(g["total_road_dist_m"].median())
                if "total_road_dist_m" in g and g["total_road_dist_m"].notna().any()
                else 500.0,
                total=float(total),
            )
        self.mines = sorted(df["load_zone"].unique())
        self.dumps = sorted(df["dump_zone"].unique())
        # best-pair dump per mine (same convention as dispatch_comparator)
        self.best_dump = {
            m: min((k for k in self.routes if k[0] == m),
                   key=lambda k: self.routes[k]["total"])[1]
            for m in self.mines
        }
        # global fallbacks for unseen pairs
        self._fb = dict(to_mine=140.0, load_wait=3.0, to_dump=42.0,
                        unload_wait=3.0, fuel=0.65, dist=500.0, total=231.0)

    def route(self, mine):
        dump = self.best_dump[mine]
        return dump, self.routes.get((mine, dump), self._fb)

    def sample(self, mine, rng):
        """One stochastic outcome for a trip to `mine` (best-pair dump)."""
        dump, r = self.route(mine)
        jit = lambda v, s: max(2.5, v * (1.0 + rng.normal(0.0, s)))
        return dict(
            dump=dump,
            to_mine=jit(r["to_mine"], 0.08),
            to_dump=jit(r["to_dump"], 0.08),
            service_load=jit(LOAD_SERVICE_S, 0.05),
            service_dump=jit(UNLOAD_SERVICE_S, 0.05),
            fuel=max(0.05, r["fuel"] * (1.0 + rng.normal(0.0, 0.10))),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  STATE — calendar + clipboard
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MineState:
    coal: float
    next_free: float = 0.0        # when the loader will next be free
    en_route: int = 0

@dataclass
class DumpState:
    next_free: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  CORE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class TripFactory:
    def __init__(self, fleet=3, episode_cap=200, seed=42, coal_cfg=MINE_CFG):
        self.tt = Timetable()
        self.rng = np.random.default_rng(seed)
        self.fleet = fleet
        self.episode_cap = episode_cap
        with open(coal_cfg) as f:
            caps = json.load(f)["coal_capacities"]
        self.coal_caps = {m: float(caps.get(m, 10000.0)) for m in self.tt.mines}
        self.rows = []

    # ---- dispatch policy for data generation: calendar-aware greedy ----
    def _greedy_pick(self, now, mines, dumps):
        best, best_score = None, float("inf")
        for m_name, m in mines.items():
            if m.coal <= 0:
                continue                                     # action masking
            dump, r = self.tt.route(m_name)
            arrive = now + r["to_mine"]
            wait = max(0.0, m.next_free - arrive)            # calendar foresight
            score = (r["to_mine"] + wait + r["load_wait"]
                     + r["to_dump"] + r["unload_wait"])
            if score < best_score:
                best_score, best = score, m_name
        return best

    def run(self, total_trips):
        trips_done, episode = 0, 0
        while trips_done < total_trips:
            episode += 1
            mines = {m: MineState(self.coal_caps[m]) for m in self.tt.mines}
            dumps = {d: DumpState() for d in self.tt.dumps}
            # event queue: (time, truck_id, location_node)
            heap = [(self.rng.uniform(0, 30) + 5 * i, i,
                     self.rng.choice(self.tt.dumps)) for i in range(self.fleet)]
            heapq.heapify(heap)
            ep_done = 0

            while heap and ep_done < self.episode_cap and trips_done < total_trips:
                t, tid, node = heapq.heappop(heap)
                mine_name = self._greedy_pick(t, mines, dumps)
                if mine_name is None:
                    break                                    # all mines depleted
                mine = mines[mine_name]
                s = self.tt.sample(mine_name, self.rng)
                dump_st = dumps[s["dump"]]

                # --- the whole trip, as arithmetic ---
                depart = t
                arrive_load = depart + s["to_mine"]
                service_start = max(arrive_load, mine.next_free)
                queue_wait = service_start - arrive_load
                load_complete = service_start + s["service_load"]
                mine.next_free = load_complete               # FCFS booking
                mine.coal = max(0.0, mine.coal - CARGO_KG)
                depart_load = load_complete + TURN_S
                arrive_dump = depart_load + s["to_dump"]
                dump_start = max(arrive_dump, dump_st.next_free)
                unload_wait = dump_start - arrive_dump
                unload_done = dump_start + s["service_dump"]
                dump_st.next_free = unload_done

                trips_done += 1
                ep_done += 1
                self.rows.append(dict(
                    trip_id=trips_done, run_id=episode, truck_id=tid,
                    load_zone=mine_name, dump_zone=s["dump"],
                    depart_time_s=round(depart, 2),
                    arrive_load_s=round(arrive_load, 2),
                    load_complete_s=round(load_complete, 2),
                    depart_load_s=round(depart_load, 2),
                    arrive_dump_s=round(arrive_dump, 2),
                    unload_complete_s=round(unload_done, 2),
                    travel_to_mine_s=round(s["to_mine"], 2),
                    load_wait_s=round(queue_wait + s["service_load"], 2),
                    queue_wait_service_s=round(queue_wait, 2),
                    travel_to_dump_s=round(s["to_dump"], 2),
                    unload_wait_s=round(unload_wait + s["service_dump"], 2),
                    total_trip_duration_s=round(unload_done - depart, 2),
                    cargo_kg=CARGO_KG,
                    fuel_L=round(s["fuel"], 4),
                    co2_kg=round(s["fuel"] * 2.68, 4),
                ))
                heapq.heappush(heap, (unload_done, tid, s["dump"]))

        return pd.DataFrame(self.rows)


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION — synthetic vs real, side by side
# ─────────────────────────────────────────────────────────────────────────────

def report(df_synth, wall_s):
    print(f"\n{'='*66}")
    print(f"  GENERATED {len(df_synth):,} trips in {wall_s*1000:.0f} ms"
          f"  ({len(df_synth)/wall_s:,.0f} trips/s, {len(df_synth)/wall_s/60:,.0f}/min)")
    print(f"{'='*66}")

    real = pd.read_csv(TRIPS_REAL)
    real = real[real["total_trip_duration_s"] < 600]
    cmp_tbl = pd.DataFrame({
        "real": real["total_trip_duration_s"].describe()[["mean", "50%", "std", "min", "max"]],
        "synthetic": df_synth["total_trip_duration_s"].describe()[["mean", "50%", "std", "min", "max"]],
    }).round(1)
    print("\nTrip duration distribution (real 182 trips vs synthetic):")
    print(cmp_tbl.to_string())

    qw = df_synth["queue_wait_service_s"]
    print(f"\nQueue wait beyond service: mean={qw.mean():.2f}s  p95={qw.quantile(.95):.2f}s  max={qw.max():.1f}s")
    print(f"Fuel/trip: synth={df_synth['fuel_L'].mean():.3f} L   real={real['fuel_L'].mean():.3f} L")
    print("\nMine usage (synthetic): ", df_synth["load_zone"].value_counts().head(6).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", type=int, default=10000)
    ap.add_argument("--fleet", type=int, default=3)
    ap.add_argument("--episode-cap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="analysis/synthetic_trips_10k.csv")
    a = ap.parse_args()

    print(f"TripFactory: generating {a.trips:,} trips "
          f"(fleet={a.fleet}, episode cap={a.episode_cap}, seed={a.seed}) ...")
    t0 = time.perf_counter()
    df = TripFactory(a.fleet, a.episode_cap, a.seed).run(a.trips)
    wall = time.perf_counter() - t0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    report(df, wall)
    print(f"\nSaved -> {a.out}")


if __name__ == "__main__":
    main()
