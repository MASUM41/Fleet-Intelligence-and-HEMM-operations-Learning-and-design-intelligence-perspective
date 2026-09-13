"""
dispatch_comparator.py  -  Event-driven offline dispatch policy evaluator.

Builds dispatch moments from logged trips/events, then replays fleet
assignments under multiple policies using:
    - real truck positions (last dump zone, DEPART_DUMP_ZONE events)
    - variable dump-zone assignment via route lookup
    - historical route duration/fuel medians
    - queue congestion context from queue_events.csv

Policies:
    1. GREEDY_NEAREST   - travel time + queue + predicted cycle time
    2. FIFO             - longest waiting mine
    3. LOAD_BALANCE     - fewest trips per mine
    4. ROUND_ROBIN      - fixed rotation
    5. ETA_MIN          - lowest predicted full-cycle time (route lookup)
    6. ETA_AWARE        - ETA-min with learned dispatch-time correction

Run from Simulation/ folder:
    python dispatch_comparator.py
"""

import glob
import os
import warnings
from collections import defaultdict

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

OUT_DIR = "analysis"
LOG_DIR = "logs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

C_GREEN = "#1A6B3A"
C_BLUE = "#2E86AB"
C_ORANGE = "#F4A261"
C_RED = "#E63946"
C_PURPLE = "#9B59B6"
C_TEAL = "#1ABC9C"
C_GRAY = "#AAAAAA"

POLICY_COLORS = {
    "GREEDY_NEAREST": C_BLUE,
    "FIFO": C_GREEN,
    "LOAD_BALANCE": C_ORANGE,
    "ROUND_ROBIN": C_PURPLE,
    "ETA_MIN": C_TEAL,
    "ETA_AWARE": C_RED,
}

LOAD_TIME_S = 3.5
UNLOAD_TIME_S = 3.5
DEFAULT_SPEED_MS = 6.94


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PREP
# ══════════════════════════════════════════════════════════════════════════════

def load_events_path():
    merged = os.path.join(LOG_DIR, "merged_events.csv")
    if os.path.exists(merged):
        return merged
    files = sorted(glob.glob(os.path.join(LOG_DIR, "events_*.csv")))
    return files[-1] if files else None


def build_road_distance_matrix(trips_df):
    road_dist = {}
    if "road_dist_to_dump_m" in trips_df.columns:
        for _, row in trips_df.iterrows():
            lz, dz = row["load_zone"], row["dump_zone"]
            d = float(row.get("road_dist_to_dump_m", 530.0))
            road_dist[(lz, dz)] = d
            road_dist[(dz, lz)] = d
        return road_dist

    for _, row in trips_df.iterrows():
        lz, dz = row["load_zone"], row["dump_zone"]
        if (lz, dz) not in road_dist:
            tt = float(row.get("travel_to_dump_s", 76.5) or 76.5)
            d = tt * DEFAULT_SPEED_MS
            road_dist[(lz, dz)] = d
            road_dist[(dz, lz)] = d
    return road_dist


class RouteLookup:
    """Historical route stats for duration/fuel estimation."""

    def __init__(self, trips_df, road_dist_matrix):
        self.road_dist = road_dist_matrix
        self.avg_speed = DEFAULT_SPEED_MS
        normal = trips_df[trips_df["total_trip_duration_s"] < 600].copy()

        if "total_road_dist_m" in normal.columns and normal["total_road_dist_m"].sum() > 0:
            self.avg_fuel_per_m = normal["fuel_L"].sum() / normal["total_road_dist_m"].sum()
        else:
            self.avg_fuel_per_m = 0.0025

        self.route = (
            normal.groupby(["load_zone", "dump_zone"], as_index=False)
            .agg(
                duration=("total_trip_duration_s", "median"),
                fuel=("fuel_L", "median"),
                co2=("co2_kg", "median"),
            )
        )

        if "travel_to_mine_s" in normal.columns:
            self.empty_leg = (
                normal.dropna(subset=["travel_to_mine_s"])
                .groupby(["dump_zone", "load_zone"], as_index=False)
                .agg(travel=("travel_to_mine_s", "median"))
            )
        else:
            self.empty_leg = pd.DataFrame(columns=["dump_zone", "load_zone", "travel"])

        self.dumps_by_mine = (
            normal.groupby("load_zone")["dump_zone"].apply(lambda s: sorted(s.unique())).to_dict()
        )
        self.all_dumps = sorted(normal["dump_zone"].dropna().unique().tolist())
        self.all_mines = sorted(normal["load_zone"].dropna().unique().tolist())

    def best_dump_for_mine(self, mine):
        sub = self.route[self.route["load_zone"] == mine]
        if sub.empty:
            return self.all_dumps[0] if self.all_dumps else "dump_zone_3"
        return sub.sort_values("duration").iloc[0]["dump_zone"]

    def estimate_trip(self, from_node, load_zone, dump_zone, congestion=1.0):
        sub = self.route[
            (self.route["load_zone"] == load_zone) & (self.route["dump_zone"] == dump_zone)
        ]
        if not sub.empty:
            base_dur = float(sub.iloc[0]["duration"])
            base_fuel = float(sub.iloc[0]["fuel"])
            base_co2 = float(sub.iloc[0]["co2"])
        else:
            d1 = self.road_dist.get((from_node, load_zone), self.road_dist.get((load_zone, from_node), 530.0))
            d2 = self.road_dist.get((load_zone, dump_zone), self.road_dist.get((dump_zone, load_zone), 530.0))
            total_dist = d1 + d2
            base_dur = total_dist / self.avg_speed + LOAD_TIME_S + UNLOAD_TIME_S
            base_fuel = total_dist * self.avg_fuel_per_m
            base_co2 = base_fuel * 2.68

        if not self.empty_leg.empty:
            route_rows = self.route[self.route["load_zone"] == load_zone]
            typical_empty = (
                normal_median
                if (normal_median := route_rows.merge(
                    self.empty_leg, on=["load_zone"], how="left", suffixes=("", "_el")
                )["travel"].median())
                is not None
                else np.nan
            )
            el = self.empty_leg[
                (self.empty_leg["dump_zone"] == from_node) & (self.empty_leg["load_zone"] == load_zone)
            ]
            if not el.empty and pd.notna(typical_empty) and typical_empty > 0:
                ratio = float(el.iloc[0]["travel"]) / float(typical_empty)
                ratio = float(np.clip(ratio, 0.6, 2.5))
                empty_share = 0.35
                base_dur = base_dur * ((1 - empty_share) + empty_share * ratio)
                base_fuel = base_fuel * ((1 - empty_share) + empty_share * ratio)
                base_co2 = base_co2 * ((1 - empty_share) + empty_share * ratio)

        dur = base_dur * congestion
        fuel = base_fuel * congestion
        co2 = base_co2 * congestion
        return dur, fuel, co2

    def estimate_best_assignment(self, from_node, load_zone, congestion=1.0):
        dumps = self.dumps_by_mine.get(load_zone, self.all_dumps)
        best_dump, best_dur, best_fuel, best_co2 = None, float("inf"), 0.0, 0.0
        for dump_zone in dumps:
            dur, fuel, co2 = self.estimate_trip(from_node, load_zone, dump_zone, congestion)
            if dur < best_dur:
                best_dump, best_dur, best_fuel, best_co2 = dump_zone, dur, fuel, co2
        if best_dump is None:
            best_dump = self.best_dump_for_mine(load_zone)
            best_dur, best_fuel, best_co2 = self.estimate_trip(
                from_node, load_zone, best_dump, congestion
            )
        return best_dump, best_dur, best_fuel, best_co2


class QueueIndex:
    """Nearest queue snapshot by zone and simulation time."""

    def __init__(self, queue_df):
        self.load_events = queue_df[queue_df["event_type"] == "ARRIVE_LOAD_ZONE"].copy()
        self.load_events["sim_time_s"] = pd.to_numeric(self.load_events["sim_time_s"], errors="coerce")
        self.load_events["trucks_queued"] = pd.to_numeric(
            self.load_events["trucks_queued"], errors="coerce"
        )

    def depth_at(self, zone, t, tolerance_s=30.0):
        if self.load_events.empty or pd.isna(t):
            return 2.0
        sub = self.load_events[self.load_events["zone"] == zone]
        if sub.empty:
            return 2.0
        diffs = (sub["sim_time_s"] - t).abs()
        idx = diffs.idxmin()
        if diffs.loc[idx] <= tolerance_s:
            return float(sub.loc[idx, "trucks_queued"])
        return 2.0

    def congestion_factor(self, zone, t):
        depth = self.depth_at(zone, t)
        return 1.0 + max(0.0, depth - 2.0) * 0.08


class ETAAwareEstimator:
    """
    Lightweight dispatch-time ETA correction model.

    Uses historical residuals between actual and route-lookup duration,
    conditioned by mine and truck. This keeps the comparator event-driven
    while making ETA-based assignment more context-aware than route medians.
    """

    def __init__(self, trips_df, dispatch_events, route_lookup, queue_index):
        self.route_lookup = route_lookup
        self.queue_index = queue_index
        self.global_bias = 0.0
        self.mine_bias = {}
        self.truck_bias = {}
        self._fit(trips_df, dispatch_events)

    def _fit(self, trips_df, dispatch_events):
        if trips_df.empty or dispatch_events.empty:
            return

        events_small = dispatch_events[["trip_id", "current_node", "sim_time_s"]].copy()
        train = trips_df.merge(events_small, on="trip_id", how="left")
        if train.empty:
            return

        residuals = []
        mine_keys = []
        truck_keys = []
        for _, row in train.iterrows():
            load_zone = row.get("load_zone")
            dump_zone = row.get("dump_zone")
            if pd.isna(load_zone) or pd.isna(dump_zone):
                continue
            current_node = row.get("current_node")
            if pd.isna(current_node) or str(current_node).strip() == "":
                current_node = dump_zone
            sim_time = float(row.get("sim_time_s", 0.0) or 0.0)
            cong = self.queue_index.congestion_factor(load_zone, sim_time)
            base_dur, _, _ = self.route_lookup.estimate_trip(current_node, load_zone, dump_zone, cong)
            actual_dur = float(row.get("total_trip_duration_s", np.nan))
            if pd.isna(actual_dur):
                continue
            residuals.append(actual_dur - base_dur)
            mine_keys.append(load_zone)
            truck_keys.append(int(row.get("truck_id")))

        if not residuals:
            return

        fit_df = pd.DataFrame(
            {"residual": residuals, "mine": mine_keys, "truck_id": truck_keys}
        )
        self.global_bias = float(fit_df["residual"].median())
        self.mine_bias = fit_df.groupby("mine")["residual"].median().to_dict()
        self.truck_bias = fit_df.groupby("truck_id")["residual"].median().to_dict()

    def predict_best_assignment(self, current_node, truck_id, load_zone, sim_time_s):
        dumps = self.route_lookup.dumps_by_mine.get(load_zone, self.route_lookup.all_dumps)
        best_dump = None
        best_pred = float("inf")
        for dump_zone in dumps:
            cong = self.queue_index.congestion_factor(load_zone, sim_time_s)
            base_dur, _, _ = self.route_lookup.estimate_trip(current_node, load_zone, dump_zone, cong)
            correction = (
                self.global_bias
                + self.mine_bias.get(load_zone, 0.0)
                + self.truck_bias.get(int(truck_id), 0.0)
            )
            pred = max(30.0, base_dur + correction)
            if pred < best_pred:
                best_dump = dump_zone
                best_pred = pred

        if best_dump is None:
            best_dump = self.route_lookup.best_dump_for_mine(load_zone)
            cong = self.queue_index.congestion_factor(load_zone, sim_time_s)
            best_pred, _, _ = self.route_lookup.estimate_trip(current_node, load_zone, best_dump, cong)
        return best_dump, best_pred


def build_dispatch_events(trips_df, events_df=None):
    """
    One row per dispatch moment: truck leaves dump and needs a new mine assignment.
    """
    trips = trips_df.sort_values("trip_id").copy()
    depart_nodes = defaultdict(list)
    if events_df is not None:
        departs = events_df[events_df["event_type"] == "DEPART_DUMP_ZONE"].copy()
        for _, row in departs.iterrows():
            depart_nodes[int(row["truck_id"])].append(
                (float(row["sim_time_s"]), row["node"])
            )
        for tid in depart_nodes:
            depart_nodes[tid].sort(key=lambda x: x[0])

    truck_last_dump = {}
    rows = []
    for _, row in trips.iterrows():
        tid = int(row["truck_id"])
        if pd.notna(row.get("depart_time_s")) and str(row.get("depart_time_s")).strip() not in ("", "nan"):
            t_dispatch = float(row["depart_time_s"])
        elif pd.notna(row.get("arrive_load_s")) and pd.notna(row.get("travel_to_mine_s")):
            if str(row.get("travel_to_mine_s")).strip() not in ("", "nan"):
                t_dispatch = float(row["arrive_load_s"]) - float(row["travel_to_mine_s"])
            else:
                t_dispatch = float(row["arrive_load_s"])
        else:
            t_dispatch = float(row["arrive_load_s"]) if pd.notna(row.get("arrive_load_s")) else 0.0

        current_node = truck_last_dump.get(tid)
        if depart_nodes[tid]:
            best_match = min(depart_nodes[tid], key=lambda x: abs(x[0] - t_dispatch))
            if abs(best_match[0] - t_dispatch) <= 5.0:
                current_node = best_match[1]
        if not current_node:
            current_node = row["dump_zone"]

        rows.append(
            {
                "sim_time_s": t_dispatch,
                "truck_id": tid,
                "current_node": current_node,
                "actual_load_zone": row["load_zone"],
                "actual_dump_zone": row["dump_zone"],
                "actual_duration": float(row["total_trip_duration_s"]),
                "actual_fuel": float(row["fuel_L"]),
                "actual_co2": float(row["co2_kg"]),
                "cargo_kg": float(row["cargo_kg"]),
                "trip_id": int(row["trip_id"]),
            }
        )
        truck_last_dump[tid] = row["dump_zone"]

    events = pd.DataFrame(rows).sort_values("sim_time_s").reset_index(drop=True)
    return events


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION STATE
# ══════════════════════════════════════════════════════════════════════════════

class MineState:
    def __init__(self, name, coal_capacity=10000.0):
        self.name = name
        self.coal_remaining = coal_capacity
        self.en_route = 0
        self.total_trips = 0
        self.last_assigned = 0.0


class TruckState:
    def __init__(self, truck_id):
        self.id = truck_id
        self.current_node = None
        self.trips_done = 0
        self.total_fuel = 0.0
        self.total_co2 = 0.0
        self.total_coal = 0.0


class SimState:
    def __init__(self, mines, trucks, road_dist_matrix):
        self.mines = {m: MineState(m) for m in mines}
        self.trucks = {t: TruckState(t) for t in trucks}
        self.dist = road_dist_matrix
        self.sim_time = 0.0
        self.trip_log = []
        self.route_lookup = None
        self.queue_index = None
        self.eta_estimator = None

    def assign(self, truck_id, mine_name):
        self.mines[mine_name].en_route += 1
        self.mines[mine_name].last_assigned = self.sim_time
        self.trucks[truck_id].current_node = mine_name

    def complete_trip(self, truck_id, mine_name, dump_zone, duration_s, fuel_L, co2_kg, cargo_kg):
        self.mines[mine_name].en_route = max(0, self.mines[mine_name].en_route - 1)
        self.mines[mine_name].coal_remaining = max(0, self.mines[mine_name].coal_remaining - cargo_kg)
        self.mines[mine_name].total_trips += 1
        self.trucks[truck_id].trips_done += 1
        self.trucks[truck_id].total_fuel += fuel_L
        self.trucks[truck_id].total_co2 += co2_kg
        self.trucks[truck_id].total_coal += cargo_kg
        self.trucks[truck_id].current_node = dump_zone
        self.trip_log.append(
            {
                "sim_time_s": self.sim_time,
                "truck_id": truck_id,
                "mine": mine_name,
                "dump_zone": dump_zone,
                "duration_s": duration_s,
                "fuel_L": fuel_L,
                "co2_kg": co2_kg,
                "cargo_kg": cargo_kg,
            }
        )


# ══════════════════════════════════════════════════════════════════════════════
#  POLICIES
# ══════════════════════════════════════════════════════════════════════════════

def travel_time(dist_matrix, frm, to, speed=DEFAULT_SPEED_MS):
    d = dist_matrix.get((frm, to), dist_matrix.get((to, frm), 530.0))
    return d / speed


def _active_mines(state):
    return [n for n, m in state.mines.items() if m.coal_remaining > 0]


def policy_greedy_nearest(truck_id, current_node, state, rr_counter=None):
    best, best_score = None, float("inf")
    rl, qi = state.route_lookup, state.queue_index
    for mine_name in _active_mines(state):
        mine = state.mines[mine_name]
        cong = qi.congestion_factor(mine_name, state.sim_time) if qi else 1.0
        if rl:
            _, cycle_dur, _, _ = rl.estimate_best_assignment(current_node, mine_name, cong)
        else:
            cycle_dur = 180.0
        tt = travel_time(state.dist, current_node, mine_name)
        wait = mine.en_route * LOAD_TIME_S
        score = tt + wait + cycle_dur
        if score < best_score:
            best_score, best = score, mine_name
    return best


def policy_eta_min(truck_id, current_node, state, rr_counter=None):
    best, best_score = None, float("inf")
    rl, qi = state.route_lookup, state.queue_index
    for mine_name in _active_mines(state):
        mine = state.mines[mine_name]
        cong = qi.congestion_factor(mine_name, state.sim_time) if qi else 1.0
        if rl:
            _, cycle_dur, _, _ = rl.estimate_best_assignment(current_node, mine_name, cong)
        else:
            cycle_dur = 180.0
        score = cycle_dur + mine.en_route * LOAD_TIME_S
        if score < best_score:
            best_score, best = score, mine_name
    return best


def policy_eta_aware(truck_id, current_node, state, rr_counter=None):
    best, best_score = None, float("inf")
    estimator = state.eta_estimator
    rl, qi = state.route_lookup, state.queue_index
    for mine_name in _active_mines(state):
        mine = state.mines[mine_name]
        if estimator is not None:
            _, cycle_dur = estimator.predict_best_assignment(
                current_node, truck_id, mine_name, state.sim_time
            )
        elif rl:
            cong = qi.congestion_factor(mine_name, state.sim_time) if qi else 1.0
            _, cycle_dur, _, _ = rl.estimate_best_assignment(current_node, mine_name, cong)
        else:
            cycle_dur = 180.0
        score = cycle_dur + mine.en_route * LOAD_TIME_S
        if score < best_score:
            best_score, best = score, mine_name
    return best


def policy_fifo(truck_id, current_node, state, rr_counter=None):
    best, oldest = None, float("inf")
    for mine_name in _active_mines(state):
        mine = state.mines[mine_name]
        if mine.last_assigned < oldest:
            oldest, best = mine.last_assigned, mine_name
    return best


def policy_load_balance(truck_id, current_node, state, rr_counter=None):
    best, fewest = None, float("inf")
    for mine_name in _active_mines(state):
        mine = state.mines[mine_name]
        score = mine.total_trips + mine.en_route
        if score < fewest:
            fewest, best = score, mine_name
    return best


def policy_round_robin(truck_id, current_node, state, rr_counter=None):
    active = sorted(_active_mines(state))
    if not active:
        return None
    idx = rr_counter[0] % len(active)
    rr_counter[0] += 1
    return active[idx]


POLICIES = {
    "GREEDY_NEAREST": policy_greedy_nearest,
    "ETA_MIN": policy_eta_min,
    "ETA_AWARE": policy_eta_aware,
    "FIFO": policy_fifo,
    "LOAD_BALANCE": policy_load_balance,
    "ROUND_ROBIN": policy_round_robin,
}


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT-DRIVEN REPLAY
# ══════════════════════════════════════════════════════════════════════════════

def replay_policy_event_driven(
    dispatch_events, policy_fn, policy_name, road_dist_matrix, route_lookup, queue_index, eta_estimator
):
    mines = list(route_lookup.all_mines)
    for extra in ["load_zone_24", "load_zone_25", "load_zone_26", "load_zone_29"]:
        if extra not in mines:
            mines.append(extra)
    trucks = sorted(dispatch_events["truck_id"].unique().tolist())

    state = SimState(mines, trucks, road_dist_matrix)
    state.route_lookup = route_lookup
    state.queue_index = queue_index
    state.eta_estimator = eta_estimator
    rr_counter = [0]
    dispatches = []
    sim_time = 0.0

    for _, ev in dispatch_events.iterrows():
        tid = int(ev["truck_id"])
        t_event = float(ev["sim_time_s"])
        sim_time = max(sim_time, t_event)
        state.sim_time = sim_time
        current_node = ev["current_node"] if pd.notna(ev["current_node"]) else "dump_zone_3"

        chosen = policy_fn(tid, current_node, state, rr_counter)
        if chosen is None:
            chosen = ev["actual_load_zone"]

        cong = queue_index.congestion_factor(chosen, sim_time)
        if policy_name == "ETA_AWARE" and eta_estimator is not None:
            dump_zone, est_dur = eta_estimator.predict_best_assignment(
                current_node, tid, chosen, sim_time
            )
            _, est_fuel, est_co2 = route_lookup.estimate_trip(
                current_node, chosen, dump_zone, cong
            )
        else:
            dump_zone, est_dur, est_fuel, est_co2 = route_lookup.estimate_best_assignment(
                current_node, chosen, cong
            )

        state.assign(tid, chosen)
        state.complete_trip(tid, chosen, dump_zone, est_dur, est_fuel, est_co2, ev["cargo_kg"])
        sim_time += est_dur
        state.sim_time = sim_time

        dispatches.append(
            {
                "policy": policy_name,
                "trip_id": int(ev["trip_id"]),
                "sim_time_s": round(t_event, 1),
                "truck_id": tid,
                "current_node": current_node,
                "chosen_mine": chosen,
                "chosen_dump": dump_zone,
                "actual_mine": ev["actual_load_zone"],
                "actual_dump": ev["actual_dump_zone"],
                "est_duration": round(est_dur, 1),
                "actual_duration": round(ev["actual_duration"], 1),
                "est_fuel": round(est_fuel, 4),
                "cargo_kg": ev["cargo_kg"],
                "queue_factor": round(cong, 3),
            }
        )

    return state, dispatches


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING + PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# Composite KPI weights (must sum to 1.0). Duration MAE excluded — it measures
# route-lookup accuracy, not fleet performance.
COMPOSITE_WEIGHTS = {
    "throughput_tph": 0.40,
    "total_fuel_L": 0.25,
    "avg_duration_s": 0.20,
    "mine_balance_std": 0.15,
}


def add_composite_scores(scores_df):
    """Add min-max normalized composite score and rank (higher = better)."""
    df = scores_df.copy()
    component_cols = []
    higher_better = {"throughput_tph"}

    df["composite_score"] = 0.0
    for col, weight in COMPOSITE_WEIGHTS.items():
        vals = df[col].astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            if col in higher_better:
                norm = 100.0 * (vals - vmin) / (vmax - vmin)
            else:
                norm = 100.0 * (vmax - vals) / (vmax - vmin)
        else:
            norm = pd.Series(100.0, index=df.index)

        comp_col = f"composite_{col}"
        df[comp_col] = norm.round(1)
        df["composite_score"] += weight * norm
        component_cols.append(comp_col)

    df["composite_score"] = df["composite_score"].round(1)
    df["composite_rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("composite_rank").reset_index(drop=True)
    return df, component_cols


def score_policy(state, dispatches, policy_name):
    trips = state.trip_log
    n_trips = len(trips)
    total_coal = sum(t["cargo_kg"] for t in trips)
    total_fuel = sum(t["fuel_L"] for t in trips)
    total_co2 = sum(t["co2_kg"] for t in trips)
    avg_dur = np.mean([t["duration_s"] for t in trips]) if trips else 0
    total_sim_time = state.sim_time
    throughput = n_trips / (total_sim_time / 3600) if total_sim_time > 0 else 0

    mine_trips = {m: state.mines[m].total_trips for m in state.mines}
    balance_std = np.std(list(mine_trips.values()))

    chosen_match = np.mean([d["chosen_mine"] == d["actual_mine"] for d in dispatches]) if dispatches else 0
    dur_mae = np.mean([abs(d["est_duration"] - d["actual_duration"]) for d in dispatches]) if dispatches else 0

    return {
        "policy": policy_name,
        "total_trips": n_trips,
        "throughput_tph": round(throughput, 2),
        "avg_duration_s": round(avg_dur, 1),
        "total_coal_t": round(total_coal / 1000, 2),
        "total_fuel_L": round(total_fuel, 3),
        "total_co2_kg": round(total_co2, 3),
        "mine_balance_std": round(balance_std, 2),
        "avg_queue_depth": round(np.mean([m.en_route for m in state.mines.values()]), 3),
        "fuel_per_trip_L": round(total_fuel / n_trips, 4) if n_trips else 0,
        "coal_per_trip_kg": round(total_coal / n_trips, 1) if n_trips else 0,
        "mine_match_rate": round(100 * chosen_match, 1),
        "duration_mae_s": round(dur_mae, 1),
        "sim_horizon_s": round(total_sim_time, 1),
    }


def plot_policy_comparison(scores_df):
    metrics = [
        ("throughput_tph", "Throughput (trips/hr)", "Higher is better", True),
        ("avg_duration_s", "Avg Trip Duration (s)", "Lower is better", False),
        ("mine_balance_std", "Mine Balance Std", "Lower is better", False),
        ("total_fuel_L", "Total Fuel (L)", "Lower is better", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Dispatch Policy Comparison (Event-Driven Replay)", fontsize=14, fontweight="bold", color=C_BLUE)
    axes = axes.flatten()

    for ax, (col, label, note, higher_better) in zip(axes, metrics):
        vals = scores_df[col].values
        names = scores_df["policy"].values
        colours = [POLICY_COLORS.get(n, C_GRAY) for n in names]
        best_idx = np.argmax(vals) if higher_better else np.argmin(vals)
        edge_cols = ["gold" if i == best_idx else "white" for i in range(len(names))]

        bars = ax.bar(range(len(names)), vals, color=colours, alpha=0.85, edgecolor=edge_cols, linewidth=2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label}\n({note})", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * max(bar.get_height(), 1),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "dispatch_policy_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_mine_utilization(all_states):
    mines = sorted(list(list(all_states.values())[0].mines.keys()))
    n_pol = len(all_states)
    x = np.arange(len(mines))
    w = 0.8 / n_pol

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (pol_name, state) in enumerate(all_states.items()):
        trips = [state.mines[m].total_trips for m in mines]
        offset = (i - n_pol / 2 + 0.5) * w
        ax.bar(
            x + offset,
            trips,
            w,
            label=pol_name,
            color=POLICY_COLORS.get(pol_name, C_GRAY),
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("load_zone_", "Mine ") for m in mines], fontsize=10)
    ax.set_ylabel("Trips Completed")
    ax.set_title("Mine Utilization by Policy", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "mine_utilization.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_cumulative_coal(all_dispatches):
    fig, ax = plt.subplots(figsize=(10, 5))
    for pol_name, dispatches in all_dispatches.items():
        cumcoal = np.cumsum([d["cargo_kg"] for d in dispatches]) / 1000
        ax.plot(
            range(1, len(cumcoal) + 1),
            cumcoal,
            label=pol_name,
            color=POLICY_COLORS.get(pol_name, C_GRAY),
            lw=2,
            marker="o",
            ms=3,
            alpha=0.85,
        )
    ax.set_xlabel("Trip number")
    ax.set_ylabel("Cumulative Coal Moved (tonnes)")
    ax.set_title("Cumulative Coal Throughput by Policy", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "cumulative_coal.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_duration_distribution(all_dispatches):
    fig, ax = plt.subplots(figsize=(10, 5))
    for pol_name, dispatches in all_dispatches.items():
        durations = [d["est_duration"] for d in dispatches]
        ax.hist(
            durations,
            bins=15,
            alpha=0.5,
            label=pol_name,
            color=POLICY_COLORS.get(pol_name, C_GRAY),
            edgecolor="white",
        )
    ax.set_xlabel("Estimated Trip Duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Trip Duration Distribution by Policy", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "duration_distribution_by_policy.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def save_assignment_interpretability(all_dispatches):
    """
    Save mine/dump choice distribution and average assignment signals per policy.
    """
    rows = []
    for pol_name, dispatches in all_dispatches.items():
        if not dispatches:
            continue
        df = pd.DataFrame(dispatches)
        total = len(df)
        mine_counts = df["chosen_mine"].value_counts()
        dump_counts = df["chosen_dump"].value_counts()
        top_mine = mine_counts.index[0] if not mine_counts.empty else "NA"
        top_dump = dump_counts.index[0] if not dump_counts.empty else "NA"
        top_mine_share = 100.0 * float(mine_counts.iloc[0]) / total if not mine_counts.empty else 0.0
        top_dump_share = 100.0 * float(dump_counts.iloc[0]) / total if not dump_counts.empty else 0.0

        for mine_name, count in mine_counts.items():
            rows.append(
                {
                    "policy": pol_name,
                    "summary_type": "mine_share",
                    "choice": mine_name,
                    "count": int(count),
                    "share_pct": round(100.0 * float(count) / total, 2),
                    "avg_est_duration_s": round(float(df["est_duration"].mean()), 2),
                    "avg_queue_factor": round(float(df["queue_factor"].mean()), 3),
                    "mine_match_rate_pct": round(
                        100.0 * float((df["chosen_mine"] == df["actual_mine"]).mean()), 2
                    ),
                    "top_mine": top_mine,
                    "top_mine_share_pct": round(top_mine_share, 2),
                    "top_dump": top_dump,
                    "top_dump_share_pct": round(top_dump_share, 2),
                }
            )

        for dump_name, count in dump_counts.items():
            rows.append(
                {
                    "policy": pol_name,
                    "summary_type": "dump_share",
                    "choice": dump_name,
                    "count": int(count),
                    "share_pct": round(100.0 * float(count) / total, 2),
                    "avg_est_duration_s": round(float(df["est_duration"].mean()), 2),
                    "avg_queue_factor": round(float(df["queue_factor"].mean()), 3),
                    "mine_match_rate_pct": round(
                        100.0 * float((df["chosen_mine"] == df["actual_mine"]).mean()), 2
                    ),
                    "top_mine": top_mine,
                    "top_mine_share_pct": round(top_mine_share, 2),
                    "top_dump": top_dump,
                    "top_dump_share_pct": round(top_dump_share, 2),
                }
            )

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "dispatch_assignment_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    if not summary_df.empty:
        overview = (
            summary_df[summary_df["summary_type"] == "mine_share"]
            .sort_values(["policy", "share_pct"], ascending=[True, False])
            .groupby("policy", as_index=False)
            .first()[["policy", "top_mine", "top_mine_share_pct", "top_dump", "top_dump_share_pct"]]
        )
        print("\nTop assignment pattern by policy:")
        for _, row in overview.iterrows():
            print(
                f"  {row['policy']:<16} top_mine={row['top_mine']}"
                f" ({row['top_mine_share_pct']}%)  top_dump={row['top_dump']}"
                f" ({row['top_dump_share_pct']}%)"
            )


def save_policy_disagreement_matrix(all_dispatches):
    """
    Save pairwise mine-choice disagreement across policies.
    """
    decisions = {}
    for pol_name, dispatches in all_dispatches.items():
        if not dispatches:
            continue
        df = pd.DataFrame(dispatches)[["trip_id", "chosen_mine"]].copy()
        df = df.sort_values("trip_id").drop_duplicates("trip_id", keep="last")
        decisions[pol_name] = df.set_index("trip_id")["chosen_mine"]

    policies = sorted(decisions.keys())
    if not policies:
        return

    matrix = pd.DataFrame(index=policies, columns=policies, dtype=float)
    for p1 in policies:
        s1 = decisions[p1]
        for p2 in policies:
            s2 = decisions[p2]
            common_ids = s1.index.intersection(s2.index)
            if len(common_ids) == 0:
                disagree_pct = np.nan
            else:
                disagree_pct = 100.0 * float((s1.loc[common_ids] != s2.loc[common_ids]).mean())
            matrix.loc[p1, p2] = round(disagree_pct, 2) if pd.notna(disagree_pct) else np.nan

    out_path = os.path.join(OUT_DIR, "dispatch_policy_disagreement_matrix.csv")
    matrix.to_csv(out_path, index=True)
    print(f"  Saved: {out_path}")

    if "ETA_AWARE" in matrix.index:
        eta_row = matrix.loc["ETA_AWARE"].drop(labels=["ETA_AWARE"], errors="ignore")
        eta_row = eta_row.sort_values(ascending=False)
        print("\nETA_AWARE mine-choice disagreement vs others:")
        for pol_name, val in eta_row.items():
            print(f"  ETA_AWARE vs {pol_name:<14} {val:.2f}%")


def plot_policy_disagreement_heatmap():
    """
    Plot pairwise mine-choice disagreement matrix saved by policy replay.
    """
    matrix_path = os.path.join(OUT_DIR, "dispatch_policy_disagreement_matrix.csv")
    if not os.path.exists(matrix_path):
        return

    mat = pd.read_csv(matrix_path, index_col=0)
    if mat.empty:
        return

    vals = mat.values.astype(float)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(vals, cmap="YlOrRd", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_title("Policy Mine-Choice Disagreement (%)", fontweight="bold")

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            color = "white" if vals[i, j] > 55 else "black"
            ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Disagreement (%)", rotation=90)
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "policy_disagreement_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== Dispatch Policy Comparator (Event-Driven) ===\n")

    graph_path = os.path.join(OUT_DIR, "trips_graph.csv")
    plain_path = os.path.join(OUT_DIR, "trips.csv")
    trips_path = graph_path if os.path.exists(graph_path) else plain_path
    trips_df = pd.read_csv(trips_path).dropna(subset=["total_trip_duration_s"])
    print(f"Loaded {len(trips_df)} trips from {trips_path}")

    events_path = load_events_path()
    events_df = pd.read_csv(events_path) if events_path else None
    if events_path:
        print(f"Loaded events from {events_path} ({len(events_df)} rows)")

    queue_path = os.path.join(OUT_DIR, "queue_events.csv")
    queue_df = pd.read_csv(queue_path) if os.path.exists(queue_path) else pd.DataFrame()
    if not queue_df.empty:
        print(f"Loaded {len(queue_df)} queue events")

    dispatch_events = build_dispatch_events(trips_df, events_df)
    print(f"Built {len(dispatch_events)} dispatch events")

    road_dist = build_road_distance_matrix(trips_df)
    route_lookup = RouteLookup(trips_df, road_dist)
    queue_index = QueueIndex(queue_df) if not queue_df.empty else QueueIndex(
        pd.DataFrame(columns=["sim_time_s", "zone", "trucks_queued", "event_type"])
    )
    eta_estimator = ETAAwareEstimator(trips_df, dispatch_events, route_lookup, queue_index)

    all_states = {}
    all_dispatches = {}
    all_scores = []

    print("\nRunning policies (event-driven replay)...")
    for pol_name, pol_fn in POLICIES.items():
        state, dispatches = replay_policy_event_driven(
            dispatch_events, pol_fn, pol_name, road_dist, route_lookup, queue_index, eta_estimator
        )
        score = score_policy(state, dispatches, pol_name)
        all_states[pol_name] = state
        all_dispatches[pol_name] = dispatches
        all_scores.append(score)
        print(
            f"  {pol_name:<16}  trips={score['total_trips']}  "
            f"tph={score['throughput_tph']:.2f}  "
            f"avg_dur={score['avg_duration_s']:.0f}s  "
            f"balance_std={score['mine_balance_std']:.2f}  "
            f"dur_mae={score['duration_mae_s']:.0f}s"
        )

    scores_df = pd.DataFrame(all_scores)
    scores_df, composite_cols = add_composite_scores(scores_df)
    for _, row in scores_df.iterrows():
        print(
            f"    -> {row['policy']:<16}  composite={row['composite_score']:.1f}  "
            f"rank=#{int(row['composite_rank'])}"
        )

    scores_path = os.path.join(OUT_DIR, "dispatch_policy_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"\n  Saved: {scores_path}")

    weight_str = ", ".join(f"{k}={int(v * 100)}%" for k, v in COMPOSITE_WEIGHTS.items())
    print(f"  Composite weights: {weight_str}")

    all_disp_df = pd.DataFrame([d for disps in all_dispatches.values() for d in disps])
    disp_path = os.path.join(OUT_DIR, "dispatch_decisions.csv")
    all_disp_df.to_csv(disp_path, index=False)
    print(f"  Saved: {disp_path}")
    save_assignment_interpretability(all_dispatches)
    save_policy_disagreement_matrix(all_dispatches)

    print("\nGenerating plots...")
    plot_policy_comparison(scores_df)
    plot_mine_utilization(all_states)
    plot_cumulative_coal(all_dispatches)
    plot_duration_distribution(all_dispatches)
    plot_policy_disagreement_heatmap()

    print(f"\n{'='*62}")
    print(f"  {'Metric':<28} {'Best Policy':<18} {'Value'}")
    print(f"  {'-'*28} {'-'*18} {'-'*8}")
    metrics_best = [
        ("throughput_tph", True, "Throughput (trips/hr)"),
        ("avg_duration_s", False, "Avg trip duration"),
        ("mine_balance_std", False, "Mine balance (std)"),
        ("total_fuel_L", False, "Total fuel (L)"),
        ("duration_mae_s", False, "Duration MAE (s)"),
    ]
    for col, higher_better, label in metrics_best:
        idx = scores_df[col].idxmax() if higher_better else scores_df[col].idxmin()
        row = scores_df.iloc[idx]
        print(f"  {label:<28} {row['policy']:<18} {row[col]}")
    best_overall = scores_df.iloc[0]
    print(f"  {'Composite score (weighted)':<28} {best_overall['policy']:<18} {best_overall['composite_score']}")
    print(f"{'='*62}")
    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print(f"Plots saved to:       {PLOT_DIR}/")


if __name__ == "__main__":
    main()
