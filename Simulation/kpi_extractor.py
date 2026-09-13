"""
kpi_extractor.py  –  Parse sim_logger CSVs into clean analysis-ready tables.

Run from Simulation/ folder:
    python kpi_extractor.py

Reads:  logs/events_*.csv   (latest file automatically picked)
        logs/telemetry_*.csv (latest file automatically picked)

Writes:
    analysis/trips.csv          – one row per completed trip
    analysis/fleet_kpis.csv     – one row per minute (fleet-level)
    analysis/queue_events.csv   – one row per queue wait instance
    analysis/idle_periods.csv   – when trucks were stationary (speed < 0.5 km/h)
"""

import argparse
import os
import glob
import csv
import math
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
LOG_DIR      = "logs"
OUT_DIR      = "analysis"
IDLE_SPEED   = 0.5      # km/h below which truck is considered idle
BIN_SECONDS  = 60.0     # fleet KPI aggregation window (1 minute)
# ─────────────────────────────────────────────────────────────────────────────


def latest_file(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching: {pattern}")
    return files[-1]


def parse_extra(extra_str):
    """Parse 'key=val|key=val' extra field into a dict."""
    result = {}
    if not extra_str:
        return result
    for part in extra_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  1.  TRIPS TABLE
#      One row per UNLOAD_COMPLETE event (= one full mine→dump cycle)
# ══════════════════════════════════════════════════════════════════════════════

def build_trips(events_path):
    """
    trips.csv columns:
        trip_id, truck_id, trip_number,
        load_zone, dump_zone,
        depart_time_s, arrive_load_s, load_complete_s,
        depart_load_s, arrive_dump_s, unload_complete_s,
        travel_to_mine_s, load_wait_s, travel_to_dump_s, unload_wait_s,
        total_trip_duration_s,
        cargo_kg, fuel_L, co2_kg,
        avg_speed_kmh (estimated from travel time + euclidean not available → left blank)
    """
    rows = []
    # per-truck state machine (keyed by run + truck when merging sessions)
    truck_state = defaultdict(dict)

    global_trip_id = 0

    with open(events_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t       = float(row["sim_time_s"])
            etype   = row["event_type"]
            tid     = int(row["truck_id"])
            node    = row["node"]
            cargo   = float(row["cargo_kg"])
            extra   = parse_extra(row.get("extra", ""))
            run_id  = row.get("run_id") or "1"
            session = row.get("session_key", "")
            state_key = (run_id, tid)
            ts      = truck_state[state_key]

            if etype == "DEPART_DUMP_ZONE":
                # New trip starts when truck leaves dump zone heading to mine
                ts["trip_start_s"]   = t
                ts["load_zone"]      = ""
                ts["dump_zone"]      = ""

            elif etype == "ARRIVE_LOAD_ZONE":
                ts["arrive_load_s"]  = t
                ts["load_zone"]      = node

            elif etype == "LOAD_COMPLETE":
                ts["load_complete_s"] = t
                ts["cargo_kg"]        = cargo
                # dump_zone is in 'node' field due to a logger quirk — skip, use ARRIVE_DUMP

            elif etype == "DEPART_LOAD_ZONE":
                ts["depart_load_s"]  = t

            elif etype == "ARRIVE_DUMP_ZONE":
                ts["arrive_dump_s"]  = t
                ts["dump_zone"]      = node

            elif etype == "UNLOAD_COMPLETE":
                ts["unload_complete_s"] = t
                trip_num = int(extra.get("trip_id", 0))
                dur      = float(extra.get("trip_duration_s", 0))
                fuel     = float(extra.get("trip_fuel_L", 0))
                co2      = float(extra.get("trip_co2_kg", 0))

                # Compute sub-durations safely
                def tdiff(a, b):
                    va = ts.get(a, 0)
                    vb = ts.get(b, 0)
                    return round(vb - va, 2) if va and vb else ""

                travel_to_mine = tdiff("trip_start_s",   "arrive_load_s")
                load_wait      = tdiff("arrive_load_s",  "load_complete_s")
                travel_to_dump = tdiff("depart_load_s",  "arrive_dump_s")
                unload_wait    = tdiff("arrive_dump_s",  "unload_complete_s")

                global_trip_id += 1
                rows.append({
                    "trip_id"              : global_trip_id,
                    "run_id"               : run_id,
                    "session_key"          : session,
                    "truck_id"             : tid,
                    "trip_number"          : trip_num,
                    "load_zone"            : ts.get("load_zone", ""),
                    "dump_zone"            : ts.get("dump_zone", node),
                    "depart_time_s"        : ts.get("trip_start_s", ""),
                    "arrive_load_s"        : ts.get("arrive_load_s", ""),
                    "load_complete_s"      : ts.get("load_complete_s", ""),
                    "depart_load_s"        : ts.get("depart_load_s", ""),
                    "arrive_dump_s"        : ts.get("arrive_dump_s", ""),
                    "unload_complete_s"    : t,
                    "travel_to_mine_s"     : travel_to_mine,
                    "load_wait_s"          : load_wait,
                    "travel_to_dump_s"     : travel_to_dump,
                    "unload_wait_s"        : unload_wait,
                    "total_trip_duration_s": round(dur, 2),
                    "cargo_kg"             : ts.get("cargo_kg", cargo),
                    "fuel_L"               : round(fuel, 4),
                    "co2_kg"               : round(co2, 6),
                })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  2.  FLEET KPIs TABLE  (1-minute bins)
# ══════════════════════════════════════════════════════════════════════════════

def build_fleet_kpis(telemetry_path, trips_rows):
    """
    fleet_kpis.csv columns:
        bin_start_s, bin_end_s,
        trips_completed,
        coal_moved_kg,
        avg_speed_kmh,          (fleet average over all trucks in bin)
        pct_moving,             (% of truck-seconds where speed > IDLE_SPEED)
        pct_loading,            (% of truck-seconds in LOADING state)
        pct_unloading,
        pct_travelling,
        trucks_active,
        total_fuel_L,
        total_co2_kg,
        throughput_trips_per_hr
    """
    # Read telemetry into bins
    bins = defaultdict(lambda: {
        "speed_sum": 0.0, "speed_count": 0,
        "moving": 0, "total": 0,
        "loading": 0, "unloading": 0, "travelling": 0,
        "trucks": set(),
        "fuel": 0.0, "co2": 0.0,
    })

    with open(telemetry_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t     = float(row["sim_time_s"])
            b     = int(t // BIN_SECONDS) * BIN_SECONDS
            speed = float(row["speed_kmh"])
            state = row["op_state"]
            tid   = int(row["truck_id"])
            fuel  = float(row["fuel_L_step"])
            co2   = float(row["co2_kg_step"])

            d = bins[b]
            d["speed_sum"]   += speed
            d["speed_count"] += 1
            d["total"]       += 1
            d["trucks"].add(tid)
            d["fuel"]        += fuel
            d["co2"]         += co2

            if speed > IDLE_SPEED:
                d["moving"] += 1
            if state == "LOADING":
                d["loading"] += 1
            elif state == "UNLOADING":
                d["unloading"] += 1
            elif state in ("GOING_TO_ENDPOINT", "RETURNING_TO_START"):
                d["travelling"] += 1

    # Count trips per bin
    trips_per_bin = defaultdict(int)
    coal_per_bin  = defaultdict(float)
    for tr in trips_rows:
        if tr["unload_complete_s"]:
            b = int(float(tr["unload_complete_s"]) // BIN_SECONDS) * BIN_SECONDS
            trips_per_bin[b] += 1
            coal_per_bin[b]  += float(tr["cargo_kg"]) if tr["cargo_kg"] else 0

    all_bins = sorted(set(list(bins.keys()) + list(trips_per_bin.keys())))

    rows = []
    for b in all_bins:
        d     = bins[b]
        total = d["total"] or 1
        avg_spd = round(d["speed_sum"] / d["speed_count"], 2) if d["speed_count"] else 0
        trips   = trips_per_bin[b]
        tph     = round(trips * (3600 / BIN_SECONDS), 2) if trips else 0

        rows.append({
            "bin_start_s"           : b,
            "bin_end_s"             : b + BIN_SECONDS,
            "trips_completed"       : trips,
            "coal_moved_kg"         : round(coal_per_bin[b], 1),
            "avg_speed_kmh"         : avg_spd,
            "pct_moving"            : round(100 * d["moving"]     / total, 1),
            "pct_loading"           : round(100 * d["loading"]    / total, 1),
            "pct_unloading"         : round(100 * d["unloading"]  / total, 1),
            "pct_travelling"        : round(100 * d["travelling"] / total, 1),
            "trucks_active"         : len(d["trucks"]),
            "total_fuel_L"          : round(d["fuel"], 4),
            "total_co2_kg"          : round(d["co2"], 6),
            "throughput_trips_per_hr": tph,
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  3.  QUEUE EVENTS TABLE
#      When 2+ trucks converged on same load/dump zone simultaneously
# ══════════════════════════════════════════════════════════════════════════════

def build_queue_events(events_path):
    """
    queue_events.csv columns:
        sim_time_s, zone, trucks_queued, event_type
    Detected when ARRIVE_LOAD_ZONE or ARRIVE_DUMP_ZONE fires while
    another truck is already at that zone (en_route count > 1).
    """
    rows = []
    zone_occupancy = defaultdict(list)   # zone -> [truck_ids currently there]

    with open(events_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t     = float(row["sim_time_s"])
            etype = row["event_type"]
            tid   = int(row["truck_id"])
            node  = row["node"]

            if etype in ("ARRIVE_LOAD_ZONE", "ARRIVE_DUMP_ZONE"):
                zone_occupancy[node].append(tid)
                if len(zone_occupancy[node]) > 1:
                    rows.append({
                        "sim_time_s"   : t,
                        "zone"         : node,
                        "trucks_queued": len(zone_occupancy[node]),
                        "truck_ids"    : "+".join(str(x) for x in zone_occupancy[node]),
                        "event_type"   : etype,
                    })

            elif etype in ("DEPART_LOAD_ZONE", "DEPART_DUMP_ZONE",
                           "LOAD_COMPLETE",    "UNLOAD_COMPLETE"):
                if tid in zone_occupancy[node]:
                    zone_occupancy[node].remove(tid)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  4.  IDLE PERIODS TABLE
#      Stretches where a truck's speed stayed below IDLE_SPEED
# ══════════════════════════════════════════════════════════════════════════════

def build_idle_periods(telemetry_path):
    """
    idle_periods.csv columns:
        truck_id, idle_start_s, idle_end_s, duration_s, op_state, x_m, y_m
    """
    rows = []
    truck_idle = {}   # tid -> {start, state, x, y}

    with open(telemetry_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t     = float(row["sim_time_s"])
            tid   = int(row["truck_id"])
            speed = float(row["speed_kmh"])
            state = row["op_state"]
            x     = float(row["x_m"])
            y     = float(row["y_m"])

            if speed < IDLE_SPEED:
                if tid not in truck_idle:
                    truck_idle[tid] = {"start": t, "state": state, "x": x, "y": y}
            else:
                if tid in truck_idle:
                    idle = truck_idle.pop(tid)
                    dur  = round(t - idle["start"], 2)
                    if dur >= 2.0:   # ignore sub-2s micro-stops
                        rows.append({
                            "truck_id"    : tid,
                            "idle_start_s": idle["start"],
                            "idle_end_s"  : t,
                            "duration_s"  : dur,
                            "op_state"    : idle["state"],
                            "x_m"         : round(idle["x"], 2),
                            "y_m"         : round(idle["y"], 2),
                        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def resolve_log_paths(use_merged=False):
    if use_merged:
        events_path = os.path.join(LOG_DIR, "merged_events.csv")
        telemetry_path = os.path.join(LOG_DIR, "merged_telemetry.csv")
        if not os.path.exists(events_path) or not os.path.exists(telemetry_path):
            raise FileNotFoundError(
                "Merged logs not found. Run: python merge_logs.py"
            )
        return events_path, telemetry_path

    return (
        latest_file(os.path.join(LOG_DIR, "events_*.csv")),
        latest_file(os.path.join(LOG_DIR, "telemetry_*.csv")),
    )


def main():
    parser = argparse.ArgumentParser(description="Extract KPI tables from sim logs")
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Use logs/merged_events.csv and logs/merged_telemetry.csv",
    )
    args = parser.parse_args()

    print("=== HEMM KPI Extractor ===\n")

    events_path, telemetry_path = resolve_log_paths(use_merged=args.merged)

    print(f"Events file   : {events_path}")
    print(f"Telemetry file: {telemetry_path}\n")

    # Build tables
    print("Building trips table...")
    trips = build_trips(events_path)

    print("Building fleet KPIs table...")
    fleet = build_fleet_kpis(telemetry_path, trips)

    print("Building queue events table...")
    queue = build_queue_events(events_path)

    print("Building idle periods table...")
    idle  = build_idle_periods(telemetry_path)

    # Write outputs
    print("\nWriting outputs...")

    write_csv(
        os.path.join(OUT_DIR, "trips.csv"), trips,
        fieldnames=[
            "trip_id", "run_id", "session_key", "truck_id", "trip_number",
            "load_zone", "dump_zone",
            "depart_time_s", "arrive_load_s", "load_complete_s",
            "depart_load_s", "arrive_dump_s", "unload_complete_s",
            "travel_to_mine_s", "load_wait_s",
            "travel_to_dump_s", "unload_wait_s",
            "total_trip_duration_s",
            "cargo_kg", "fuel_L", "co2_kg",
        ]
    )

    write_csv(
        os.path.join(OUT_DIR, "fleet_kpis.csv"), fleet,
        fieldnames=[
            "bin_start_s", "bin_end_s",
            "trips_completed", "coal_moved_kg",
            "avg_speed_kmh",
            "pct_moving", "pct_loading", "pct_unloading", "pct_travelling",
            "trucks_active",
            "total_fuel_L", "total_co2_kg",
            "throughput_trips_per_hr",
        ]
    )

    write_csv(
        os.path.join(OUT_DIR, "queue_events.csv"), queue,
        fieldnames=["sim_time_s", "zone", "trucks_queued", "truck_ids", "event_type"]
    )

    write_csv(
        os.path.join(OUT_DIR, "idle_periods.csv"), idle,
        fieldnames=["truck_id", "idle_start_s", "idle_end_s", "duration_s", "op_state", "x_m", "y_m"]
    )

    # Quick summary
    print("\n--- Summary -------------------------------")
    print(f"  Total trips completed : {len(trips)}")
    if trips:
        durations = [t["total_trip_duration_s"] for t in trips if t["total_trip_duration_s"]]
        fuels     = [t["fuel_L"] for t in trips if t["fuel_L"]]
        print(f"  Avg trip duration     : {sum(durations)/len(durations):.1f} s")
        print(f"  Avg fuel per trip     : {sum(fuels)/len(fuels):.3f} L")
        total_coal = sum(float(t["cargo_kg"]) for t in trips if t["cargo_kg"])
        print(f"  Total coal moved      : {total_coal/1000:.2f} tonnes")
    print(f"  Queue events detected : {len(queue)}")
    print(f"  Idle periods logged   : {len(idle)}")
    print(f"  Fleet KPI bins        : {len(fleet)}")
    print("-------------------------------------------")
    print(f"\nAll outputs written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
