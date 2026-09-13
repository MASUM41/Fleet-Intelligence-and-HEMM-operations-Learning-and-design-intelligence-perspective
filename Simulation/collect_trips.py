"""
collect_trips.py - Run multiple headless simulations and rebuild the trip dataset.

Example (target ~150+ trips):
    python generate_assets.py
    python collect_trips.py --runs 4 --duration 7200 --speed 5
    python graph_features.py

Run from Simulation/:
    python collect_trips.py --help
"""

import argparse
import subprocess
import sys

from main import run_simulation


def count_unload_events():
    import csv
    import glob
    import os

    total = 0
    for path in glob.glob(os.path.join("logs", "events_*.csv")):
        if "merged_" in os.path.basename(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            total += sum(1 for row in csv.DictReader(f) if row.get("event_type") == "UNLOAD_COMPLETE")
    return total


def run_pipeline():
    subprocess.run([sys.executable, "merge_logs.py"], check=True)
    subprocess.run([sys.executable, "kpi_extractor.py", "--merged"], check=True)


def main():
    parser = argparse.ArgumentParser(description="Collect fleet trip data via headless simulation")
    parser.add_argument("--runs", type=int, default=3, help="Number of simulation sessions")
    parser.add_argument("--duration", type=float, default=7200.0, help="Sim seconds per run (default 2 hours)")
    parser.add_argument("--speed", type=float, default=5.0, help="Simulation speed multiplier")
    parser.add_argument("--target-trips", type=int, default=150, help="Stop early once this many trips exist")
    parser.add_argument("--skip-sim", action="store_true", help="Only merge existing logs and extract KPIs")
    args = parser.parse_args()

    if not args.skip_sim:
        before = count_unload_events()
        print(f"=== Trip Collection ===")
        print(f"Existing completed trips in logs: {before}")
        print(f"Target: {args.target_trips} trips\n")

        for run_idx in range(1, args.runs + 1):
            current = count_unload_events()
            if current >= args.target_trips:
                print(f"Target reached ({current} trips). Stopping early.")
                break

            print(f"--- Simulation run {run_idx}/{args.runs} ---")
            run_simulation(
                headless=True,
                duration_s=args.duration,
                sim_speed_init=args.speed,
            )
            after = count_unload_events()
            print(f"Trips so far: {after} (+{after - current})\n")

    print("Merging logs and extracting KPIs...")
    run_pipeline()

    import pandas as pd

    trips = pd.read_csv("analysis/trips.csv")
    print(f"\nDone. trips.csv now has {len(trips)} rows.")
    if len(trips) < args.target_trips:
        print(
            f"Still below target ({args.target_trips}). "
            f"Try more runs or longer duration, e.g.\n"
            f"  python collect_trips.py --runs 6 --duration 10800 --speed 5"
        )


if __name__ == "__main__":
    main()
