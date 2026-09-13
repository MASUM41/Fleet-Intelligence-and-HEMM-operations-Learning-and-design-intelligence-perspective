"""
merge_logs.py - Combine multiple simulation log sessions into one dataset.

Pairs events_*.csv with telemetry_*.csv by timestamp suffix, offsets sim_time
so sessions do not overlap, and writes merged CSVs for kpi_extractor.

Run from Simulation/:
    python merge_logs.py
    python kpi_extractor.py --merged
"""

import csv
import glob
import os
import re

LOG_DIR = "logs"
GAP_BETWEEN_RUNS_S = 60.0


def session_key(path):
    match = re.search(r"_(\d{8}_\d{6})\.csv$", os.path.basename(path))
    return match.group(1) if match else path


def pair_log_files():
    events = {
        session_key(p): p
        for p in glob.glob(os.path.join(LOG_DIR, "events_*.csv"))
        if "merged_" not in os.path.basename(p)
    }
    telem = {
        session_key(p): p
        for p in glob.glob(os.path.join(LOG_DIR, "telemetry_*.csv"))
        if "merged_" not in os.path.basename(p)
    }
    keys = sorted(set(events) & set(telem))
    return [(k, events[k], telem[k]) for k in keys]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def merge_sessions(pairs):
    merged_events = []
    merged_telemetry = []
    event_fields = None
    telem_fields = None
    offset = 0.0

    for run_idx, (key, events_path, telem_path) in enumerate(pairs, start=1):
        e_fields, e_rows = read_csv(events_path)
        t_fields, t_rows = read_csv(telem_path)
        event_fields = event_fields or e_fields
        telem_fields = telem_fields or t_fields

        max_time = 0.0
        for row in e_rows + t_rows:
            t = float(row["sim_time_s"])
            max_time = max(max_time, t)

        for row in e_rows:
            out = dict(row)
            out["sim_time_s"] = round(float(row["sim_time_s"]) + offset, 3)
            out["run_id"] = run_idx
            out["session_key"] = key
            merged_events.append(out)

        for row in t_rows:
            out = dict(row)
            out["sim_time_s"] = round(float(row["sim_time_s"]) + offset, 3)
            out["run_id"] = run_idx
            out["session_key"] = key
            merged_telemetry.append(out)

        print(
            f"  Run {run_idx} ({key}): "
            f"{len(e_rows)} events, {len(t_rows)} telemetry rows, "
            f"duration={max_time:.1f}s"
        )
        offset += max_time + GAP_BETWEEN_RUNS_S

    return event_fields, telem_fields, merged_events, merged_telemetry


def write_csv(path, fieldnames, rows):
    extra = ["run_id", "session_key"]
    out_fields = list(fieldnames)
    for col in extra:
        if col not in out_fields:
            out_fields.append(col)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")


def main():
    print("=== Merge Simulation Logs ===\n")
    pairs = pair_log_files()
    if not pairs:
        raise FileNotFoundError(f"No matching events/telemetry pairs in {LOG_DIR}/")

    print(f"Found {len(pairs)} session(s):\n")
    e_fields, t_fields, events, telemetry = merge_sessions(pairs)

    events_path = os.path.join(LOG_DIR, "merged_events.csv")
    telem_path = os.path.join(LOG_DIR, "merged_telemetry.csv")
    print()
    write_csv(events_path, e_fields, events)
    write_csv(telem_path, t_fields, telemetry)

    unload_count = sum(1 for r in events if r.get("event_type") == "UNLOAD_COMPLETE")
    print(f"\nMerged sessions : {len(pairs)}")
    print(f"Completed trips   : {unload_count}  (UNLOAD_COMPLETE events)")
    print("Next: python kpi_extractor.py --merged")


if __name__ == "__main__":
    main()
