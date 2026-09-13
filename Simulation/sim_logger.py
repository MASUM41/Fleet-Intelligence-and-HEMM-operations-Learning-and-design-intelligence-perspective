"""
sim_logger.py  –  Drop-in logging layer for the Autonomous-Navigation simulator.

Two log streams written to logs/ directory:
  telemetry.csv   – per-truck snapshot sampled at TELEMETRY_HZ  (default 1 Hz)
  events.csv      – state-transition events (arrive, load, unload, dispatch, …)

Usage (in Simulation/main.py):
-----------------------------------------------------------------------
    from sim_logger import SimLogger          # <-- ADD THIS

    logger = SimLogger()                      # <-- ADD THIS (before main loop)

    # Inside the main loop, inside `if sim_dt > 0:` block:
    logger.tick(sim_time, sim_dt, cars, dispatcher)   # <-- ADD THIS

    # On clean exit (after `while running` loop ends):
    logger.close()                            # <-- ADD THIS
-----------------------------------------------------------------------

sim_time must be a float you maintain yourself:
    sim_time = 0.0          # before loop
    sim_time += sim_dt      # inside `if sim_dt > 0` block
"""

import csv
import os
import math
from datetime import datetime

# ── tunables ──────────────────────────────────────────────────────────────────
TELEMETRY_HZ        = 1.0          # how often to sample per-truck state (Hz)
LOG_DIR             = "logs"       # directory to write CSV files into
# ─────────────────────────────────────────────────────────────────────────────


# ── helpers ───────────────────────────────────────────────────────────────────

def _speed_kmh(speed_ms: float) -> float:
    return round(speed_ms * 3.6, 3)

def _cargo_kg(car) -> float:
    """Returns cargo mass in kg (0 when empty)."""
    from config import MASS_KG
    return max(0.0, round(car.current_mass_kg - MASS_KG, 2))

def _fuel_proxy(car, dt: float) -> float:
    """
    Rough fuel-consumption proxy (litres) for one physics step.
    Formula: proportional to power = F * v, with idle floor.
    Uses: accel_ms2, current_mass_kg, speed_ms.
    Replace with a real emission model when available.

    Calibrated loosely on diesel HEMM:  ~0.25 L / (kW·h)  at 85 % eff.
    """
    IDLE_RATE_L_PER_S   = 0.003    # ~10 L/h idle
    EFFICIENCY          = 0.30     # drivetrain efficiency
    CALORIFIC_VALUE_MJ  = 35.8     # MJ/litre diesel
    CALORIFIC_W         = CALORIFIC_VALUE_MJ * 1e6  # J/litre

    mass    = car.current_mass_kg
    accel   = car.accel_ms2
    speed   = car.speed_ms

    # Net traction force (ignore grade for now – no elevation data)
    ROLLING_RESIST = 0.02 * mass * 9.81
    traction = mass * accel + ROLLING_RESIST
    traction = max(traction, 0.0)          # engine only pulls, not brakes

    power_W  = traction * speed            # watts
    energy_J = power_W * dt               # joules this step

    # Convert J → litres, add idle floor
    fuel_L   = energy_J / (CALORIFIC_W * EFFICIENCY)
    fuel_L  += IDLE_RATE_L_PER_S * dt

    return round(max(fuel_L, 0.0), 6)

def _co2_kg(fuel_L: float) -> float:
    """CO₂ from diesel combustion: 2.68 kg CO₂ per litre."""
    return round(fuel_L * 2.68, 6)

# ── main class ────────────────────────────────────────────────────────────────

class SimLogger:
    """
    Attach to the simulator main loop.  Writes two CSV files:

    telemetry.csv columns:
        sim_time_s, wall_time, truck_id,
        x_m, y_m, speed_kmh, accel_ms2,
        cargo_kg, op_state,
        current_node, target_node,
        fuel_L_step, co2_kg_step

    events.csv columns:
        sim_time_s, wall_time, event_type, truck_id,
        node, cargo_kg, coal_remaining_kg, extra
    """

    # ── Ordered state machine: what transition each (from, to) pair means ──
    _TRANSITION_EVENT = {
        # from_state               to_state              event_type
        ("GOING_TO_ENDPOINT",    "LOADING")          : "ARRIVE_LOAD_ZONE",
        ("LOADING",              "TURNING_AROUND")   : "LOAD_COMPLETE",
        ("RETURNING_TO_START",   "UNLOADING")        : "ARRIVE_DUMP_ZONE",
        ("UNLOADING",            "TURNING_AROUND")   : "UNLOAD_COMPLETE",
        ("TURNING_AROUND",       "GOING_TO_ENDPOINT"): "DEPART_DUMP_ZONE",
        ("TURNING_AROUND",       "RETURNING_TO_START"): "DEPART_LOAD_ZONE",
    }

    def __init__(self, log_dir: str = LOG_DIR, hz: float = TELEMETRY_HZ):
        self._hz        = hz
        self._interval  = 1.0 / max(hz, 0.001)
        self._telem_acc = 0.0   # accumulator for telemetry sampling

        # ── per-truck state shadow (for transition detection) ──
        self._prev_state: dict[int, str] = {}    # truck_id -> last known op_state

        # ── per-truck trip tracking ──
        # trip = one load-zone → dump-zone cycle
        self._trip_start: dict[int, float]  = {}   # truck_id -> sim_time when trip began
        self._trip_fuel:  dict[int, float]  = {}   # truck_id -> cumulative fuel this trip
        self._trip_co2:   dict[int, float]  = {}   # truck_id -> cumulative CO₂ this trip
        self._trip_count: dict[int, int]    = {}   # truck_id -> completed trip count

        # ── cumulative fuel / CO₂ per truck (whole session) ──
        self._total_fuel: dict[int, float] = {}
        self._total_co2:  dict[int, float] = {}

        # ── file setup ──
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        telem_path  = os.path.join(log_dir, f"telemetry_{ts}.csv")
        events_path = os.path.join(log_dir, f"events_{ts}.csv")

        self._telem_f  = open(telem_path,  "w", newline="", encoding="utf-8")
        self._events_f = open(events_path, "w", newline="", encoding="utf-8")

        self._telem_w  = csv.writer(self._telem_f)
        self._events_w = csv.writer(self._events_f)

        # Write headers
        self._telem_w.writerow([
            "sim_time_s", "wall_time", "truck_id",
            "x_m", "y_m", "speed_kmh", "accel_ms2",
            "cargo_kg", "op_state",
            "current_node", "target_node",
            "fuel_L_step", "co2_kg_step",
        ])
        self._events_w.writerow([
            "sim_time_s", "wall_time", "event_type", "truck_id",
            "node", "cargo_kg", "coal_remaining_kg", "extra",
        ])

        self._telem_f.flush()
        self._events_f.flush()

        print(f"[SimLogger] Logging started -> {telem_path}")
        print(f"[SimLogger]                 -> {events_path}")

    # ── public API ────────────────────────────────────────────────────────────

    def tick(self, sim_time: float, sim_dt: float, cars: list, dispatcher) -> None:
        """
        Call once per simulation step, inside `if sim_dt > 0` block.

        Parameters
        ----------
        sim_time   : current simulation clock (seconds, you maintain this)
        sim_dt     : time elapsed this step (seconds)
        cars       : list of Car objects
        dispatcher : Dispatcher instance
        """
        wall_time = datetime.now().isoformat(timespec="milliseconds")

        # ── 1. Per-step fuel & emission accumulation + event detection ──
        for car in cars:
            tid = car.id
            fuel_step = _fuel_proxy(car, sim_dt)
            co2_step  = _co2_kg(fuel_step)

            # Initialise accumulators on first sight
            if tid not in self._total_fuel:
                self._total_fuel[tid] = 0.0
                self._total_co2[tid]  = 0.0
                self._trip_fuel[tid]  = 0.0
                self._trip_co2[tid]   = 0.0
                self._trip_count[tid] = 0
                self._trip_start[tid] = sim_time

            self._total_fuel[tid] += fuel_step
            self._total_co2[tid]  += co2_step
            self._trip_fuel[tid]  += fuel_step
            self._trip_co2[tid]   += co2_step

            # ── Transition detection ──
            prev  = self._prev_state.get(tid)
            curr  = car.op_state

            if prev is not None and prev != curr:
                self._handle_transition(
                    sim_time, wall_time,
                    prev, curr, car, dispatcher,
                    fuel_step, co2_step,
                )

            self._prev_state[tid] = curr

        # ── 2. Telemetry sampling ──
        self._telem_acc += sim_dt
        if self._telem_acc >= self._interval:
            self._telem_acc = 0.0
            self._write_telemetry(sim_time, wall_time, cars, sim_dt)

    def close(self) -> None:
        """Flush and close log files. Call after the simulation loop ends."""
        self._telem_f.flush();  self._telem_f.close()
        self._events_f.flush(); self._events_f.close()
        print("[SimLogger] Log files closed.")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _write_telemetry(self, sim_time, wall_time, cars, sim_dt):
        for car in cars:
            tid      = car.id
            fuel_s   = _fuel_proxy(car, sim_dt)
            co2_s    = _co2_kg(fuel_s)

            self._telem_w.writerow([
                round(sim_time, 3),
                wall_time,
                tid,
                round(car.x_m, 3),
                round(car.y_m, 3),
                _speed_kmh(car.speed_ms),
                round(car.accel_ms2, 4),
                _cargo_kg(car),
                car.op_state,
                car.current_node_name,
                car.target_node_name,
                fuel_s,
                co2_s,
            ])
        self._telem_f.flush()

    def _handle_transition(
        self, sim_time, wall_time,
        from_state, to_state,
        car, dispatcher,
        fuel_step, co2_step,
    ):
        tid       = car.id
        key       = (from_state, to_state)
        event_type = self._TRANSITION_EVENT.get(key)

        if event_type is None:
            # Unknown / intermediate transition – still log it generically
            event_type = f"STATE_{from_state}_TO_{to_state}"

        node   = car.target_node_name or car.current_node_name
        cargo  = _cargo_kg(car)

        # Coal remaining at the involved site
        coal_rem = dispatcher.get_coal_remaining(node) \
                   if hasattr(dispatcher, "get_coal_remaining") else "n/a"
        if coal_rem == float("inf"):
            coal_rem = "unlimited"
        else:
            coal_rem = round(float(coal_rem), 1)

        # Build extra metadata
        extra_parts = []

        if event_type == "LOAD_COMPLETE":
            extra_parts.append(f"trip_fuel_so_far_L={round(self._trip_fuel[tid], 4)}")

        if event_type == "UNLOAD_COMPLETE":
            # Trip completed
            self._trip_count[tid] += 1
            trip_dur   = round(sim_time - self._trip_start[tid], 2)
            trip_fuel  = round(self._trip_fuel[tid], 4)
            trip_co2   = round(self._trip_co2[tid], 6)
            extra_parts += [
                f"trip_id={self._trip_count[tid]}",
                f"trip_duration_s={trip_dur}",
                f"trip_fuel_L={trip_fuel}",
                f"trip_co2_kg={trip_co2}",
            ]
            # Reset trip accumulators
            self._trip_start[tid] = sim_time
            self._trip_fuel[tid]  = 0.0
            self._trip_co2[tid]   = 0.0

        if event_type in ("DEPART_LOAD_ZONE", "DEPART_DUMP_ZONE"):
            extra_parts.append(f"session_fuel_L={round(self._total_fuel[tid], 4)}")
            extra_parts.append(f"session_co2_kg={round(self._total_co2[tid], 6)}")

        extra = "|".join(extra_parts) if extra_parts else ""

        self._events_w.writerow([
            round(sim_time, 3),
            wall_time,
            event_type,
            tid,
            node,
            cargo,
            coal_rem,
            extra,
        ])
        self._events_f.flush()

        print(f"[SimLogger] t={sim_time:.1f}s  Truck {tid}  {event_type}  @ {node}  cargo={cargo:.0f}kg")
