# ════════════════════════════════════════════════════════════════════════════
#  PATCH  –  add logging to Simulation/main.py
#  Copy sim_logger.py into your Simulation/ folder first.
#  Then make exactly these 5 small changes.
# ════════════════════════════════════════════════════════════════════════════

# ── CHANGE 1 ─────────────────────────────────────────────────────────────────
# At the top of the file, after all existing imports, add:

from sim_logger import SimLogger


# ── CHANGE 2 ─────────────────────────────────────────────────────────────────
# Inside run_simulation(), right after this comment block:
#     # --- Initial MPC run for all trucks ---
#     for car in cars:
#         car.run_mpc([])
# ADD these two lines:

sim_time = 0.0
logger   = SimLogger()


# ── CHANGE 3 ─────────────────────────────────────────────────────────────────
# Inside the main loop, at the START of `if sim_dt > 0:` block.
# The block currently starts with:
#     if sim_dt > 0:
#         mpc_timer += sim_dt
#         ...
# ADD this line immediately after `if sim_dt > 0:`:

        sim_time += sim_dt


# ── CHANGE 4 ─────────────────────────────────────────────────────────────────
# Inside the same `if sim_dt > 0:` block, after the Physics Update loop
# (after the `for idx, car in enumerate(cars):` block ends), ADD:

            logger.tick(sim_time, sim_dt, cars, dispatcher)


# ── CHANGE 5 ─────────────────────────────────────────────────────────────────
# After `pygame.quit()` at the very end of run_simulation(), ADD:

    logger.close()


# ════════════════════════════════════════════════════════════════════════════
#  CONTEXT VIEW  –  showing each change in place
# ════════════════════════════════════════════════════════════════════════════

# ---------- CHANGE 1: top of file ----------
"""
import pygame
import numpy as np
...
from Algorithm.planner_registry import load_local_planner, DEFAULT_GLOBAL_PLANNER, DEFAULT_LOCAL_PLANNER

from sim_logger import SimLogger          # ← ADD THIS LINE
"""

# ---------- CHANGE 2: after initial MPC runs ----------
"""
    # --- Initial MPC run for all trucks ---
    for car in cars:
        car.run_mpc([])

    sim_time = 0.0          # ← ADD
    logger   = SimLogger()  # ← ADD

    # --- Timers ---
    mpc_timer = 0.0
"""

# ---------- CHANGE 3 + 4: inside main loop ----------
"""
    while running:
        frame_dt = clock.tick(60) / 1000.0
        if frame_dt == 0: continue

        sim_dt = 0.0 if paused else frame_dt * sim_speed
        if sim_dt > 0:
            sim_time += sim_dt      # ← ADD (Change 3)
            mpc_timer += sim_dt
            traffic_update_timer += sim_dt
            global_opt_timer += sim_dt

            # ... (traffic update, global opt, MPC update, physics loop all unchanged) ...

            # Physics Update loop ends here ↑

            logger.tick(sim_time, sim_dt, cars, dispatcher)   # ← ADD (Change 4)
"""

# ---------- CHANGE 5: after pygame.quit() ----------
"""
    pygame.quit()
    logger.close()          # ← ADD
"""
