"""
analytics_overlay.py  –  Visual analytics layer for the HEMM simulator.

Draws on top of the existing pygame render. Zero changes to car.py,
graphics.py, dispatcher.py, or any other existing file.

Usage in Simulation/main.py:
-----------------------------------------------------------------------
    from analytics_overlay import AnalyticsOverlay      # ADD at top

    overlay = AnalyticsOverlay(screen)                  # ADD after pygame.init()

    # Inside main loop, AFTER pygame.display.flip() — NO, BEFORE it:
    # Replace the existing HUD block with:
    overlay.draw(screen, cars, dispatcher, sim_time, sim_speed, paused, g_to_s, scale)

    pygame.display.flip()                               # keep as-is
-----------------------------------------------------------------------
"""

import pygame
import math
import numpy as np
from config import MASS_KG, CARGO_TON, SPEED_MS_EMPTY, SPEED_MS_LOADED
from Map import map_loader as map_data

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BG_PANEL   = (10,  14,  20,  210)   # dark navy, semi-transparent
C_BG_CARD    = (18,  24,  34,  230)
C_BORDER     = (40,  60,  90,  255)
C_ACCENT     = (0,   200, 120, 255)   # emerald green
C_WARN       = (255, 160, 30,  255)   # amber
C_DANGER     = (220, 50,  50,  255)   # red
C_TEXT_HI    = (220, 230, 245, 255)   # bright white-blue
C_TEXT_LO    = (110, 130, 160, 255)   # muted

# State → colour
STATE_COLOR = {
    "GOING_TO_ENDPOINT"  : (30,  180, 90),    # green  – heading to mine
    "LOADING"            : (255, 200, 0),      # yellow – loading
    "TURNING_AROUND"     : (100, 160, 255),    # blue   – manoeuvring
    "RETURNING_TO_START" : (255, 120, 30),     # orange – loaded, heading to dump
    "UNLOADING"          : (200, 80,  200),    # purple – unloading
}
STATE_SHORT = {
    "GOING_TO_ENDPOINT"  : "→ MINE",
    "LOADING"            : "⬛ LOAD",
    "TURNING_AROUND"     : "↺ TURN",
    "RETURNING_TO_START" : "→ DUMP",
    "UNLOADING"          : "⬛ DUMP",
}

# Zone marker sizes
MINE_RADIUS  = 10
DUMP_RADIUS  = 10


class AnalyticsOverlay:
    def __init__(self, screen: pygame.Surface):
        pygame.font.init()
        self._screen_w, self._screen_h = screen.get_size()

        # Fonts
        self.font_title  = pygame.font.SysFont("Consolas", 13, bold=True)
        self.font_body   = pygame.font.SysFont("Consolas", 12)
        self.font_small  = pygame.font.SysFont("Consolas", 10)
        self.font_big    = pygame.font.SysFont("Consolas", 15, bold=True)

        # Per-truck session accumulators (trips, fuel, coal)
        self._trips      : dict[int, int]   = {}
        self._fuel       : dict[int, float] = {}
        self._coal_moved : dict[int, float] = {}
        self._prev_state : dict[int, str]   = {}
        self._trip_start : dict[int, float] = {}
        self._trip_times : dict[int, list]  = {}   # recent trip durations

        # Fleet-level totals
        self._fleet_coal  = 0.0
        self._fleet_fuel  = 0.0
        self._start_time  = None   # first tick sim_time

        # Congestion heatmap: node_name → hit_count
        self._congestion : dict[str, int] = {}

    # ── public entry point ────────────────────────────────────────────────────

    def draw(self, screen, cars, dispatcher, sim_time,
             sim_speed, paused, g_to_s, scale):
        """Call this once per frame, before pygame.display.flip()."""

        if self._start_time is None:
            self._start_time = sim_time

        self._screen_w, self._screen_h = screen.get_size()

        self._update_accumulators(cars, dispatcher, sim_time)
        self._draw_zone_markers(screen, g_to_s, scale, dispatcher)
        self._draw_trucks(screen, cars, g_to_s, scale)
        self._draw_right_panel(screen, cars, dispatcher, sim_time, sim_speed, paused)
        self._draw_congestion_heatmap(screen, cars, g_to_s, scale)

    # ── internal: accumulator update ─────────────────────────────────────────

    def _update_accumulators(self, cars, dispatcher, sim_time):
        for car in cars:
            tid   = car.id
            state = car.op_state
            prev  = self._prev_state.get(tid, state)

            if tid not in self._trips:
                self._trips[tid]      = 0
                self._fuel[tid]       = 0.0
                self._coal_moved[tid] = 0.0
                self._trip_start[tid] = sim_time
                self._trip_times[tid] = []

            # Trip complete when unloading finishes
            if prev == "UNLOADING" and state == "TURNING_AROUND":
                self._trips[tid] += 1
                duration = sim_time - self._trip_start[tid]
                self._trip_times[tid].append(duration)
                if len(self._trip_times[tid]) > 5:
                    self._trip_times[tid].pop(0)
                self._trip_start[tid] = sim_time

                cargo = max(0.0, car.current_mass_kg - MASS_KG)
                self._coal_moved[tid] += cargo
                self._fleet_coal      += cargo

            # Rough fuel proxy
            fuel_step = max(0.0, car.accel_ms2) * car.current_mass_kg * 0.000001
            self._fuel[tid]  += fuel_step
            self._fleet_fuel += fuel_step

            self._prev_state[tid] = state

        # Congestion: find nearest node for each car
        self._congestion = {}
        for car in cars:
            c_pos = np.array([car.x_m, car.y_m])
            closest, min_d = None, float("inf")
            for name, pos in map_data.NODES.items():
                d = np.linalg.norm(pos - c_pos)
                if d < min_d:
                    min_d, closest = d, name
            if closest and min_d < 25.0:
                self._congestion[closest] = self._congestion.get(closest, 0) + 1

    # ── internal: zone markers ────────────────────────────────────────────────

    def _draw_zone_markers(self, screen, g_to_s, scale, dispatcher):
        r = max(6, int(MINE_RADIUS * scale * 0.6))

        for node_name in map_data.LOAD_ZONES:
            pos = map_data.NODES.get(node_name)
            if pos is None: continue
            sx, sy = g_to_s(pos)

            # Coal remaining bar
            coal_rem = dispatcher.get_coal_remaining(node_name)
            if coal_rem == float("inf"):
                fill = 1.0
            else:
                raw = dispatcher.coal_capacities.get(node_name, float("inf"))
                fill = coal_rem / raw if (raw and raw != float("inf")) else 1.0
            fill = max(0.0, min(1.0, fill))

            # Outer ring
            bar_col = (
                int(50 + 200 * fill),
                int(200 * fill),
                30,
            )
            pygame.draw.circle(screen, bar_col, (sx, sy), r + 3, 2)
            pygame.draw.circle(screen, (0, 180, 60), (sx, sy), r, 0)

            # Coal bar above node
            bw, bh = 28, 5
            bx, by = sx - bw // 2, sy - r - 10
            pygame.draw.rect(screen, (40, 40, 40), (bx, by, bw, bh))
            pygame.draw.rect(screen, bar_col, (bx, by, int(bw * fill), bh))
            pygame.draw.rect(screen, (120, 120, 120), (bx, by, bw, bh), 1)

            # Label
            lbl = self.font_small.render(node_name.replace("load_zone_", "M"), True, (0, 220, 80))
            screen.blit(lbl, (sx + r + 2, sy - 6))

        for node_name in map_data.DUMP_ZONES:
            pos = map_data.NODES.get(node_name)
            if pos is None: continue
            sx, sy = g_to_s(pos)
            r2 = max(6, int(DUMP_RADIUS * scale * 0.6))

            # Draw diamond shape
            pts = [(sx, sy - r2), (sx + r2, sy), (sx, sy + r2), (sx - r2, sy)]
            pygame.draw.polygon(screen, (200, 50, 50), pts)
            pygame.draw.polygon(screen, (255, 100, 100), pts, 2)

            # Total dumped
            dumped = dispatcher.site_states.get(node_name, {}).get("coal_dumped", 0)
            lbl = self.font_small.render(f"D:{dumped/1000:.1f}t", True, (255, 120, 120))
            screen.blit(lbl, (sx + r2 + 2, sy - 6))

    # ── internal: truck rendering ─────────────────────────────────────────────

    def _draw_trucks(self, screen, cars, g_to_s, scale):
        """Draw trucks as solid rectangles with ID badge and state colour."""
        from config import CAR_LENGTH_M, CAR_WIDTH_M, METERS_TO_PIXELS

        for car in cars:
            sx, sy   = g_to_s((car.x_m, car.y_m))
            col      = STATE_COLOR.get(car.op_state, (180, 180, 180))
            length_px = max(8, int(CAR_LENGTH_M * METERS_TO_PIXELS * scale))
            width_px  = max(4, int(CAR_WIDTH_M  * METERS_TO_PIXELS * scale))

            # Truck body surface
            surf = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
            surf.fill(col)

            # Cab indicator (front darker stripe)
            cab_w = max(2, length_px // 4)
            cab_col = tuple(max(0, c - 60) for c in col)
            pygame.draw.rect(surf, cab_col, (0, 0, cab_w, width_px))

            rotated = pygame.transform.rotate(surf, -math.degrees(car.angle))
            rect    = rotated.get_rect(center=(sx, sy))
            screen.blit(rotated, rect.topleft)

            # Cargo indicator bar below truck
            cargo_ratio = max(0.0, (car.current_mass_kg - MASS_KG) / (CARGO_TON * 1000))
            if cargo_ratio > 0:
                bw = length_px
                bx = sx - bw // 2
                by = sy + width_px // 2 + 3
                pygame.draw.rect(screen, (40, 40, 40), (bx, by, bw, 4))
                pygame.draw.rect(screen, C_WARN,        (bx, by, int(bw * cargo_ratio), 4))

            # ID badge
            badge = self.font_small.render(f"T{car.id}", True, (255, 255, 255))
            screen.blit(badge, (sx - badge.get_width() // 2, sy - length_px // 2 - 14))

            # Speed label
            spd = self.font_small.render(f"{car.speed_ms*3.6:.0f}k", True, C_TEXT_LO)
            screen.blit(spd, (sx - spd.get_width() // 2, sy + length_px // 2 + 8))

    # ── internal: right analytics panel ──────────────────────────────────────

    def _draw_right_panel(self, screen, cars, dispatcher, sim_time, sim_speed, paused):
        PW, PH   = 220, self._screen_h
        px       = self._screen_w - PW
        panel    = pygame.Surface((PW, PH), pygame.SRCALPHA)
        panel.fill(C_BG_PANEL)
        pygame.draw.line(panel, C_BORDER, (0, 0), (0, PH), 2)

        y = 8

        def txt(text, font, col, indent=8):
            nonlocal y
            surf = font.render(text, True, col)
            panel.blit(surf, (indent, y))
            y += surf.get_height() + 3

        def divider():
            nonlocal y
            pygame.draw.line(panel, C_BORDER, (8, y), (PW - 8, y), 1)
            y += 6

        # ── Header ──
        txt("▶ FLEET ANALYTICS", self.font_title, C_ACCENT)
        elapsed = sim_time - (self._start_time or sim_time)
        mins, secs = divmod(int(elapsed), 60)
        txt(f"  Time: {mins:02d}:{secs:02d}  {'⏸ PAUSED' if paused else f'{sim_speed}x'}", self.font_body, C_TEXT_LO)
        divider()

        # ── Fleet KPIs ──
        txt("FLEET TOTALS", self.font_title, C_TEXT_HI)

        total_trips = sum(self._trips.get(c.id, 0) for c in cars)
        txt(f"  Trips completed : {total_trips}", self.font_body, C_TEXT_HI)

        coal_t = self._fleet_coal / 1000.0
        txt(f"  Coal moved      : {coal_t:.1f} t", self.font_body, C_TEXT_HI)

        # Throughput
        if elapsed > 60:
            tph = total_trips / (elapsed / 3600)
            txt(f"  Throughput      : {tph:.1f} trips/hr", self.font_body, C_ACCENT)
        else:
            txt(f"  Throughput      : --", self.font_body, C_TEXT_LO)

        # Queue pressure
        max_queue = max(
            (dispatcher.site_states.get(z, {}).get("en_route", 0) for z in map_data.LOAD_ZONES),
            default=0,
        )
        q_col = C_DANGER if max_queue >= 2 else C_WARN if max_queue == 1 else C_ACCENT
        txt(f"  Max mine queue  : {max_queue}", self.font_body, q_col)

        divider()

        # ── Per-truck cards ──
        txt("PER-TRUCK STATUS", self.font_title, C_TEXT_HI)

        for car in cars:
            tid   = car.id
            state = car.op_state
            col   = STATE_COLOR.get(state, (180, 180, 180))
            short = STATE_SHORT.get(state, state)

            # Card background
            card_h = 68
            card_surf = pygame.Surface((PW - 16, card_h), pygame.SRCALPHA)
            card_surf.fill(C_BG_CARD)
            pygame.draw.rect(card_surf, col, (0, 0, 3, card_h))   # left accent stripe
            panel.blit(card_surf, (8, y))

            cy = y + 4

            # Truck ID + state
            id_surf = self.font_big.render(f"Truck {tid}", self.font_big, True if False else col)
            id_surf = self.font_big.render(f"Truck {tid}", True, col)
            panel.blit(id_surf, (14, cy)); cy += 16

            st_surf = self.font_body.render(short, True, C_TEXT_HI)
            panel.blit(st_surf, (14, cy)); cy += 14

            # Speed
            spd_txt = f"  {car.speed_ms*3.6:.1f} km/h"
            panel.blit(self.font_small.render(spd_txt, True, C_TEXT_LO), (14, cy))

            # Cargo bar
            cargo_r = max(0.0, (car.current_mass_kg - MASS_KG) / (CARGO_TON * 1000))
            bw = PW - 40
            bx, by2 = 14, cy + 12
            pygame.draw.rect(panel, (30, 30, 30), (bx, by2, bw, 6))
            if cargo_r > 0:
                pygame.draw.rect(panel, C_WARN, (bx, by2, int(bw * cargo_r), 6))
            panel.blit(self.font_small.render(f"cargo {cargo_r*100:.0f}%", True, C_TEXT_LO), (bx, by2 + 8))

            # Trips
            trips_done = self._trips.get(tid, 0)
            avg_t = ""
            if self._trip_times.get(tid):
                avg = sum(self._trip_times[tid]) / len(self._trip_times[tid])
                avg_t = f"  avg {avg:.0f}s"
            tr_surf = self.font_small.render(f"  trips:{trips_done}{avg_t}", True, C_ACCENT)
            panel.blit(tr_surf, (bx + 60, cy))

            y += card_h + 6

        divider()

        # ── Mine coal status ──
        txt("MINE COAL STATUS", self.font_title, C_TEXT_HI)
        for zone in map_data.LOAD_ZONES[:8]:   # show max 8
            rem  = dispatcher.get_coal_remaining(zone)
            raw  = dispatcher.coal_capacities.get(zone, float("inf"))
            name = zone.replace("load_zone_", "Mine ")
            if rem == float("inf"):
                bar_f, rem_str = 1.0, "∞"
            else:
                bar_f   = rem / raw if raw and raw != float("inf") else 1.0
                rem_str = f"{rem/1000:.1f}t"

            bar_f = max(0.0, min(1.0, bar_f))
            col   = C_DANGER if bar_f < 0.2 else C_WARN if bar_f < 0.5 else C_ACCENT

            panel.blit(self.font_small.render(f"  {name:<12} {rem_str:>6}", True, col), (8, y))
            y += 12
            bw2 = PW - 24
            pygame.draw.rect(panel, (30, 30, 30), (12, y, bw2, 4))
            pygame.draw.rect(panel, col,           (12, y, int(bw2 * bar_f), 4))
            y += 8

        screen.blit(panel, (px, 0))

    # ── internal: congestion heatmap dots ────────────────────────────────────

    def _draw_congestion_heatmap(self, screen, cars, g_to_s, scale):
        """Draw a glowing dot on nodes where 2+ trucks are nearby."""
        for node_name, count in self._congestion.items():
            if count < 2: continue
            pos = map_data.NODES.get(node_name)
            if pos is None: continue
            sx, sy = g_to_s(pos)
            r = min(30, 8 * count)
            # Glow effect: multiple transparent circles
            for i in range(3, 0, -1):
                glow = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
                alpha = 40 * i
                pygame.draw.circle(glow, (255, 60, 0, alpha), (r * 2, r * 2), r * i)
                screen.blit(glow, (sx - r * 2, sy - r * 2))
            # Core dot
            pygame.draw.circle(screen, (255, 80, 0), (sx, sy), 5, 0)
