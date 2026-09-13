# Fleet Intelligence for HEMM Operations: Report Draft

## 1) Problem Statement

This project extends an autonomous haul-truck simulator with a fleet intelligence layer for:

- predictive analytics (ETA estimation),
- congestion-aware operational diagnostics,
- prescriptive dispatch policy comparison under throughput-fuel-duration trade-offs.

The goal is to move from animation/rule execution to decision-support analytics using simulator traces.

## 2) Data Pipeline and Experimental Setup

All experiments were run from `Simulation/` using merged logs and extracted KPI tables.

- Trips used: **182**
- Run groups for grouped CV: **9 (`run_id`)**
- Unique routes: **13**

Primary generated datasets:

- `analysis/trips.csv`
- `analysis/trips_graph.csv`
- `analysis/queue_events.csv`
- `analysis/fleet_kpis.csv`
- `analysis/idle_periods.csv`
- `logs/merged_events.csv`

## 3) Exploratory Observations

EDA indicates that route geometry and operational state jointly drive cycle-time variability:

- Total road distance has moderate positive relation with trip duration.
- Same route can show large spread in completion time, suggesting congestion and local delays.
- Raw queue-at-load alone is weak; engineered temporal queue features are more informative.

Representative figures are available in:

- `analysis/plots/eda/road_dist_vs_duration.png`
- `analysis/plots/eda/duration_by_route.png`
- `analysis/plots/eda/trip_duration_analysis.png`
- `analysis/plots/eda/detour_factor.png`

## 4) ETA Modeling

### 4.1 Baseline ETA (`eta_model.py`)

From `analysis/eta_model_summary.txt`:

- Best model: **Random Forest**
- Validation: **Leave-One-Out CV**
- Best MAE: **15.9 s** (~7.4% of average trip duration)

This is a strong optimistic baseline, but LOO can overestimate generalization when runs are correlated.

### 4.2 ETA v2 (`eta_model_v2.py`)

ETA v2 uses grouped CV by `run_id` and separates:

- **dispatch-time ETA** (decision-time prediction),
- **mid-trip ETA** (prediction after partial trip observations).

From `analysis/eta_v2_summary.txt`:

- Best dispatch-time variant: `dispatch_graph` (RF), **63.98 s MAE**
- Best mid-trip variant: `midtrip_full` (RF), **34.06 s MAE**

Interpretation: mid-trip ETA is easier because progress is already observed; dispatch-time ETA remains the operationally relevant hard task.

## 5) Event-Driven Dispatch Comparator

`dispatch_comparator.py` was upgraded from static replay to event-driven replay:

- dispatch moments from `DEPART_DUMP_ZONE` events,
- truck current-node reconstruction,
- route-based duration/fuel estimates by `(load_zone, dump_zone)`,
- queue-conditioned congestion factor at assignment time,
- multiple policy replay over identical dispatch events.

Evaluated policies:

- `GREEDY_NEAREST`
- `ETA_MIN`
- `ETA_AWARE`
- `FIFO`
- `LOAD_BALANCE`
- `ROUND_ROBIN`

## 6) Dispatch Results (Current Run)

From `analysis/dispatch_policy_scores.csv`:

- Best throughput: **ROUND_ROBIN (8.28 trips/hr)**
- Best fuel and avg duration: **ETA_AWARE (114.994 L, 196.4 s)**
- Best weighted composite: **ETA_AWARE (60.0)**

Composite weights:

- Throughput: 40%
- Total fuel: 25%
- Avg duration: 20%
- Mine balance std: 15%

Duration MAE is reported separately and excluded from composite ranking because it measures estimate error, not direct fleet KPI performance.

## 7) Policy Behavior Analysis

### 7.1 Assignment distribution

Saved in `analysis/dispatch_assignment_summary.csv`.

Current run shows similar top route tendency across policies (same dominant mine/dump share), so KPI differences are driven by finer timing and route-cost differences rather than major route redistribution.

### 7.2 Policy disagreement matrix

Saved in:

- `analysis/dispatch_policy_disagreement_matrix.csv`
- `analysis/plots/policy_disagreement_heatmap.png`

Key disagreement findings:

- `ETA_AWARE` vs `ETA_MIN`: **0.00%**
- `ETA_AWARE` vs `GREEDY_NEAREST`: **1.10%**
- `ETA_AWARE` vs FIFO-family policies: **44.51%**

This suggests two clusters of behavior: ETA-based policies and FIFO-family policies.

## 8) Limitations

- Dispatch comparison is offline replay, not live simulator A/B with closed-loop traffic feedback.
- Route/queue estimators introduce approximation error.
- Throughput differences are modest on current dataset (~1-2%), so conclusions should be treated as directional.
- Baseline ETA (LOO) and ETA v2 (grouped CV) are not directly comparable due to validation protocol differences.

## 9) Conclusion

The project successfully establishes a reproducible fleet intelligence pipeline:

- KPI extraction from simulator logs,
- ETA modeling with grouped evaluation,
- event-driven policy replay with interpretable policy diagnostics.

Under current weighted operational priorities, **ETA_AWARE** provides the best overall dispatch score, while **ROUND_ROBIN** maximizes throughput. This demonstrates measurable trade-offs and provides a concrete basis for future live dispatch integration.

## 10) Next Work

- Integrate ETA-aware policy into live `dispatcher.py` for online A/B testing.
- Expand dispatch-time feature set with stronger temporal congestion signals.
- Evaluate robustness on larger and more diverse run sets.
- Extend graph-aware modeling (including GNN options) in Semester 2.
