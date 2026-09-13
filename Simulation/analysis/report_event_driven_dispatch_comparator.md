# Report: Event-Driven Dispatch Comparator

## 1. Objective

The objective of this module is to compare dispatch strategies under a unified, event-driven offline replay framework using simulator logs. The comparator evaluates productivity-efficiency trade-offs while keeping the historical dispatch timeline fixed.

## 2. Why an Event-Driven Comparator?

A static replay with fixed trip assumptions can hide policy differences. The event-driven design improves realism by replaying each true dispatch moment and reconstructing assignment context from logs.

This enables policy comparison without rerunning the full simulator for every strategy.

## 3. Data Inputs

Primary files:

- `analysis/trips_graph.csv`
- `analysis/queue_events.csv`
- `logs/merged_events.csv`

Key event source:

- `DEPART_DUMP_ZONE` events are used to identify dispatch moments and truck location context.

## 4. Methodology

### 4.1 Dispatch event construction

`build_dispatch_events()` creates one decision row per dispatch:

- dispatch timestamp (`sim_time_s`)
- truck id
- current node (dump-side position)
- actual logged mine/dump outcome
- actual duration/fuel/co2 for reference

### 4.2 Route and congestion estimation

- `RouteLookup` stores historical route medians for duration/fuel/co2.
- `QueueIndex` provides queue depth near assignment time and a congestion multiplier.

### 4.3 Policy replay loop

For each policy and each dispatch event:

1. policy chooses a mine,
2. best assignment/dump estimate is computed,
3. trip is simulated into policy-specific state,
4. KPI outputs are accumulated.

## 5. Policies Evaluated

- `GREEDY_NEAREST`
- `ETA_MIN`
- `ETA_AWARE`
- `FIFO`
- `LOAD_BALANCE`
- `ROUND_ROBIN`

### 5.1 ETA_AWARE policy

ETA_AWARE adds a learned correction over route-based ETA using historical residual patterns:

- global bias,
- mine-level bias,
- truck-level bias.

It remains lightweight and dispatch-time compatible, while being more context-aware than route medians alone.

## 6. Current Results (182 dispatch events)

From `analysis/dispatch_policy_scores.csv`:

- Best throughput: **ROUND_ROBIN** (`8.28 trips/hr`)
- Best avg duration: **ETA_AWARE** (`196.4 s`)
- Best total fuel: **ETA_AWARE** (`114.994 L`)
- Best weighted composite: **ETA_AWARE** (`60.0`)

Composite weights used:

- Throughput: 40%
- Total fuel: 25%
- Avg duration: 20%
- Mine balance std: 15%

`duration_mae_s` is reported separately (estimation quality) and excluded from composite KPI ranking.

## 7. Interpretability Outputs

### 7.1 Assignment summary

Saved in `analysis/dispatch_assignment_summary.csv`.

Includes:

- mine share and dump share per policy,
- top mine/dump concentration,
- policy-level mean estimated duration and queue factor,
- mine match rate to historical choices.

### 7.2 Policy disagreement matrix

Saved in:

- `analysis/dispatch_policy_disagreement_matrix.csv`
- `analysis/plots/policy_disagreement_heatmap.png`

Key pairwise mine-choice disagreement:

- `ETA_AWARE` vs `ETA_MIN`: **0.00%**
- `ETA_AWARE` vs `GREEDY_NEAREST`: **1.10%**
- `ETA_AWARE` vs FIFO-family policies: **44.51%**

This indicates two behavioral clusters:

1. ETA-based policies (`ETA_AWARE`, `ETA_MIN`, near-`GREEDY_NEAREST`)
2. FIFO-family policies (`FIFO`, `LOAD_BALANCE`, `ROUND_ROBIN`)

## 8. Generated Artifacts

CSV outputs:

- `analysis/dispatch_policy_scores.csv`
- `analysis/dispatch_decisions.csv`
- `analysis/dispatch_assignment_summary.csv`
- `analysis/dispatch_policy_disagreement_matrix.csv`

Plot outputs:

- `analysis/plots/dispatch_policy_comparison.png`
- `analysis/plots/mine_utilization.png`
- `analysis/plots/cumulative_coal.png`
- `analysis/plots/duration_distribution_by_policy.png`
- `analysis/plots/policy_disagreement_heatmap.png`

## 9. Limitations

- Offline replay approximates policy impact and does not model full closed-loop multi-agent feedback.
- Throughput differences are modest (~1-2%) and should be interpreted as directional.
- Estimation quality depends on route/queue approximation quality in sparse contexts.

## 10. Conclusion

The event-driven dispatch comparator provides a practical and reproducible framework to compare dispatch logic under realistic decision timing. The addition of ETA_AWARE demonstrates how predictive modeling can be integrated into prescriptive dispatch analysis, producing transparent KPI trade-offs and interpretable behavioral differences.
