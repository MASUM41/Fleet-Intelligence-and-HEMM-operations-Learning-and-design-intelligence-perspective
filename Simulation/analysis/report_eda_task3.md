# Task 3 Report: Exploratory Data Analysis (EDA)

## 1. Objective

The objective of Task 3 is to explore simulator-generated fleet data and identify the main operational factors affecting haul-cycle time, congestion behavior, and route-level variability. This EDA supports both predictive modeling (Task 4) and dispatch policy analysis.

## 2. Data Sources

EDA was performed using:

- `analysis/trips.csv`
- `analysis/trips_graph.csv`
- `analysis/queue_events.csv`
- `analysis/fleet_kpis.csv`
- `analysis/idle_periods.csv`

Primary focus was on trip-level and route-level records from `trips_graph.csv`, with congestion context from `queue_events.csv`.

## 3. Dataset Snapshot

From the current analysis snapshot:

- Total trips: **182**
- Unique route pairs: **13**
- Queue event rows: **274**

The dataset is sufficient for directional insights and comparative analytics, while still moderate in scale for advanced generalization.

## 4. Core EDA Questions

1. How strongly does route geometry (distance/path complexity) influence trip duration?
2. How much variability exists within the same route?
3. Do raw queue indicators alone explain delays?
4. Which features are likely useful for predictive ETA and dispatch scoring?

## 5. Key Visual Outputs

The following EDA figures were generated and used:

- `analysis/plots/eda/road_dist_vs_duration.png`
- `analysis/plots/eda/duration_by_route.png`
- `analysis/plots/eda/trip_duration_analysis.png`
- `analysis/plots/eda/detour_factor.png`

## 6. Findings

### 6.1 Route distance vs duration

Trip duration generally increases with road distance. The observed trend is positive but not perfectly linear, indicating that distance is important but not the only driver of cycle time.

Interpretation:

- Distance-related features should be included in ETA modeling.
- Residual spread suggests additional effects from queueing and operational state.

### 6.2 High within-route variance

For the same route, trip duration can vary substantially. This indicates route identity alone is insufficient for accurate ETA prediction.

Interpretation:

- Dynamic conditions (queue depth, short-term congestion, loading delays) must be modeled.
- Statistical route medians are useful baselines but cannot capture all variability.

### 6.3 Queue signal behavior

Raw queue-at-load values alone show weak explanatory power for trip duration. However, engineered queue context over recent windows (counts, maxima, means, threshold exceedance) provides better signal for downstream models.

Interpretation:

- Temporal queue feature engineering is preferable to a single static queue snapshot.

### 6.4 Detour and route-efficiency effects

Detour-related metrics expose differences in path efficiency and contribute additional explanatory power for operational variability.

Interpretation:

- Graph-aware features are operationally meaningful even when marginal gains in MAE are modest.

## 7. Implications for Task 4 (ETA Modeling)

EDA directly motivates:

- adding graph-distance and detour features,
- using queue-history windows at dispatch/mid-trip timestamps,
- separating dispatch-time prediction from mid-trip prediction.

These were implemented in `eta_model_v2.py`.

## 8. Limitations of EDA

- Observational analysis does not establish strict causality.
- Congestion effects may be partially confounded by run-specific conditions.
- Dataset size is moderate; rare operational regimes may be underrepresented.

## 9. Conclusion

Task 3 EDA confirms that trip duration is driven by a combination of static route geometry and dynamic operational state. Route distance is a strong baseline signal, but queue/context features are required to explain route-level variance. These findings justify the ETA feature design and event-driven dispatch replay methodology used in subsequent tasks.
