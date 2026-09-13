# Task 4 Report: ETA Modeling

## 1. Objective

The objective of Task 4 is to build and evaluate ETA models for haul-truck trips using simulator-derived features, then compare baseline and improved formulations under realistic validation schemes.

## 2. Data and Modeling Inputs

Primary inputs:

- `analysis/trips.csv`
- `analysis/trips_graph.csv`
- `analysis/queue_events.csv`

Main scripts:

- Baseline ETA: `eta_model.py`
- Enhanced ETA: `eta_model_v2.py`

Summary outputs:

- `analysis/eta_model_summary.txt`
- `analysis/eta_v2_summary.txt`
- `analysis/eta_predictions.csv`
- `analysis/eta_v2_predictions.csv`

## 3. Baseline ETA (v1)

From `eta_model.py` summary:

- Trips: **182** total
- Normal trips: **178**
- Outliers: **4** (queue-delayed)
- Best model: **Random Forest**
- Validation: **Leave-One-Out CV**
- Best MAE: **15.9 s** (about **7.4%** of average trip duration)

### 3.1 Interpretation

The v1 baseline demonstrates that tabular features can produce low error on this dataset. However, LOO-CV may be optimistic when runs are correlated, motivating grouped validation in ETA v2.

## 4. Enhanced ETA (v2): Grouped CV + Feature Design

`eta_model_v2.py` introduces:

1. **Grouped cross-validation by `run_id`** (stronger generalization test)
2. **Dispatch-time vs mid-trip task split**
3. **Graph-aware route features** (distance, detour, edge metrics)
4. **Temporal queue context features** for dispatch and mid-trip timestamps

### 4.1 Why split dispatch-time and mid-trip?

- **Dispatch-time ETA**: harder, but operationally critical for decision-making.
- **Mid-trip ETA**: easier because partial trip progress is already observed.

This separation avoids mixing two fundamentally different prediction problems.

## 5. ETA v2 Results (Grouped CV)

From `analysis/eta_v2_summary.txt`:

### Dispatch-time variants

- `dispatch_baseline` (RF): **64.60 s MAE**
- `dispatch_graph` (RF): **63.98 s MAE** (best dispatch-time)
- `dispatch_queue` (RF): **68.01 s MAE**
- `dispatch_full` (RF): **67.68 s MAE**

### Mid-trip variants

- `midtrip_baseline` (RF): **35.09 s MAE**
- `midtrip_graph` (RF): **35.58 s MAE**
- `midtrip_queue` (RF): **36.27 s MAE**
- `midtrip_full` (RF): **34.06 s MAE** (best mid-trip)

## 6. Comparative Discussion

### 6.1 v1 vs v2 metrics are not directly interchangeable

v1 (15.9 s) uses LOO-CV; v2 uses grouped CV by run. Grouped CV is stricter and typically yields larger, more realistic errors.

### 6.2 Dispatch-time remains the harder real-time task

The dispatch-time MAE (~64 s) is substantially higher than mid-trip (~34 s), consistent with lower information availability at assignment time.

### 6.3 Graph features provide modest dispatch gain

`dispatch_graph` slightly outperforms `dispatch_baseline`, supporting inclusion of graph-distance signals for decision-time prediction.

## 7. Practical Use in Fleet Intelligence

- Use **dispatch-time ETA** for assignment support and policy scoring.
- Use **mid-trip ETA** for progress monitoring and dynamic rescheduling.

This aligns predictive outputs to operational decision moments.

## 8. Limitations

- Moderate sample size and route diversity.
- Queue effects are partially noisy and timestamp-sensitive.
- Extreme congestion cases remain limited in count.

## 9. Conclusion

Task 4 establishes a robust ETA modeling pipeline with stronger validation and task-aware design. The key contribution is not only lower error in selected variants, but also a clearer mapping from model outputs to operational usage (dispatch-time decisions vs in-trip monitoring).
