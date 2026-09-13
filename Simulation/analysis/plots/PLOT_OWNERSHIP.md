# Plot Ownership and Organization

This folder now contains module-specific subfolders so it is clear which analysis task generated each plot.

## 1) EDA (Task 3)

Folder: `analysis/plots/eda_task3/`  
Primary report: `analysis/report_eda_task3.md`  
Primary scripts: EDA pipeline outputs and `graph_features.py`/EDA generation steps

Plots:

- `road_dist_vs_duration.png`
- `duration_by_route.png`
- `trip_duration_analysis.png`
- `detour_factor.png`

## 2) ETA Modeling (Task 4)

Folder: `analysis/plots/eta_modeling_task4/`  
Primary report: `analysis/report_eta_modeling_task4.md`  
Primary scripts: `eta_model.py`, `eta_model_v2.py`, `graph_features.py`

Plots:

- `eta_random_forest.png`
- `eta_gradient_boosting.png`
- `eta_linear_regression.png`
- `importance_random_forest.png`
- `importance_gradient_boosting.png`
- `baseline_vs_graph.png`
- `importance_graph_random_forest.png`
- `eta_v2_variant_mae.png`
- `fuel_vs_duration.png`

## 3) Event-Driven Dispatch Comparator

Folder: `analysis/plots/event_driven_dispatch/`  
Primary report: `analysis/report_event_driven_dispatch_comparator.md`  
Primary script: `dispatch_comparator.py`

Plots:

- `dispatch_policy_comparison.png`
- `mine_utilization.png`
- `cumulative_coal.png`
- `duration_distribution_by_policy.png`
- `policy_disagreement_heatmap.png`

## Notes

- Original plot files are kept in `analysis/plots/` for backward compatibility.
- The three subfolders contain organized copies for report-ready usage by module.
