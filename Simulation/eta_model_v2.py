"""
eta_model_v2.py  -  ETA v2 with grouped CV, graph, and queue features.

Reads:
    analysis/trips.csv
    analysis/trips_graph.csv
    analysis/queue_events.csv

Writes:
    analysis/eta_v2_predictions.csv
    analysis/eta_v2_summary.txt
    analysis/plots/eta_v2_variant_mae.png

Run from Simulation/ folder:
    python eta_model_v2.py
"""

import os
import warnings
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT_DIR = "analysis"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trips = pd.read_csv(os.path.join(OUT_DIR, "trips.csv"))
    trips_graph = pd.read_csv(os.path.join(OUT_DIR, "trips_graph.csv"))
    queue_events = pd.read_csv(os.path.join(OUT_DIR, "queue_events.csv"))

    trips = trips.dropna(subset=["total_trip_duration_s"]).copy()
    return trips, trips_graph, queue_events


def nearest_queue_depth(
    queue_df: pd.DataFrame,
    zone: str,
    timestamp: float,
    event_type: str,
    tolerance_s: float = 5.0,
) -> float:
    if pd.isna(timestamp) or not zone:
        return np.nan
    sub = queue_df[(queue_df["zone"] == zone) & (queue_df["event_type"] == event_type)]
    if sub.empty:
        return np.nan
    diffs = (sub["sim_time_s"] - timestamp).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] <= tolerance_s:
        return float(sub.loc[idx, "trucks_queued"])
    return np.nan


def rolling_queue_count(
    queue_df: pd.DataFrame,
    zone: str,
    timestamp: float,
    window_s: float = 120.0,
) -> float:
    if pd.isna(timestamp) or not zone:
        return np.nan
    mask = (
        (queue_df["zone"] == zone)
        & (queue_df["sim_time_s"] <= timestamp)
        & (queue_df["sim_time_s"] >= (timestamp - window_s))
    )
    return float(mask.sum())


def rolling_queue_stats(
    queue_df: pd.DataFrame,
    zone: str,
    timestamp: float,
    window_s: float = 120.0,
) -> Dict[str, float]:
    if pd.isna(timestamp) or not zone:
        return {
            "count": np.nan,
            "max_queue": np.nan,
            "mean_queue": np.nan,
            "ge_3": np.nan,
        }
    mask = (
        (queue_df["zone"] == zone)
        & (queue_df["sim_time_s"] <= timestamp)
        & (queue_df["sim_time_s"] >= (timestamp - window_s))
    )
    sub = queue_df.loc[mask, "trucks_queued"]
    if sub.empty:
        return {"count": 0.0, "max_queue": 0.0, "mean_queue": 0.0, "ge_3": 0.0}
    return {
        "count": float(sub.count()),
        "max_queue": float(sub.max()),
        "mean_queue": float(sub.mean()),
        "ge_3": float((sub >= 3).any()),
    }


def add_queue_features(df: pd.DataFrame, queue_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    queue_df = queue_df.copy()
    queue_df["sim_time_s"] = pd.to_numeric(queue_df["sim_time_s"], errors="coerce")
    queue_df["trucks_queued"] = pd.to_numeric(queue_df["trucks_queued"], errors="coerce")
    queue_df = queue_df.dropna(subset=["sim_time_s", "trucks_queued"])

    q_load = []
    q_dump = []
    q_recent_load = []
    q_recent_dump = []
    q_recent_load_max = []
    q_recent_dump_max = []
    q_recent_load_mean = []
    q_recent_dump_mean = []
    q_ge3_load = []
    q_ge3_dump = []
    dispatch_load_count = []
    dispatch_dump_count = []
    dispatch_load_max = []
    dispatch_dump_max = []
    dispatch_load_mean = []
    dispatch_dump_mean = []
    dispatch_ge3_load = []
    dispatch_ge3_dump = []
    midtrip_load_count = []
    midtrip_dump_count = []
    midtrip_load_max = []
    midtrip_dump_max = []
    midtrip_load_mean = []
    midtrip_dump_mean = []
    midtrip_ge3_load = []
    midtrip_ge3_dump = []

    for _, row in df.iterrows():
        q_load.append(
            nearest_queue_depth(
                queue_df,
                row.get("load_zone", ""),
                row.get("arrive_load_s", np.nan),
                "ARRIVE_LOAD_ZONE",
            )
        )
        q_dump.append(
            nearest_queue_depth(
                queue_df,
                row.get("dump_zone", ""),
                row.get("arrive_dump_s", np.nan),
                "ARRIVE_DUMP_ZONE",
            )
        )
        load_ts = row.get("arrive_load_s", row.get("depart_time_s", np.nan))
        dump_ts = row.get("arrive_dump_s", row.get("depart_time_s", np.nan))
        load_stats = rolling_queue_stats(queue_df, row.get("load_zone", ""), load_ts)
        dump_stats = rolling_queue_stats(queue_df, row.get("dump_zone", ""), dump_ts)
        dispatch_ts = row.get("depart_time_s", np.nan)
        midtrip_ts = row.get("depart_load_s", row.get("load_complete_s", np.nan))
        dispatch_load_stats = rolling_queue_stats(queue_df, row.get("load_zone", ""), dispatch_ts)
        dispatch_dump_stats = rolling_queue_stats(queue_df, row.get("dump_zone", ""), dispatch_ts)
        midtrip_load_stats = rolling_queue_stats(queue_df, row.get("load_zone", ""), midtrip_ts)
        midtrip_dump_stats = rolling_queue_stats(queue_df, row.get("dump_zone", ""), midtrip_ts)
        q_recent_load.append(
            load_stats["count"]
        )
        q_recent_dump.append(
            dump_stats["count"]
        )
        q_recent_load_max.append(load_stats["max_queue"])
        q_recent_dump_max.append(dump_stats["max_queue"])
        q_recent_load_mean.append(load_stats["mean_queue"])
        q_recent_dump_mean.append(dump_stats["mean_queue"])
        q_ge3_load.append(load_stats["ge_3"])
        q_ge3_dump.append(dump_stats["ge_3"])
        dispatch_load_count.append(dispatch_load_stats["count"])
        dispatch_dump_count.append(dispatch_dump_stats["count"])
        dispatch_load_max.append(dispatch_load_stats["max_queue"])
        dispatch_dump_max.append(dispatch_dump_stats["max_queue"])
        dispatch_load_mean.append(dispatch_load_stats["mean_queue"])
        dispatch_dump_mean.append(dispatch_dump_stats["mean_queue"])
        dispatch_ge3_load.append(dispatch_load_stats["ge_3"])
        dispatch_ge3_dump.append(dispatch_dump_stats["ge_3"])
        midtrip_load_count.append(midtrip_load_stats["count"])
        midtrip_dump_count.append(midtrip_dump_stats["count"])
        midtrip_load_max.append(midtrip_load_stats["max_queue"])
        midtrip_dump_max.append(midtrip_dump_stats["max_queue"])
        midtrip_load_mean.append(midtrip_load_stats["mean_queue"])
        midtrip_dump_mean.append(midtrip_dump_stats["mean_queue"])
        midtrip_ge3_load.append(midtrip_load_stats["ge_3"])
        midtrip_ge3_dump.append(midtrip_dump_stats["ge_3"])

    df["queue_at_load"] = q_load
    df["queue_at_dump"] = q_dump
    df["queue_events_last120s_load"] = q_recent_load
    df["queue_events_last120s_dump"] = q_recent_dump
    df["max_queue_last120s_load"] = q_recent_load_max
    df["max_queue_last120s_dump"] = q_recent_dump_max
    df["mean_queue_last120s_load"] = q_recent_load_mean
    df["mean_queue_last120s_dump"] = q_recent_dump_mean
    df["queue_ge_3_load"] = q_ge3_load
    df["queue_ge_3_dump"] = q_ge3_dump
    df["dispatch_queue_events_last120s_load"] = dispatch_load_count
    df["dispatch_queue_events_last120s_dump"] = dispatch_dump_count
    df["dispatch_max_queue_last120s_load"] = dispatch_load_max
    df["dispatch_max_queue_last120s_dump"] = dispatch_dump_max
    df["dispatch_mean_queue_last120s_load"] = dispatch_load_mean
    df["dispatch_mean_queue_last120s_dump"] = dispatch_dump_mean
    df["dispatch_queue_ge_3_load"] = dispatch_ge3_load
    df["dispatch_queue_ge_3_dump"] = dispatch_ge3_dump
    df["midtrip_queue_events_last120s_load"] = midtrip_load_count
    df["midtrip_queue_events_last120s_dump"] = midtrip_dump_count
    df["midtrip_max_queue_last120s_load"] = midtrip_load_max
    df["midtrip_max_queue_last120s_dump"] = midtrip_dump_max
    df["midtrip_mean_queue_last120s_load"] = midtrip_load_mean
    df["midtrip_mean_queue_last120s_dump"] = midtrip_dump_mean
    df["midtrip_queue_ge_3_load"] = midtrip_ge3_load
    df["midtrip_queue_ge_3_dump"] = midtrip_ge3_dump
    return df


def merge_features(trips: pd.DataFrame, trips_graph: pd.DataFrame, queue_df: pd.DataFrame) -> pd.DataFrame:
    graph_cols = [
        "trip_id",
        "road_dist_to_mine_m",
        "road_dist_to_dump_m",
        "total_road_dist_m",
        "detour_factor",
        "avg_edge_weight",
    ]
    graph_available = [c for c in graph_cols if c in trips_graph.columns]
    merged = trips.merge(trips_graph[graph_available], on="trip_id", how="left")

    # Fill missing graph rows (happens when trips_graph is stale) with route medians first.
    for col in [c for c in graph_cols if c != "trip_id" and c in merged.columns]:
        route_med = merged.groupby(["load_zone", "dump_zone"])[col].transform("median")
        merged[col] = merged[col].fillna(route_med)
        merged[col] = merged[col].fillna(merged[col].median())

    merged = add_queue_features(merged, queue_df)
    return merged


def add_route_relative_features(df_fit: pd.DataFrame, df_apply: pd.DataFrame) -> pd.DataFrame:
    df_apply = df_apply.copy()
    route_keys = list(zip(df_apply["load_zone"], df_apply["dump_zone"]))

    empty_map = (
        df_fit.groupby(["load_zone", "dump_zone"])["travel_to_mine_s"].median().to_dict()
        if "travel_to_mine_s" in df_fit.columns
        else {}
    )
    loaded_map = (
        df_fit.groupby(["load_zone", "dump_zone"])["travel_to_dump_s"].median().to_dict()
        if "travel_to_dump_s" in df_fit.columns
        else {}
    )
    wait_map = (
        df_fit.groupby(["load_zone", "dump_zone"])["load_wait_s"].median().to_dict()
        if "load_wait_s" in df_fit.columns
        else {}
    )
    unload_map = (
        df_fit.groupby(["load_zone", "dump_zone"])["unload_wait_s"].median().to_dict()
        if "unload_wait_s" in df_fit.columns
        else {}
    )
    total_map = (
        df_fit.groupby(["load_zone", "dump_zone"])["total_trip_duration_s"].median().to_dict()
        if "total_trip_duration_s" in df_fit.columns
        else {}
    )

    empty_default = float(df_fit["travel_to_mine_s"].median()) if "travel_to_mine_s" in df_fit.columns else np.nan
    loaded_default = float(df_fit["travel_to_dump_s"].median()) if "travel_to_dump_s" in df_fit.columns else np.nan
    wait_default = float(df_fit["load_wait_s"].median()) if "load_wait_s" in df_fit.columns else np.nan
    unload_default = float(df_fit["unload_wait_s"].median()) if "unload_wait_s" in df_fit.columns else np.nan
    total_default = float(df_fit["total_trip_duration_s"].median()) if "total_trip_duration_s" in df_fit.columns else np.nan

    df_apply["expected_empty_time_by_route"] = [empty_map.get(k, empty_default) for k in route_keys]
    df_apply["expected_loaded_time_by_route"] = [loaded_map.get(k, loaded_default) for k in route_keys]
    df_apply["expected_load_wait_by_route"] = [wait_map.get(k, wait_default) for k in route_keys]
    df_apply["expected_unload_wait_by_route"] = [unload_map.get(k, unload_default) for k in route_keys]
    df_apply["expected_total_time_by_route"] = [total_map.get(k, total_default) for k in route_keys]

    df_apply["empty_delay_ratio"] = (
        df_apply["travel_to_mine_s"] / df_apply["expected_empty_time_by_route"].replace(0, np.nan)
    )
    df_apply["load_wait_ratio"] = (
        df_apply["load_wait_s"] / df_apply["expected_load_wait_by_route"].replace(0, np.nan)
    )
    df_apply["observed_elapsed_before_dump_s"] = (
        df_apply["travel_to_mine_s"].fillna(0) + df_apply["load_wait_s"].fillna(0)
    )
    df_apply["remaining_time_proxy_s"] = (
        df_apply["expected_total_time_by_route"] - df_apply["observed_elapsed_before_dump_s"]
    )
    df_apply["progress_ratio_to_expected"] = (
        df_apply["observed_elapsed_before_dump_s"]
        / df_apply["expected_total_time_by_route"].replace(0, np.nan)
    )

    if {"road_dist_to_mine_m", "travel_to_mine_s"}.issubset(df_apply.columns):
        df_apply["empty_speed_proxy_kmh"] = (
            3.6 * df_apply["road_dist_to_mine_m"] / df_apply["travel_to_mine_s"].replace(0, np.nan)
        )

    return df_apply


def make_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )


def get_splits(groups: pd.Series, n_rows: int):
    unique_groups = groups.nunique()
    if unique_groups >= 2:
        n_splits = min(5, unique_groups)
        return GroupKFold(n_splits=n_splits), "GroupKFold"
    return KFold(n_splits=min(5, n_rows), shuffle=True, random_state=42), "KFold"


def evaluate_variant(
    df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    variant_name: str,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    y = df["total_trip_duration_s"].values
    groups = df["run_id"].astype(str) if "run_id" in df.columns else pd.Series(["1"] * len(df))
    feature_cols = cat_cols + num_cols

    splitter, cv_name = get_splits(groups, len(df))
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    results = {}
    predictions_wide = pd.DataFrame({"trip_id": df["trip_id"].values, "actual_duration_s": y})

    for model_name, model in models.items():
        oof = np.zeros(len(df), dtype=float)
        used = np.zeros(len(df), dtype=bool)

        for train_idx, test_idx in splitter.split(df, y, groups):
            fold_train = add_route_relative_features(df.iloc[train_idx], df.iloc[train_idx])
            fold_test = add_route_relative_features(df.iloc[train_idx], df.iloc[test_idx])
            X_train = fold_train[feature_cols]
            X_test = fold_test[feature_cols]
            y_train = y[train_idx]

            pipe = Pipeline(
                steps=[
                    ("prep", make_preprocessor(cat_cols, num_cols)),
                    ("model", model),
                ]
            )
            pipe.fit(X_train, y_train)
            oof[test_idx] = pipe.predict(X_test)
            used[test_idx] = True

        mae = mean_absolute_error(y[used], oof[used])
        mape = mean_absolute_percentage_error(y[used], oof[used]) * 100.0
        r2 = r2_score(y[used], oof[used])

        # Refit on all data for feature importances and deployment output.
        full_df = add_route_relative_features(df, df)
        final_pipe = Pipeline(
            steps=[
                ("prep", make_preprocessor(cat_cols, num_cols)),
                ("model", model),
            ]
        )
        final_pipe.fit(full_df[feature_cols], y)
        full_pred = final_pipe.predict(full_df[feature_cols])

        results[model_name] = {
            "mae_cv": float(mae),
            "mape_cv": float(mape),
            "r2_cv": float(r2),
            "cv_type": cv_name,
            "model": final_pipe,
            "oof_pred": oof,
            "full_pred": full_pred,
        }
        predictions_wide[f"{variant_name}__{model_name}"] = full_pred
        print(f"  [{variant_name}] {model_name:<18} MAE={mae:.2f}s  MAPE={mape:.2f}%  R2={r2:.3f}")

    return results, predictions_wide


def save_summary(all_results: Dict[str, Dict[str, Dict[str, float]]], df: pd.DataFrame) -> None:
    path = os.path.join(OUT_DIR, "eta_v2_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("ETA v2 Summary (Grouped CV by run_id)\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"Trips used              : {len(df)}\n")
        f.write(f"Runs (groups)           : {df['run_id'].nunique() if 'run_id' in df.columns else 1}\n")
        f.write(f"Unique routes           : {df.groupby(['load_zone', 'dump_zone']).ngroups}\n")
        f.write(f"Avg trip duration (s)   : {df['total_trip_duration_s'].mean():.2f}\n\n")

        for variant, model_res in all_results.items():
            f.write(f"{variant}\n")
            f.write("-" * len(variant) + "\n")
            f.write(f"{'Model':<20} {'MAE(s)':>10} {'MAPE(%)':>10} {'R2':>8}\n")
            for model_name, res in model_res.items():
                f.write(
                    f"{model_name:<20} {res['mae_cv']:>10.2f} {res['mape_cv']:>10.2f} {res['r2_cv']:>8.3f}\n"
                )
            best = min(model_res, key=lambda m: model_res[m]["mae_cv"])
            f.write(f"Best: {best} (MAE={model_res[best]['mae_cv']:.2f}s)\n\n")

    print(f"Saved: {path}")


def save_predictions(
    df: pd.DataFrame,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    out = df[
        [
            "trip_id",
            "run_id",
            "session_key",
            "truck_id",
            "load_zone",
            "dump_zone",
            "total_trip_duration_s",
            "fuel_L",
            "co2_kg",
        ]
    ].copy()

    for variant, model_res in all_results.items():
        best = min(model_res, key=lambda m: model_res[m]["mae_cv"])
        pred = model_res[best]["full_pred"]
        out[f"pred_{variant}_s"] = np.round(pred, 2)
        out[f"err_{variant}_s"] = np.round(pred - out["total_trip_duration_s"].values, 2)

    path = os.path.join(OUT_DIR, "eta_v2_predictions.csv")
    out.to_csv(path, index=False)
    print(f"Saved: {path}")


def plot_variant_mae(all_results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    variants = list(all_results.keys())
    best_maes = []
    best_names = []
    for variant in variants:
        best = min(all_results[variant], key=lambda m: all_results[variant][m]["mae_cv"])
        best_names.append(best)
        best_maes.append(all_results[variant][best]["mae_cv"])

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(variants)))
    bars = ax.bar(variants, best_maes, color=colors, alpha=0.9)
    ax.set_title("ETA v2 Variant Comparison (Best model per variant)")
    ax.set_ylabel("Grouped CV MAE (seconds)")
    ax.grid(True, axis="y", alpha=0.3)

    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.6,
            f"{best_maes[i]:.1f}s\n{best_names[i]}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "eta_v2_variant_mae.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    print("=== ETA v2 (Grouped CV + Graph + Queue) ===")
    trips, trips_graph, queue = load_data()
    print(f"Trips rows      : {len(trips)}")
    print(f"Trips graph rows: {len(trips_graph)}")
    print(f"Queue rows      : {len(queue)}")

    df = merge_features(trips, trips_graph, queue)
    print(f"Merged rows     : {len(df)}")

    common_cat = ["load_zone", "dump_zone", "truck_id"]
    dispatch_common_num = [
        "cargo_kg",
        "trip_number",
        "expected_empty_time_by_route",
        "expected_loaded_time_by_route",
        "expected_load_wait_by_route",
        "expected_unload_wait_by_route",
        "expected_total_time_by_route",
    ]
    dispatch_graph_num = [
        "road_dist_to_mine_m",
        "road_dist_to_dump_m",
        "total_road_dist_m",
        "detour_factor",
        "avg_edge_weight",
    ]
    dispatch_queue_num = [
        "dispatch_queue_events_last120s_load",
        "dispatch_queue_events_last120s_dump",
        "dispatch_max_queue_last120s_load",
        "dispatch_max_queue_last120s_dump",
        "dispatch_mean_queue_last120s_load",
        "dispatch_mean_queue_last120s_dump",
        "dispatch_queue_ge_3_load",
        "dispatch_queue_ge_3_dump",
    ]
    midtrip_common_num = [
        "cargo_kg",
        "trip_number",
        "travel_to_mine_s",
        "load_wait_s",
        "expected_empty_time_by_route",
        "expected_loaded_time_by_route",
        "expected_load_wait_by_route",
        "expected_unload_wait_by_route",
        "expected_total_time_by_route",
        "observed_elapsed_before_dump_s",
        "remaining_time_proxy_s",
        "progress_ratio_to_expected",
        "empty_delay_ratio",
        "load_wait_ratio",
        "empty_speed_proxy_kmh",
    ]
    midtrip_graph_num = dispatch_graph_num
    midtrip_queue_num = [
        "queue_at_load",
        "midtrip_queue_events_last120s_load",
        "midtrip_queue_events_last120s_dump",
        "midtrip_max_queue_last120s_load",
        "midtrip_max_queue_last120s_dump",
        "midtrip_mean_queue_last120s_load",
        "midtrip_mean_queue_last120s_dump",
        "midtrip_queue_ge_3_load",
        "midtrip_queue_ge_3_dump",
    ]

    variants = {
        "dispatch_baseline": (common_cat, dispatch_common_num),
        "dispatch_graph": (common_cat, dispatch_common_num + dispatch_graph_num),
        "dispatch_queue": (common_cat, dispatch_common_num + dispatch_queue_num),
        "dispatch_full": (common_cat, dispatch_common_num + dispatch_graph_num + dispatch_queue_num),
        "midtrip_baseline": (common_cat, midtrip_common_num),
        "midtrip_graph": (common_cat, midtrip_common_num + midtrip_graph_num),
        "midtrip_queue": (common_cat, midtrip_common_num + midtrip_queue_num),
        "midtrip_full": (common_cat, midtrip_common_num + midtrip_graph_num + midtrip_queue_num),
    }

    all_results = {}
    for variant_name, (cat_cols, num_cols) in variants.items():
        print(f"\nTraining variant: {variant_name}")
        results, _ = evaluate_variant(df, cat_cols, num_cols, variant_name)
        all_results[variant_name] = results

    print("\nSaving outputs...")
    save_predictions(df, all_results)
    save_summary(all_results, df)
    plot_variant_mae(all_results)

    print("\nBest MAE by variant:")
    for variant, model_res in all_results.items():
        best = min(model_res, key=lambda m: model_res[m]["mae_cv"])
        mae = model_res[best]["mae_cv"]
        print(f"  {variant:<8} -> {best:<18} {mae:>6.2f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
