"""
eta_model.py  –  Baseline ETA predictor for HEMM fleet trips.

Reads:  analysis/trips.csv
Writes: analysis/eta_predictions.csv
        analysis/eta_model_summary.txt
        analysis/plots/  (diagnostic charts)

Run from Simulation/ folder:
    python eta_model.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

OUT_DIR  = "analysis"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

C_GREEN  = "#1A6B3A"
C_BLUE   = "#2E86AB"
C_ORANGE = "#F4A261"
C_RED    = "#E63946"
C_GRAY   = "#AAAAAA"

# ── load & clean ──────────────────────────────────────────────────────────────

def load_trips(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["total_trip_duration_s"])
    mean_dur = df["total_trip_duration_s"].mean()
    std_dur  = df["total_trip_duration_s"].std()
    df["is_outlier"] = df["total_trip_duration_s"] > (mean_dur + 2 * std_dur)
    print(f"Loaded {len(df)} trips  ({df['is_outlier'].sum()} flagged as queue outliers)")
    return df

# ── features ──────────────────────────────────────────────────────────────────

def build_features(df):
    le_load = LabelEncoder()
    le_dump = LabelEncoder()
    df = df.copy()
    df["load_zone_enc"] = le_load.fit_transform(df["load_zone"])
    df["dump_zone_enc"] = le_dump.fit_transform(df["dump_zone"])

    avg_travel = df["travel_to_mine_s"].median()
    df["travel_to_mine_s"] = df["travel_to_mine_s"].fillna(avg_travel)

    route_avg = df.groupby(["load_zone", "dump_zone"])["travel_to_mine_s"].transform("median")
    df["queue_proxy"] = df["travel_to_mine_s"] / route_avg.replace(0, 1)

    FEATURES = [
        "load_zone_enc", "dump_zone_enc", "cargo_kg",
        "travel_to_mine_s", "load_wait_s", "trip_number",
        "truck_id", "queue_proxy",
    ]
    TARGET = "total_trip_duration_s"

    train = df[~df["is_outlier"]]
    X_train = train[FEATURES].values
    y_train = train[TARGET].values
    X_all   = df[FEATURES].values
    y_all   = df[TARGET].values
    return X_train, y_train, X_all, y_all, FEATURES, df

# ── models ────────────────────────────────────────────────────────────────────

def train_models(X_train, y_train):
    models = {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }
    results = {}
    for name, model in models.items():
        loo = LeaveOneOut()
        cv  = cross_val_score(model, X_train, y_train, cv=loo, scoring="neg_mean_absolute_error")
        mae_cv = -cv.mean()
        model.fit(X_train, y_train)
        preds  = model.predict(X_train)
        r2     = r2_score(y_train, preds)
        results[name] = {"model": model, "mae_cv": round(mae_cv,2), "r2_train": round(r2,4)}
        print(f"  {name:<22}  LOO-MAE={mae_cv:.1f}s   train-R²={r2:.3f}")
    return results

# ── plots ─────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_true, y_pred, model_name, df_full):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"ETA Model: {model_name}", fontsize=14, fontweight="bold", color=C_GREEN)
    colours = [C_RED if o else C_BLUE for o in df_full["is_outlier"]]

    ax = axes[0]
    ax.scatter(y_true, y_pred, c=colours, alpha=0.8, edgecolors="white", s=80, zorder=3)
    lims = [min(y_true.min(), y_pred.min())-20, max(y_true.max(), y_pred.max())+20]
    ax.plot(lims, lims, "--", color=C_GRAY, lw=1.5)
    ax.set_xlabel("Actual (s)"); ax.set_ylabel("Predicted (s)"); ax.set_title("Actual vs Predicted")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.grid(True, alpha=0.3)
    ax.legend(handles=[mpatches.Patch(color=C_BLUE, label="Normal"),
                        mpatches.Patch(color=C_RED,  label="Outlier")], fontsize=9)

    ax2 = axes[1]
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, c=colours, alpha=0.8, edgecolors="white", s=80)
    ax2.axhline(0, color=C_GRAY, lw=1.5, linestyle="--")
    mae = mean_absolute_error(y_true, y_pred)
    ax2.set_xlabel("Predicted (s)"); ax2.set_ylabel("Residual (s)")
    ax2.set_title(f"Residuals  (MAE={mae:.1f}s)"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, f"eta_{model_name.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"): return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(imp)), imp[idx], color=C_GREEN, edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Importance"); ax.set_title(f"Feature Importance — {model_name}", fontweight="bold", color=C_GREEN)
    ax.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    fname = os.path.join(PLOT_DIR, f"importance_{model_name.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def plot_trip_duration_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Trip Duration Analysis", fontsize=13, fontweight="bold", color=C_BLUE)
    normal  = df[~df["is_outlier"]]["total_trip_duration_s"]
    outlier = df[ df["is_outlier"]]["total_trip_duration_s"]

    ax = axes[0]
    ax.hist(normal,  bins=12, color=C_BLUE, alpha=0.7, label="Normal", edgecolor="white")
    ax.hist(outlier, bins=5,  color=C_RED,  alpha=0.7, label="Outlier", edgecolor="white")
    ax.axvline(normal.mean(), color=C_GREEN, lw=2, linestyle="--", label=f"Mean={normal.mean():.0f}s")
    ax.set_xlabel("Trip Duration (s)"); ax.set_ylabel("Count"); ax.set_title("Distribution")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df.index, df["total_trip_duration_s"], "o-", color=C_BLUE, alpha=0.6, ms=5)
    ax2.scatter(df[df["is_outlier"]].index, df[df["is_outlier"]]["total_trip_duration_s"],
                color=C_RED, s=100, zorder=5, label="Outlier")
    ax2.set_xlabel("Trip index"); ax2.set_ylabel("Duration (s)"); ax2.set_title("Duration Over Time")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "trip_duration_analysis.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def plot_per_route(df):
    routes = df.groupby(["load_zone", "dump_zone"])["total_trip_duration_s"]
    labels = [f"{l.replace('load_zone_','M')}\n→{d.replace('dump_zone_','D')}" for (l,d),_ in routes]
    means  = [g.mean() for _,g in routes]
    stds   = [g.std()  for _,g in routes]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(labels)), means, yerr=stds, color=C_GREEN, alpha=0.8,
           edgecolor="white", capsize=5, error_kw={"ecolor": C_ORANGE})
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Avg Duration (s)"); ax.set_title("Avg Trip Duration by Route ± std", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "duration_by_route.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def plot_fuel_vs_duration(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = [C_RED if o else C_BLUE for o in df["is_outlier"]]
    ax.scatter(df["total_trip_duration_s"], df["fuel_L"], c=colours, s=80, alpha=0.8, edgecolors="white")
    ax.set_xlabel("Trip Duration (s)"); ax.set_ylabel("Fuel (L)")
    ax.set_title("Fuel vs Trip Duration", fontweight="bold", color=C_BLUE)
    ax.legend(handles=[mpatches.Patch(color=C_BLUE, label="Normal"),
                        mpatches.Patch(color=C_RED,  label="Outlier")])
    ax.grid(True, alpha=0.3); plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "fuel_vs_duration.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

# ── save ──────────────────────────────────────────────────────────────────────

def save_predictions(df, best_model, X_all):
    preds = best_model.predict(X_all)
    out = df[["trip_id","truck_id","load_zone","dump_zone",
              "total_trip_duration_s","fuel_L","co2_kg"]].copy()
    out["predicted_duration_s"] = preds.round(1)
    out["error_s"]   = (preds - df["total_trip_duration_s"]).round(1)
    out["pct_error"] = ((out["error_s"] / df["total_trip_duration_s"]) * 100).round(1)
    out["is_outlier"] = df["is_outlier"]
    path = os.path.join(OUT_DIR, "eta_predictions.csv")
    out.to_csv(path, index=False)
    print(f"  Saved: {path}")

def save_summary(model_results, df):
    normal = df[~df["is_outlier"]]
    path = os.path.join(OUT_DIR, "eta_model_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("═══════════════════════════════════════════\n")
        f.write("  HEMM Fleet ETA Model Summary\n")
        f.write("═══════════════════════════════════════════\n\n")
        f.write(f"  Total trips     : {len(df)}\n")
        f.write(f"  Normal trips    : {len(normal)}\n")
        f.write(f"  Outliers        : {df['is_outlier'].sum()} (queue-delayed)\n")
        f.write(f"  Unique routes   : {df.groupby(['load_zone','dump_zone']).ngroups}\n")
        f.write(f"  Avg duration    : {normal['total_trip_duration_s'].mean():.1f} s\n")
        f.write(f"  Avg fuel/trip   : {normal['fuel_L'].mean():.3f} L\n\n")
        f.write(f"  {'Model':<25} {'LOO-MAE':>10} {'Train-R²':>10}\n")
        f.write(f"  {'─'*25} {'─'*10} {'─'*10}\n")
        for name, res in model_results.items():
            f.write(f"  {name:<25} {res['mae_cv']:>10.1f} {res['r2_train']:>10.4f}\n")
        best = min(model_results, key=lambda n: model_results[n]["mae_cv"])
        f.write(f"\n  Best model : {best}\n")
        f.write(f"  LOO-MAE    : {model_results[best]['mae_cv']:.1f}s  "
                f"({100*model_results[best]['mae_cv']/normal['total_trip_duration_s'].mean():.1f}% of avg trip)\n\n")
        f.write("  Notes\n")
        f.write("  - 29 trips is small; collect 100+ for reliable generalisation.\n")
        f.write("  - Outlier trips (queue-delayed) excluded from training.\n")
        f.write("  - Add graph-distance features next for Semester 2 improvements.\n")
    print(f"  Saved: {path}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== HEMM ETA Model ===\n")
    df = load_trips(os.path.join(OUT_DIR, "trips.csv"))

    print("\nBuilding features...")
    X_train, y_train, X_all, y_all, features, df_fe = build_features(df)
    print(f"Training: {len(X_train)} trips  |  Features: {features}\n")

    print("Training models...")
    results = train_models(X_train, y_train)

    best_name  = min(results, key=lambda n: results[n]["mae_cv"])
    best_model = results[best_name]["model"]
    print(f"\nBest: {best_name}  LOO-MAE={results[best_name]['mae_cv']:.1f}s\n")

    print("Generating plots...")
    plot_trip_duration_distribution(df_fe)
    plot_per_route(df_fe)
    plot_fuel_vs_duration(df_fe)
    for name, res in results.items():
        preds = res["model"].predict(X_all)
        plot_actual_vs_predicted(y_all, preds, name, df_fe)
        plot_feature_importance(res["model"], features, name)

    print("\nSaving results...")
    save_predictions(df_fe, best_model, X_all)
    save_summary(results, df_fe)

    avg = df_fe[~df_fe["is_outlier"]]["total_trip_duration_s"].mean()
    print(f"\n─── Done ───────────────────────────────────────")
    print(f"  Best model   : {best_name}")
    print(f"  LOO-MAE      : {results[best_name]['mae_cv']:.1f}s  ({100*results[best_name]['mae_cv']/avg:.1f}% of avg trip)")
    print(f"  Plots        : {PLOT_DIR}/")
    print(f"  Results      : {OUT_DIR}/")
    print(f"────────────────────────────────────────────────")

if __name__ == "__main__":
    main()
