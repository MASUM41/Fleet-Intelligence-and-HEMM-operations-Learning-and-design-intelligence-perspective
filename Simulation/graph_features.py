"""
graph_features.py  –  Graph-aware feature extraction for HEMM ETA model.

Computes shortest-path road distances between all load/dump zone pairs
using Dijkstra on the actual road graph from map_cache.pkl.

Adds these features to trips.csv and retrains the ETA model:
    road_dist_to_mine_m     – actual road distance from last dump to mine
    road_dist_to_dump_m     – actual road distance from mine to dump
    total_road_dist_m       – full trip road distance
    dist_vs_euclidean       – ratio of road to straight-line (detour factor)
    avg_edge_weight         – mean edge weight on shortest path (proxy for road quality)

Run from Simulation/ folder:
    python graph_features.py
"""

import os
import pickle
import heapq
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

# ══════════════════════════════════════════════════════════════════════════════
#  1. DIJKSTRA ON ROAD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def dijkstra(graph, source):
    """
    Standard Dijkstra from source node.
    Returns: dist dict {node: shortest_distance}, prev dict {node: parent}
    """
    dist = {node: float("inf") for node in graph}
    dist[source] = 0.0
    prev = {node: None for node in graph}
    pq   = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            nd = dist[u] + float(w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def reconstruct_path(prev, source, target):
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    if path and path[0] == source:
        return path
    return []


def build_distance_matrix(graph, zones):
    """
    Run Dijkstra from every zone and build a distance + path matrix.
    Returns:
        dist_matrix  : {source: {target: distance}}
        path_matrix  : {source: {target: [node list]}}
    """
    dist_matrix = {}
    path_matrix = {}

    for zone in zones:
        if zone not in graph:
            print(f"  Warning: {zone} not in graph — skipping")
            continue
        dist, prev = dijkstra(graph, zone)
        dist_matrix[zone] = dist
        path_matrix[zone] = {t: reconstruct_path(prev, zone, t) for t in zones}

    return dist_matrix, path_matrix


def path_avg_edge_weight(graph, path):
    """Mean edge weight along a path (proxy for road difficulty)."""
    if len(path) < 2:
        return 0.0
    weights = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        for neighbor, w in graph.get(u, []):
            if neighbor == v:
                weights.append(float(w))
                break
    return np.mean(weights) if weights else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  2. NODE POSITIONS (for euclidean comparison)
# ══════════════════════════════════════════════════════════════════════════════

def load_node_positions():
    """Load NODES dict from map_loader."""
    try:
        import sys
        sys.path.insert(0, ".")
        from Map import map_loader as map_data
        return {k: np.array(v) for k, v in map_data.NODES.items()}
    except Exception as e:
        print(f"  Warning: could not load node positions ({e}). Euclidean ratios unavailable.")
        return {}


def euclidean_dist(nodes, a, b):
    if a in nodes and b in nodes:
        return float(np.linalg.norm(nodes[a] - nodes[b]))
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  3. ENRICH TRIPS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def enrich_trips(df, dist_matrix, path_matrix, graph, nodes):
    df = df.copy()

    road_dist_to_mine  = []
    road_dist_to_dump  = []
    total_road_dist    = []
    detour_factor      = []
    avg_edge_wt        = []

    for _, row in df.iterrows():
        lz = row["load_zone"]
        dz = row["dump_zone"]

        # Road distance dump → mine  (empty leg)
        d_to_mine = dist_matrix.get(dz, {}).get(lz, np.nan)

        # Road distance mine → dump  (loaded leg)
        d_to_dump = dist_matrix.get(lz, {}).get(dz, np.nan)

        total = (d_to_mine if not np.isnan(d_to_mine) else 0) + \
                (d_to_dump if not np.isnan(d_to_dump) else 0)

        # Detour factor: road / euclidean
        euclid = euclidean_dist(nodes, lz, dz)
        if euclid and euclid > 0 and not np.isnan(d_to_dump):
            detour = d_to_dump / euclid
        else:
            detour = np.nan

        # Avg edge weight on loaded leg path
        path = path_matrix.get(lz, {}).get(dz, [])
        avg_w = path_avg_edge_weight(graph, path)

        road_dist_to_mine.append(round(d_to_mine, 2) if not np.isnan(d_to_mine) else np.nan)
        road_dist_to_dump.append(round(d_to_dump, 2) if not np.isnan(d_to_dump) else np.nan)
        total_road_dist.append(round(total, 2))
        detour_factor.append(round(detour, 4) if not np.isnan(detour) else np.nan)
        avg_edge_wt.append(round(avg_w, 2))

    df["road_dist_to_mine_m"] = road_dist_to_mine
    df["road_dist_to_dump_m"] = road_dist_to_dump
    df["total_road_dist_m"]   = total_road_dist
    df["detour_factor"]       = detour_factor
    df["avg_edge_weight"]     = avg_edge_wt

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  4. RETRAIN ETA MODEL WITH GRAPH FEATURES
# ══════════════════════════════════════════════════════════════════════════════

FEATURES_BASE = [
    "load_zone_enc", "dump_zone_enc", "cargo_kg",
    "travel_to_mine_s", "load_wait_s", "trip_number",
    "truck_id", "queue_proxy",
]

FEATURES_GRAPH = [
    "load_zone_enc", "dump_zone_enc", "cargo_kg",
    "road_dist_to_mine_m", "road_dist_to_dump_m", "total_road_dist_m",
    "detour_factor", "avg_edge_weight",
    "load_wait_s", "trip_number", "truck_id", "queue_proxy",
]


def prepare_xy(df, features):
    df = df.copy()

    le_load = LabelEncoder()
    le_dump = LabelEncoder()
    df["load_zone_enc"] = le_load.fit_transform(df["load_zone"])
    df["dump_zone_enc"] = le_dump.fit_transform(df["dump_zone"])

    avg_travel = df["travel_to_mine_s"].median()
    df["travel_to_mine_s"] = df["travel_to_mine_s"].fillna(avg_travel)

    route_avg = df.groupby(["load_zone","dump_zone"])["travel_to_mine_s"].transform("median")
    df["queue_proxy"] = df["travel_to_mine_s"] / route_avg.replace(0, 1)

    mean_dur = df["total_trip_duration_s"].mean()
    std_dur  = df["total_trip_duration_s"].std()
    df["is_outlier"] = df["total_trip_duration_s"] > (mean_dur + 2 * std_dur)

    # Fill any NaN graph features with median
    for f in features:
        if f in df.columns and df[f].isna().any():
            df[f] = df[f].fillna(df[f].median())

    train = df[~df["is_outlier"]]
    avail = [f for f in features if f in df.columns]

    X_train = train[avail].values
    y_train = train["total_trip_duration_s"].values
    X_all   = df[avail].values
    y_all   = df["total_trip_duration_s"].values

    return X_train, y_train, X_all, y_all, avail, df


def run_model(X_train, y_train, label):
    models = {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }
    results = {}
    print(f"\n  [{label}]")
    for name, model in models.items():
        loo = LeaveOneOut()
        cv  = cross_val_score(model, X_train, y_train, cv=loo,
                              scoring="neg_mean_absolute_error")
        mae_cv = -cv.mean()
        model.fit(X_train, y_train)
        r2 = r2_score(y_train, model.predict(X_train))
        results[name] = {"model": model, "mae_cv": round(mae_cv, 2), "r2": round(r2, 4)}
        print(f"    {name:<22}  LOO-MAE={mae_cv:.1f}s   R²={r2:.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  5. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(base_results, graph_results):
    """Bar chart comparing LOO-MAE: baseline vs graph-aware."""
    model_names = list(base_results.keys())
    base_maes  = [base_results[n]["mae_cv"]  for n in model_names]
    graph_maes = [graph_results[n]["mae_cv"] for n in model_names]

    x   = np.arange(len(model_names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, base_maes,  w, label="Baseline (time features)", color=C_BLUE,  alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, graph_maes, w, label="Graph-aware features",      color=C_GREEN, alpha=0.85, edgecolor="white")

    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("LOO Cross-val MAE (seconds)", fontsize=11)
    ax.set_title("Baseline vs Graph-Aware ETA Model\nLower is better", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)

    for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                            f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=9, color=C_BLUE)
    for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                            f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=9, color=C_GREEN)

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "baseline_vs_graph.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def plot_road_dist_vs_duration(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Road Distance vs Trip Duration", fontsize=13, fontweight="bold", color=C_GREEN)
    colours = [C_RED if o else C_BLUE for o in df["is_outlier"]]

    ax = axes[0]
    ax.scatter(df["road_dist_to_dump_m"], df["total_trip_duration_s"],
               c=colours, s=80, alpha=0.8, edgecolors="white")
    ax.set_xlabel("Road Distance Mine→Dump (m)"); ax.set_ylabel("Trip Duration (s)")
    ax.set_title("Loaded Leg Distance vs Duration"); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(df["total_road_dist_m"], df["total_trip_duration_s"],
                c=colours, s=80, alpha=0.8, edgecolors="white")
    ax2.set_xlabel("Total Round-Trip Road Distance (m)"); ax2.set_ylabel("Trip Duration (s)")
    ax2.set_title("Total Road Distance vs Duration"); ax2.grid(True, alpha=0.3)
    ax2.legend(handles=[mpatches.Patch(color=C_BLUE, label="Normal"),
                         mpatches.Patch(color=C_RED,  label="Outlier")])

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "road_dist_vs_duration.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def plot_detour_factor(df):
    normal = df[~df["is_outlier"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    routes = normal.groupby(["load_zone","dump_zone"])["detour_factor"].mean()
    labels = [f"{l.replace('load_zone_','M')}→{d.replace('dump_zone_','D')}" for l,d in routes.index]
    ax.bar(range(len(routes)), routes.values, color=C_ORANGE, edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(routes))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.axhline(1.0, color=C_GRAY, lw=1.5, linestyle="--", label="Euclidean = 1.0")
    ax.set_ylabel("Detour Factor (road / euclidean)"); ax.set_title("Route Detour Factors", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "detour_factor.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def plot_feature_importance(model, feature_names, label):
    if not hasattr(model, "feature_importances_"): return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    colours = [C_GREEN if "road" in feature_names[i] or "detour" in feature_names[i]
               or "edge" in feature_names[i] else C_BLUE for i in idx]
    ax.bar(range(len(imp)), imp[idx], color=colours, edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Importance"); ax.set_title(f"Feature Importance — {label}", fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=C_GREEN, label="Graph feature"),
                        mpatches.Patch(color=C_BLUE,  label="Other feature")], fontsize=9)
    ax.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    fname = os.path.join(PLOT_DIR, f"importance_graph_{label.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== Graph-Aware ETA Feature Extraction ===\n")

    # Load road graph
    print("Loading road graph from Map/map_cache.pkl...")
    with open("Map/map_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    graph = cache["road_graph"]
    print(f"  Graph nodes: {len(graph)}")

    # Load node positions
    print("Loading node positions...")
    nodes = load_node_positions()
    print(f"  Nodes loaded: {len(nodes)}")

    # Load trips
    trips_path = os.path.join(OUT_DIR, "trips.csv")
    df = pd.read_csv(trips_path)
    print(f"  Trips loaded: {len(df)}")

    # Get all unique zones in the trips data
    all_zones = list(set(df["load_zone"].tolist() + df["dump_zone"].tolist()))
    print(f"\nComputing Dijkstra shortest paths for {len(all_zones)} zones...")
    dist_matrix, path_matrix = build_distance_matrix(graph, all_zones)

    # Print distance matrix
    print("\nRoad Distance Matrix (metres):")
    print(f"  {'':25}", end="")
    for dz in sorted(set(df["dump_zone"])): print(f"  {dz:>20}", end="")
    print()
    for lz in sorted(set(df["load_zone"])):
        print(f"  {lz:25}", end="")
        for dz in sorted(set(df["dump_zone"])):
            d = dist_matrix.get(lz, {}).get(dz, float("inf"))
            print(f"  {d:>20.1f}", end="")
        print()

    # Enrich trips table
    print("\nAdding graph features to trips...")
    df_enriched = enrich_trips(df, dist_matrix, path_matrix, graph, nodes)

    enriched_path = os.path.join(OUT_DIR, "trips_graph.csv")
    df_enriched.to_csv(enriched_path, index=False)
    print(f"  Saved: {enriched_path}")

    # Show sample of new features
    print("\nSample graph features (first 5 trips):")
    cols = ["trip_id","load_zone","dump_zone","road_dist_to_mine_m",
            "road_dist_to_dump_m","total_road_dist_m","detour_factor"]
    print(df_enriched[cols].head().to_string(index=False))

    # ── Baseline model (time features) ──
    print("\nTraining models...")
    X_b, y_b, Xa_b, ya_b, feats_b, df_b = prepare_xy(df,          FEATURES_BASE)
    X_g, y_g, Xa_g, ya_g, feats_g, df_g = prepare_xy(df_enriched, FEATURES_GRAPH)

    base_results  = run_model(X_b, y_b, "Baseline — time features")
    graph_results = run_model(X_g, y_g, "Graph-aware features")

    # ── Compare ──
    print("\n")
    print(f"  {'Model':<22}  {'Baseline MAE':>14}  {'Graph MAE':>10}  {'Improvement':>12}")
    # Use ASCII-only separators to avoid Windows cp1252 UnicodeEncodeError.
    print(f"  {'-'*22}  {'-'*14}  {'-'*10}  {'-'*12}")
    for name in base_results:
        b_mae = base_results[name]["mae_cv"]
        g_mae = graph_results[name]["mae_cv"]
        imp   = b_mae - g_mae
        sign  = "V" if imp > 0 else "^"
        print(f"  {name:<22}  {b_mae:>14.1f}s  {g_mae:>10.1f}s  {sign}{abs(imp):>10.1f}s")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_comparison(base_results, graph_results)
    plot_road_dist_vs_duration(df_g)
    plot_detour_factor(df_g)

    best_name  = min(graph_results, key=lambda n: graph_results[n]["mae_cv"])
    best_model = graph_results[best_name]["model"]
    plot_feature_importance(best_model, feats_g, best_name)

    # ── Summary ──
    best_base  = min(base_results,  key=lambda n: base_results[n]["mae_cv"])
    best_graph = min(graph_results, key=lambda n: graph_results[n]["mae_cv"])
    b_mae = base_results[best_base]["mae_cv"]
    g_mae = graph_results[best_graph]["mae_cv"]
    avg   = df_g[~df_g["is_outlier"]]["total_trip_duration_s"].mean()

    print(f"\n{'='*50}")
    print(f"  Baseline best  : {best_base:<22}  MAE={b_mae:.1f}s  ({100*b_mae/avg:.1f}%)")
    print(f"  Graph-aware    : {best_graph:<22}  MAE={g_mae:.1f}s  ({100*g_mae/avg:.1f}%)")
    improvement = b_mae - g_mae
    print(f"  Improvement    : {improvement:+.1f}s  ({'better' if improvement>0 else 'worse - need more data'})")
    print(f"  Plots saved to : {PLOT_DIR}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
