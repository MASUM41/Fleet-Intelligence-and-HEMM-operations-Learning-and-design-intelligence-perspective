"""
gnn_eta.py  -  Graph Neural Network for HEMM Fleet ETA Prediction.

Architecture: GraphSAGE (2-layer) + trip-level MLP head
- Node features: position (x,y), node type (load/dump/connector/hub),
  degree, avg edge weight
- Edge features: road distance
- Trip encoding: source node + destination node embeddings → MLP → ETA

Reads:  Map/map_cache.pkl        (road graph)
        Map/waypoints.pkl        (for node positions)  
        analysis/trips_graph.csv (trip labels)

Writes: analysis/gnn_eta_results.csv
        analysis/plots/gnn_*.png
        analysis/gnn_model.pt    (saved model weights)

Run from Simulation/ folder:
    python gnn_eta.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool

sys.path.insert(0, ".")
from Map import map_loader as map_data

OUT_DIR  = "analysis"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

C_GREEN  = "#1A6B3A"
C_BLUE   = "#2E86AB"
C_ORANGE = "#F4A261"
C_RED    = "#E63946"
C_GRAY   = "#AAAAAA"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. BUILD GRAPH DATA OBJECT
# ══════════════════════════════════════════════════════════════════════════════

def node_type_encoding(name):
    """One-hot: [is_load, is_dump, is_hub, is_connector, is_haul, is_parking]"""
    if "load_zone"  in name: return [1,0,0,0,0,0]
    if "dump_zone"  in name: return [0,1,0,0,0,0]
    if "hub"        in name: return [0,0,1,0,0,0]
    if "connector"  in name: return [0,0,0,1,0,0]
    if "haul"       in name: return [0,0,0,0,1,0]
    if "parking"    in name: return [0,0,0,0,0,1]
    return [0,0,0,0,0,0]


def build_pyg_graph(road_graph, node_positions):
    """
    Build a PyTorch Geometric Data object from the road graph.

    Node features (per node):
        x, y position (normalised)
        node type one-hot (6 dims)
        out-degree
        avg outgoing edge weight (normalised)
        in-degree
    Total: 2 + 6 + 3 = 11 features per node
    """
    # Create consistent node index mapping
    all_nodes = sorted(road_graph.keys())
    node_idx  = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    # ── Node features ──
    positions = np.array([
        node_positions.get(n, np.array([0.0, 0.0])) for n in all_nodes
    ], dtype=np.float32)

    # Normalise positions
    pos_mean = positions.mean(axis=0)
    pos_std  = positions.std(axis=0) + 1e-8
    pos_norm = (positions - pos_mean) / pos_std

    type_enc = np.array([node_type_encoding(n) for n in all_nodes], dtype=np.float32)

    out_degrees   = np.array([len(road_graph[n]) for n in all_nodes], dtype=np.float32)
    avg_edge_wts  = np.array([
        np.mean([float(w) for _, w in road_graph[n]]) if road_graph[n] else 0.0
        for n in all_nodes
    ], dtype=np.float32)

    # Compute in-degrees
    in_deg = np.zeros(N, dtype=np.float32)
    for n in all_nodes:
        for neighbor, _ in road_graph[n]:
            if neighbor in node_idx:
                in_deg[node_idx[neighbor]] += 1

    # Normalise scalar features
    out_deg_norm  = (out_degrees  - out_degrees.mean())  / (out_degrees.std()  + 1e-8)
    avg_wt_norm   = (avg_edge_wts - avg_edge_wts.mean()) / (avg_edge_wts.std() + 1e-8)
    in_deg_norm   = (in_deg       - in_deg.mean())       / (in_deg.std()       + 1e-8)

    node_feats = np.concatenate([
        pos_norm,
        type_enc,
        out_deg_norm.reshape(-1, 1),
        avg_wt_norm.reshape(-1, 1),
        in_deg_norm.reshape(-1, 1),
    ], axis=1)   # shape: (N, 11)

    # ── Edges ──
    edge_src, edge_dst, edge_wts = [], [], []
    for n in all_nodes:
        for neighbor, w in road_graph[n]:
            if neighbor in node_idx:
                edge_src.append(node_idx[n])
                edge_dst.append(node_idx[neighbor])
                edge_wts.append(float(w))

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_wts, dtype=torch.float32).unsqueeze(1)

    # Normalise edge weights
    ea_mean = edge_attr.mean(); ea_std = edge_attr.std() + 1e-8
    edge_attr = (edge_attr - ea_mean) / ea_std

    x = torch.tensor(node_feats, dtype=torch.float32)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    print(f"Graph: {N} nodes, {len(edge_src)} edges, {x.shape[1]} node features")
    return graph_data, node_idx, pos_mean, pos_std


# ══════════════════════════════════════════════════════════════════════════════
#  2. GNN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GraphSAGE_ETA(nn.Module):
    """
    Two-layer GraphSAGE encoder + trip-level MLP decoder.

    Forward pass:
        1. GraphSAGE encodes all nodes → node embeddings
        2. For each trip: concat(src_emb, dst_emb, trip_features) → MLP → ETA

    trip_features: [cargo_kg_norm, load_wait_s_norm, trip_number_norm, queue_proxy_norm]
    """
    def __init__(self, in_channels, hidden_channels, out_channels, trip_feat_dim, mlp_hidden):
        super().__init__()
        self.conv1 = SAGEConv(in_channels,      hidden_channels)
        self.conv2 = SAGEConv(hidden_channels,  out_channels)
        self.bn1   = nn.BatchNorm1d(hidden_channels)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)

        # MLP decoder: 2 node embeddings + trip features → ETA
        mlp_in = out_channels * 2 + trip_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in,     mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        return x   # (N, out_channels)

    def forward(self, graph_data, src_indices, dst_indices, trip_feats):
        node_emb  = self.encode(graph_data.x, graph_data.edge_index)
        src_emb   = node_emb[src_indices]   # (B, out_channels)
        dst_emb   = node_emb[dst_indices]   # (B, out_channels)
        combined  = torch.cat([src_emb, dst_emb, trip_feats], dim=1)
        eta       = self.mlp(combined).squeeze(1)
        return eta


# ══════════════════════════════════════════════════════════════════════════════
#  3. PREPARE TRIP DATASET
# ══════════════════════════════════════════════════════════════════════════════

def prepare_trip_dataset(trips_df, node_idx):
    """
    Returns tensors for training:
        src_indices  : dump_zone node index for each trip (where truck departs from)
        dst_indices  : load_zone node index for each trip (destination mine)
        trip_feats   : (cargo, load_wait, trip_number, queue_proxy) normalised
        targets      : total_trip_duration_s normalised
    """
    df = trips_df.copy()

    # Filter outliers
    mean_d = df["total_trip_duration_s"].mean()
    std_d  = df["total_trip_duration_s"].std()
    df     = df[df["total_trip_duration_s"] <= mean_d + 2 * std_d].copy()
    print(f"  Trip dataset: {len(df)} trips after outlier removal")

    # Fill missing
    df["travel_to_mine_s"] = df["travel_to_mine_s"].fillna(df["travel_to_mine_s"].median())
    route_avg = df.groupby(["load_zone","dump_zone"])["travel_to_mine_s"].transform("median")
    df["queue_proxy"] = df["travel_to_mine_s"] / route_avg.replace(0, 1)

    # Node indices
    src_indices, dst_indices = [], []
    valid_mask = []
    for _, row in df.iterrows():
        lz = row["load_zone"]
        dz = row["dump_zone"]
        if lz in node_idx and dz in node_idx:
            src_indices.append(node_idx[dz])   # departs from dump zone
            dst_indices.append(node_idx[lz])   # heads to load zone
            valid_mask.append(True)
        else:
            valid_mask.append(False)
            print(f"  Warning: {lz} or {dz} not in graph — skipping")

    df = df[valid_mask].reset_index(drop=True)

    # Trip features
    trip_feat_cols = ["cargo_kg", "load_wait_s", "trip_number", "queue_proxy"]
    trip_feats = df[trip_feat_cols].values.astype(np.float32)

    # Normalise trip features
    scaler = StandardScaler()
    trip_feats = scaler.fit_transform(trip_feats)

    # Target
    targets = df["total_trip_duration_s"].values.astype(np.float32)
    target_mean = targets.mean()
    target_std  = targets.std() + 1e-8
    targets_norm = (targets - target_mean) / target_std

    return (
        torch.tensor(src_indices, dtype=torch.long),
        torch.tensor(dst_indices, dtype=torch.long),
        torch.tensor(trip_feats,  dtype=torch.float32),
        torch.tensor(targets_norm, dtype=torch.float32),
        targets,        # raw for evaluation
        target_mean,
        target_std,
        scaler,
        df,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  4. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_gnn(model, graph_data, src_idx, dst_idx, trip_feats, targets,
              n_epochs=300, lr=0.001, patience=40):
    model = model.to(DEVICE)
    graph_data = graph_data.to(DEVICE)
    src_idx    = src_idx.to(DEVICE)
    dst_idx    = dst_idx.to(DEVICE)
    trip_feats = trip_feats.to(DEVICE)
    targets    = targets.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, verbose=False
    )

    # Leave-one-out style: train on all, track loss
    # With 27 samples, full-batch training is correct
    losses = []
    best_loss = float("inf")
    best_state = None
    no_improve = 0

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = model(graph_data, src_idx, dst_idx, trip_feats)
        loss  = F.mse_loss(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}  Loss={loss.item():.4f}  LR={optimizer.param_groups[0]['lr']:.5f}")

    # Restore best
    model.load_state_dict(best_state)
    return model, losses


# ══════════════════════════════════════════════════════════════════════════════
#  5. EVALUATION  (Leave-One-Out)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_loo(graph_data, src_idx, dst_idx, trip_feats, targets_norm,
                 targets_raw, target_mean, target_std,
                 in_channels, hidden, out_ch, trip_dim, mlp_hidden):
    """LOO cross-validation for fair comparison with sklearn baselines."""
    n = len(targets_raw)
    preds_raw = np.zeros(n)

    print(f"\n  Running LOO-CV ({n} folds)...")
    for i in range(n):
        # Train mask
        train_mask = [j for j in range(n) if j != i]
        test_mask  = [i]

        model_loo = GraphSAGE_ETA(in_channels, hidden, out_ch, trip_dim, mlp_hidden).to(DEVICE)
        optimizer = torch.optim.Adam(model_loo.parameters(), lr=0.001, weight_decay=1e-4)

        gd  = graph_data.to(DEVICE)
        s_t = src_idx[train_mask].to(DEVICE)
        d_t = dst_idx[train_mask].to(DEVICE)
        f_t = trip_feats[train_mask].to(DEVICE)
        y_t = targets_norm[train_mask].to(DEVICE)

        model_loo.train()
        for ep in range(200):
            optimizer.zero_grad()
            p = model_loo(gd, s_t, d_t, f_t)
            F.mse_loss(p, y_t).backward()
            optimizer.step()

        model_loo.eval()
        with torch.no_grad():
            s_e = src_idx[test_mask].to(DEVICE)
            d_e = dst_idx[test_mask].to(DEVICE)
            f_e = trip_feats[test_mask].to(DEVICE)
            pred_norm = model_loo(gd, s_e, d_e, f_e).cpu().numpy()
            preds_raw[i] = pred_norm[0] * target_std + target_mean

        if (i + 1) % 5 == 0:
            print(f"    Fold {i+1}/{n} done")

    mae = mean_absolute_error(targets_raw, preds_raw)
    r2  = r2_score(targets_raw, preds_raw)
    return preds_raw, mae, r2


# ══════════════════════════════════════════════════════════════════════════════
#  6. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_loss(losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color=C_BLUE, lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (normalised)")
    ax.set_title("GNN Training Loss", fontweight="bold", color=C_BLUE)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "gnn_training_loss.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def plot_actual_vs_predicted(targets_raw, preds_loo, preds_full):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GNN ETA Model — Actual vs Predicted", fontsize=13, fontweight="bold", color=C_GREEN)

    for ax, preds, title in zip(axes,
                                 [preds_loo,  preds_full],
                                 ["LOO Cross-Val", "Full Model (train set)"]):
        ax.scatter(targets_raw, preds, color=C_BLUE, alpha=0.8, edgecolors="white", s=80)
        lims = [min(targets_raw.min(), preds.min())-10,
                max(targets_raw.max(), preds.max())+10]
        ax.plot(lims, lims, "--", color=C_GRAY, lw=1.5)
        mae = mean_absolute_error(targets_raw, preds)
        r2  = r2_score(targets_raw, preds)
        ax.set_xlabel("Actual (s)"); ax.set_ylabel("Predicted (s)")
        ax.set_title(f"{title}\nMAE={mae:.1f}s  R²={r2:.3f}")
        ax.set_xlim(lims); ax.set_ylim(lims); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "gnn_actual_vs_predicted.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def plot_model_comparison(gnn_mae, gbt_mae, rf_mae, lr_mae, avg_dur):
    models = ["Linear\nRegression", "Random\nForest", "Gradient\nBoosting", "GraphSAGE\n(GNN)"]
    maes   = [lr_mae, rf_mae, gbt_mae, gnn_mae]
    colours= [C_GRAY, C_ORANGE, C_BLUE, C_GREEN]
    edge   = ["white","white","white","gold"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, maes, color=colours, edgecolor=edge, linewidth=2, alpha=0.85)
    ax.axhline(avg_dur * 0.05, color=C_RED, lw=1.5, linestyle="--",
               label="5% of avg trip duration")
    ax.set_ylabel("LOO Cross-Val MAE (seconds)", fontsize=11)
    ax.set_title("ETA Model Comparison\nBaseline vs Graph-Aware vs GNN",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

    for bar, mae in zip(bars, maes):
        pct = 100 * mae / avg_dur
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{mae:.1f}s\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "gnn_model_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def plot_node_embeddings(model, graph_data, node_idx):
    """t-SNE of learned node embeddings coloured by node type."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return

    model.eval()
    graph_data = graph_data.to(DEVICE)
    with torch.no_grad():
        emb = model.encode(graph_data.x, graph_data.edge_index).cpu().numpy()

    # Colour by node type
    idx_to_name = {v: k for k, v in node_idx.items()}
    type_colors = []
    for i in range(len(node_idx)):
        n = idx_to_name.get(i, "")
        if   "load_zone" in n: type_colors.append(C_GREEN)
        elif "dump_zone" in n: type_colors.append(C_RED)
        elif "hub"       in n: type_colors.append(C_ORANGE)
        elif "connector" in n: type_colors.append(C_BLUE)
        else:                  type_colors.append(C_GRAY)

    if emb.shape[0] > 5:
        perp = min(30, emb.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        emb2d = tsne.fit_transform(emb)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(emb2d[:, 0], emb2d[:, 1], c=type_colors, alpha=0.6, s=30)
        legend_items = [
            mpatches.Patch(color=C_GREEN,  label="Load zone (mine)"),
            mpatches.Patch(color=C_RED,    label="Dump zone"),
            mpatches.Patch(color=C_ORANGE, label="Hub"),
            mpatches.Patch(color=C_BLUE,   label="Connector"),
            mpatches.Patch(color=C_GRAY,   label="Other"),
        ]
        ax.legend(handles=legend_items, fontsize=9)
        ax.set_title("t-SNE of Learned Node Embeddings", fontweight="bold", color=C_GREEN)
        ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fname = os.path.join(PLOT_DIR, "gnn_node_embeddings_tsne.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== GNN ETA Model (GraphSAGE) ===\n")

    # ── Load graph ──
    print("Loading road graph...")
    with open("Map/map_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    road_graph = cache["road_graph"]

    node_positions = {k: np.array(v) for k, v in map_data.NODES.items()}

    graph_data, node_idx, pos_mean, pos_std = build_pyg_graph(road_graph, node_positions)

    # ── Load trips ──
    trips_path = os.path.join(OUT_DIR, "trips_graph.csv")
    if not os.path.exists(trips_path):
        trips_path = os.path.join(OUT_DIR, "trips.csv")
    print(f"Loading trips from {trips_path}...")
    df = pd.read_csv(trips_path)

    print("Preparing trip dataset...")
    (src_idx, dst_idx, trip_feats, targets_norm,
     targets_raw, t_mean, t_std, scaler, df_clean) = prepare_trip_dataset(df, node_idx)

    n_trips    = len(targets_raw)
    in_ch      = graph_data.x.shape[1]   # 11
    hidden_ch  = 64
    out_ch     = 32
    trip_dim   = trip_feats.shape[1]     # 4
    mlp_hidden = 64

    print(f"\nModel config:")
    print(f"  Node feature dim : {in_ch}")
    print(f"  Hidden channels  : {hidden_ch}")
    print(f"  Output channels  : {out_ch}")
    print(f"  Trip feature dim : {trip_dim}")
    print(f"  MLP hidden       : {mlp_hidden}")
    print(f"  Training samples : {n_trips}")
    print(f"  Device           : {DEVICE}")

    # ── Train full model ──
    print("\nTraining full model...")
    model = GraphSAGE_ETA(in_ch, hidden_ch, out_ch, trip_dim, mlp_hidden)
    model, losses = train_gnn(
        model, graph_data,
        src_idx, dst_idx, trip_feats, targets_norm,
        n_epochs=500, lr=0.001, patience=60
    )

    # Full model predictions
    model.eval()
    graph_data_dev = graph_data.to(DEVICE)
    with torch.no_grad():
        preds_norm_full = model(
            graph_data_dev,
            src_idx.to(DEVICE),
            dst_idx.to(DEVICE),
            trip_feats.to(DEVICE)
        ).cpu().numpy()
    preds_full = preds_norm_full * t_std + t_mean

    mae_full = mean_absolute_error(targets_raw, preds_full)
    r2_full  = r2_score(targets_raw, preds_full)
    print(f"\nFull model  MAE={mae_full:.1f}s  R²={r2_full:.3f}")

    # ── LOO Cross-validation ──
    print("\nRunning Leave-One-Out cross-validation...")
    preds_loo, mae_loo, r2_loo = evaluate_loo(
        graph_data, src_idx, dst_idx, trip_feats, targets_norm,
        targets_raw, t_mean, t_std,
        in_ch, hidden_ch, out_ch, trip_dim, mlp_hidden
    )
    print(f"\nLOO-CV  MAE={mae_loo:.1f}s  R²={r2_loo:.3f}")

    # ── Save model ──
    model_path = os.path.join(OUT_DIR, "gnn_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "node_idx"        : node_idx,
        "scaler"          : scaler,
        "target_mean"     : t_mean,
        "target_std"      : t_std,
        "config"          : {
            "in_ch": in_ch, "hidden_ch": hidden_ch,
            "out_ch": out_ch, "trip_dim": trip_dim, "mlp_hidden": mlp_hidden
        }
    }, model_path)
    print(f"\n  Model saved: {model_path}")

    # ── Save results ──
    results_df = df_clean[["trip_id","truck_id","load_zone","dump_zone",
                            "total_trip_duration_s","fuel_L","co2_kg"]].copy()
    results_df["gnn_predicted_s"] = preds_full.round(1)
    results_df["gnn_loo_pred_s"]  = preds_loo.round(1)
    results_df["gnn_error_s"]     = (preds_full - targets_raw).round(1)
    results_path = os.path.join(OUT_DIR, "gnn_eta_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Results saved: {results_path}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_training_loss(losses)
    plot_actual_vs_predicted(targets_raw, preds_loo, preds_full)
    plot_node_embeddings(model, graph_data, node_idx)

    # Comparison with baselines (from previous run)
    avg_dur = targets_raw.mean()
    plot_model_comparison(
        gnn_mae=mae_loo,
        gbt_mae=11.3,   # from eta_model.py run
        rf_mae =11.8,
        lr_mae =19.6,
        avg_dur=avg_dur
    )

    # ── Final summary ──
    pct = 100 * mae_loo / avg_dur
    print(f"\n{'='*52}")
    print(f"  GraphSAGE GNN")
    print(f"  LOO-MAE          : {mae_loo:.1f}s  ({pct:.1f}% of avg trip)")
    print(f"  LOO-R²           : {r2_loo:.3f}")
    print(f"  Full model MAE   : {mae_full:.1f}s")
    print(f"  Full model R²    : {r2_full:.3f}")
    print(f"  Baseline GBT MAE : 11.3s  (3.3%)")
    delta = 11.3 - mae_loo
    print(f"  vs Baseline      : {delta:+.1f}s  ({'GNN wins' if delta > 0 else 'need more data'})")
    print(f"  Plots saved to   : {PLOT_DIR}/")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
