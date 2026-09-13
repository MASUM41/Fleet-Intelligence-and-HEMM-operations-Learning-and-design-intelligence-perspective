"""
gnn_eta_model.py  -  Graph Neural Network ETA predictor for HEMM fleet.

Architecture:
    - Node features: position (x,y), node type (load/dump/connector/hub),
      degree, avg edge weight
    - Edge features: road distance, normalised weight
    - GNN: 3-layer GraphSAGE (message passing) -> trip-level pooling
    - Head: MLP predicting trip duration

Training:
    - Each trip becomes a graph-level sample
    - Source node = load_zone, target node = dump_zone
    - Node embeddings from full road graph via GraphSAGE
    - Trip features appended to pooled embeddings

Run from Simulation/ folder:
    python gnn_eta_model.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from Map import map_loader as map_data

OUT_DIR  = "analysis"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

C_GREEN  = "#1A6B3A"
C_BLUE   = "#2E86AB"
C_ORANGE = "#F4A261"
C_RED    = "#E63946"

# ══════════════════════════════════════════════════════════════════════════════
#  1. BUILD GRAPH DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

NODE_TYPES = {
    "load_zone" : 0,
    "dump_zone" : 1,
    "connector" : 2,
    "hub"       : 3,
    "n_haul"    : 4,
    "parking"   : 5,
    "other"     : 6,
}

def get_node_type(name):
    for key in NODE_TYPES:
        if key in name:
            return NODE_TYPES[key]
    return NODE_TYPES["other"]


def build_pyg_graph(road_graph, nodes_pos):
    """
    Convert road_graph adjacency list + node positions into a
    PyTorch Geometric Data object with node and edge features.
    """
    node_names = sorted(road_graph.keys())
    node_idx   = {name: i for i, name in enumerate(node_names)}
    N          = len(node_names)

    # ── Node features ────────────────────────────────────────────────────────
    # [x_norm, y_norm, node_type_onehot(7), degree, avg_edge_weight_norm]
    all_positions = np.array([nodes_pos.get(n, np.zeros(2)) for n in node_names])
    pos_min, pos_max = all_positions.min(0), all_positions.max(0)
    pos_range = pos_max - pos_min + 1e-8

    node_feats = []
    for name in node_names:
        pos   = nodes_pos.get(name, np.zeros(2))
        x_n   = (pos[0] - pos_min[0]) / pos_range[0]
        y_n   = (pos[1] - pos_min[1]) / pos_range[1]
        ntype = get_node_type(name)
        onehot = [1.0 if i == ntype else 0.0 for i in range(len(NODE_TYPES))]
        edges  = road_graph.get(name, [])
        degree = len(edges)
        avg_w  = np.mean([float(w) for _, w in edges]) if edges else 0.0
        node_feats.append([x_n, y_n] + onehot + [degree / 10.0, avg_w / 1000.0])

    x = torch.tensor(node_feats, dtype=torch.float)

    # ── Edges ─────────────────────────────────────────────────────────────────
    edge_src, edge_dst, edge_attr = [], [], []
    all_weights = [float(w) for edges in road_graph.values() for _, w in edges]
    w_max = max(all_weights) + 1e-8

    for src_name, edges in road_graph.items():
        if src_name not in node_idx: continue
        for dst_name, weight in edges:
            if dst_name not in node_idx: continue
            edge_src.append(node_idx[src_name])
            edge_dst.append(node_idx[dst_name])
            edge_attr.append([float(weight) / w_max])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_idx, node_names


# ══════════════════════════════════════════════════════════════════════════════
#  2. GNN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TripGNN(nn.Module):
    """
    GraphSAGE-based ETA predictor.

    Forward pass:
        1. Run 3 GraphSAGE layers on full road graph
        2. Extract embeddings for source (load_zone) and target (dump_zone) nodes
        3. Concatenate with trip-level features (cargo, queue_proxy, trip_number)
        4. MLP regression head -> predicted trip duration
    """
    def __init__(self, node_feat_dim, trip_feat_dim, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(node_feat_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)

        # MLP head: source_emb + target_emb + trip_feats -> duration
        mlp_in = hidden * 2 + trip_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_graph(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

    def forward(self, x, edge_index, src_idx, dst_idx, trip_feats):
        # Node embeddings for full graph
        node_emb = self.encode_graph(x, edge_index)   # (N, hidden)

        # Extract source and destination embeddings
        src_emb = node_emb[src_idx]   # (B, hidden)
        dst_emb = node_emb[dst_idx]   # (B, hidden)

        # Concatenate with trip features
        combined = torch.cat([src_emb, dst_emb, trip_feats], dim=1)
        out = self.mlp(combined)
        return out.squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
#  3. DATASET PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_dataset(trips_df, node_idx):
    """
    For each trip, extract:
        - src_idx: index of load_zone node
        - dst_idx: index of dump_zone node
        - trip_feats: [cargo_kg_norm, queue_proxy, trip_number_norm, load_wait_norm]
        - target: total_trip_duration_s (normalised)
    """
    trips_df = trips_df.copy()

    # Drop outliers
    mean_d = trips_df["total_trip_duration_s"].mean()
    std_d  = trips_df["total_trip_duration_s"].std()
    trips_df = trips_df[trips_df["total_trip_duration_s"] < mean_d + 2*std_d].reset_index(drop=True)

    # Fill missing
    trips_df["travel_to_mine_s"] = trips_df["travel_to_mine_s"].fillna(
        trips_df["travel_to_mine_s"].median())

    route_avg = trips_df.groupby(["load_zone","dump_zone"])["travel_to_mine_s"].transform("median")
    trips_df["queue_proxy"] = trips_df["travel_to_mine_s"] / route_avg.replace(0, 1)

    # Normalise trip features
    scaler = StandardScaler()
    trip_feat_cols = ["cargo_kg", "queue_proxy", "trip_number", "load_wait_s"]
    trip_feats_np  = scaler.fit_transform(trips_df[trip_feat_cols].values.astype(float))

    # Target normalisation
    y_mean = trips_df["total_trip_duration_s"].mean()
    y_std  = trips_df["total_trip_duration_s"].std()
    y_norm = ((trips_df["total_trip_duration_s"] - y_mean) / y_std).values

    # Node indices
    src_indices, dst_indices = [], []
    valid_mask = []

    for _, row in trips_df.iterrows():
        lz = row["load_zone"]
        dz = row["dump_zone"]
        if lz in node_idx and dz in node_idx:
            src_indices.append(node_idx[lz])
            dst_indices.append(node_idx[dz])
            valid_mask.append(True)
        else:
            src_indices.append(0)
            dst_indices.append(0)
            valid_mask.append(False)
            print(f"  Warning: {lz} or {dz} not in graph")

    return (
        torch.tensor(src_indices, dtype=torch.long),
        torch.tensor(dst_indices, dtype=torch.long),
        torch.tensor(trip_feats_np, dtype=torch.float),
        torch.tensor(y_norm, dtype=torch.float),
        y_mean, y_std, scaler,
        trips_df["total_trip_duration_s"].values,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  4. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_gnn(model, graph_data, src_idx, dst_idx, trip_feats, y,
              n_epochs=300, lr=1e-3, patience=40):
    model = model.to(DEVICE)
    x          = graph_data.x.to(DEVICE)
    edge_index = graph_data.edge_index.to(DEVICE)
    src_idx    = src_idx.to(DEVICE)
    dst_idx    = dst_idx.to(DEVICE)
    trip_feats = trip_feats.to(DEVICE)
    y          = y.to(DEVICE)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    train_losses = []
    best_loss    = float("inf")
    best_state   = None
    no_improve   = 0

    # Leave-one-out style: train on all but one, validate on that one
    # With 27 samples, use 80/20 split for simplicity
    n     = len(y)
    perm  = torch.randperm(n)
    split = int(0.8 * n)
    tr_idx = perm[:split]
    va_idx = perm[split:]

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x, edge_index, src_idx[tr_idx], dst_idx[tr_idx], trip_feats[tr_idx])
        loss = F.mse_loss(pred, y[tr_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x, edge_index, src_idx[va_idx], dst_idx[va_idx], trip_feats[va_idx])
            val_loss = F.mse_loss(val_pred, y[va_idx]).item()

        scheduler.step(val_loss)
        train_losses.append(loss.item())

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, train_losses, tr_idx, va_idx


# ══════════════════════════════════════════════════════════════════════════════
#  5. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, graph_data, src_idx, dst_idx, trip_feats, y_norm, y_actual, y_mean, y_std):
    model.eval()
    x          = graph_data.x.to(DEVICE)
    edge_index = graph_data.edge_index.to(DEVICE)

    with torch.no_grad():
        pred_norm = model(
            x, edge_index,
            src_idx.to(DEVICE),
            dst_idx.to(DEVICE),
            trip_feats.to(DEVICE),
        ).cpu().numpy()

    pred_actual = pred_norm * y_std + y_mean
    mae  = mean_absolute_error(y_actual, pred_actual)
    r2   = r2_score(y_actual, pred_actual)
    return pred_actual, mae, r2


# ══════════════════════════════════════════════════════════════════════════════
#  6. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_loss(train_losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, color=C_BLUE, lw=1.5, alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (normalised)")
    ax.set_title("GNN Training Loss", fontweight="bold", color=C_BLUE)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "gnn_training_loss.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def plot_gnn_vs_baseline(y_actual, gnn_pred, baseline_mae, gnn_mae):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GNN vs Gradient Boosting Baseline", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_actual, gnn_pred, color=C_GREEN, alpha=0.8, edgecolors="white", s=80, label="GNN")
    lims = [min(y_actual.min(), gnn_pred.min())-10, max(y_actual.max(), gnn_pred.max())+10]
    ax.plot(lims, lims, "--", color=C_GRAY if True else "", lw=1.5, label="Perfect")
    ax.set_xlabel("Actual Duration (s)"); ax.set_ylabel("Predicted Duration (s)")
    ax.set_title("GNN: Actual vs Predicted"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(lims); ax.set_ylim(lims)

    ax2 = axes[1]
    models  = ["Gradient Boosting\n(Baseline)", "GNN\n(Graph-Aware)"]
    maes    = [baseline_mae, gnn_mae]
    colours = [C_BLUE, C_GREEN]
    bars = ax2.bar(models, maes, color=colours, alpha=0.85, edgecolor="white", width=0.4)
    ax2.set_ylabel("MAE (seconds)"); ax2.set_title("MAE Comparison\n(Lower is better)")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, maes):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f"{val:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

    winner = "GNN" if gnn_mae < baseline_mae else "Gradient Boosting"
    ax2.set_title(f"MAE Comparison — {winner} wins", fontweight="bold")

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "gnn_vs_baseline.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


def plot_node_embeddings(model, graph_data, node_names, node_idx):
    """t-SNE of node embeddings coloured by node type."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return

    model.eval()
    with torch.no_grad():
        emb = model.encode_graph(
            graph_data.x.to(DEVICE),
            graph_data.edge_index.to(DEVICE),
        ).cpu().numpy()

    if emb.shape[0] > 50:
        tsne   = TSNE(n_components=2, random_state=42, perplexity=min(30, emb.shape[0]//3))
        emb_2d = tsne.fit_transform(emb)
    else:
        emb_2d = emb[:, :2]

    type_colors = {0: C_GREEN, 1: C_RED, 2: C_BLUE, 3: C_ORANGE, 4:"#9B59B6", 5:"#555", 6:C_GRAY}
    type_labels = {0:"Load Zone", 1:"Dump Zone", 2:"Connector", 3:"Hub", 4:"Haul Road", 5:"Parking", 6:"Other"}

    fig, ax = plt.subplots(figsize=(10, 7))
    for ttype, color in type_colors.items():
        mask = [get_node_type(n) == ttype for n in node_names]
        if any(mask):
            pts = emb_2d[mask]
            ax.scatter(pts[:,0], pts[:,1], c=color, s=30, alpha=0.7,
                       label=type_labels[ttype], edgecolors="white", linewidths=0.3)

    # Label key zones
    for name, idx in node_idx.items():
        if "load_zone" in name or "dump_zone" in name:
            ax.annotate(name.replace("load_zone_","M").replace("dump_zone_","D"),
                        emb_2d[idx], fontsize=7, alpha=0.8)

    ax.set_title("Node Embeddings (t-SNE) — Learned by GNN", fontweight="bold", color=C_BLUE)
    ax.legend(fontsize=9, loc="best"); ax.axis("off")
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "gnn_node_embeddings.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== GNN ETA Model for HEMM Fleet ===\n")
    print(f"Using device: {DEVICE}\n")

    # ── Load graph ──
    print("Loading road graph...")
    with open("Map/map_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    road_graph = cache["road_graph"]

    nodes_pos = {k: np.array(v) for k, v in map_data.NODES.items()}
    print(f"  Nodes: {len(road_graph)}  |  Positions: {len(nodes_pos)}")

    print("Building PyG graph...")
    graph_data, node_idx, node_names = build_pyg_graph(road_graph, nodes_pos)
    print(f"  Node features: {graph_data.x.shape}")
    print(f"  Edges: {graph_data.edge_index.shape[1]}")

    # ── Load trips ──
    trips_path = os.path.join(OUT_DIR, "trips_graph.csv")
    if not os.path.exists(trips_path):
        trips_path = os.path.join(OUT_DIR, "trips.csv")
    df = pd.read_csv(trips_path)
    print(f"\nTrips loaded: {len(df)}")

    # ── Prepare dataset ──
    print("Preparing dataset...")
    src_idx, dst_idx, trip_feats, y_norm, y_mean, y_std, scaler, y_actual = \
        prepare_dataset(df, node_idx)
    print(f"  Training samples: {len(y_norm)}  (after outlier removal)")
    print(f"  Trip feat dim: {trip_feats.shape[1]}")

    # ── Build model ──
    node_feat_dim = graph_data.x.shape[1]
    trip_feat_dim = trip_feats.shape[1]
    model = TripGNN(node_feat_dim, trip_feat_dim, hidden=64)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: TripGNN  |  Params: {total_params:,}")
    print(f"  Node feat dim : {node_feat_dim}")
    print(f"  Trip feat dim : {trip_feat_dim}")
    print(f"  Hidden dim    : 64")
    print(f"  Layers        : 3x GraphSAGE + MLP head\n")

    # ── Train ──
    print("Training...")
    model, train_losses, tr_idx, va_idx = train_gnn(
        model, graph_data, src_idx, dst_idx, trip_feats, y_norm,
        n_epochs=500, lr=1e-3, patience=50,
    )

    # ── Evaluate ──
    print("\nEvaluating on all samples...")
    pred_actual, mae, r2 = evaluate(
        model, graph_data, src_idx, dst_idx, trip_feats, y_norm, y_actual, y_mean, y_std
    )

    baseline_mae = 11.3   # from eta_model.py results
    avg_dur      = np.mean(y_actual)

    print(f"\n{'='*50}")
    print(f"  GNN MAE          : {mae:.1f}s  ({100*mae/avg_dur:.1f}% of avg trip)")
    print(f"  GNN R²           : {r2:.4f}")
    print(f"  Baseline MAE     : {baseline_mae:.1f}s  (Gradient Boosting)")
    improvement = baseline_mae - mae
    print(f"  Improvement      : {improvement:+.1f}s  ({'GNN wins' if improvement>0 else 'need more data for GNN to shine'})")
    print(f"{'='*50}\n")

    # ── Save predictions ──
    results_df = pd.DataFrame({
        "trip_id"       : df["trip_id"].values[:len(pred_actual)],
        "actual_s"      : y_actual,
        "gnn_pred_s"    : pred_actual.round(1),
        "error_s"       : (pred_actual - y_actual).round(1),
    })
    results_path = os.path.join(OUT_DIR, "gnn_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

    # ── Save model ──
    model_path = os.path.join(OUT_DIR, "gnn_eta_model.pt")
    torch.save({
        "model_state"   : model.state_dict(),
        "node_idx"      : node_idx,
        "y_mean"        : y_mean,
        "y_std"         : y_std,
        "scaler"        : scaler,
        "node_feat_dim" : node_feat_dim,
        "trip_feat_dim" : trip_feat_dim,
    }, model_path)
    print(f"  Saved: {model_path}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_training_loss(train_losses)
    plot_gnn_vs_baseline(y_actual, pred_actual, baseline_mae, mae)
    plot_node_embeddings(model, graph_data, node_names, node_idx)

    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print(f"Plots saved to:       {PLOT_DIR}/")


if __name__ == "__main__":
    main()
