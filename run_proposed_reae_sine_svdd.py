import os
import random
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, add_self_loops, degree

from karateclub import SINE


warnings.filterwarnings("ignore")


# =========================================================
# 1. Reproducibility
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 2. ReAE / WL-like Attribute Aggregation
# =========================================================

def reae_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    ReAE-style rectification and row-wise normalization.

    This approximates the rectification step described in the manuscript:
    after neighborhood aggregation, insignificant propagated values are
    reduced and each node representation is normalized.
    """
    x = F.relu(x)
    row_sum = x.sum(dim=1, keepdim=True).clamp(min=eps)
    return x / row_sum


def wl_reae_aggregation(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_iterations: int = 2,
    add_loops: bool = True,
) -> torch.Tensor:
    """
    WL-like attribute aggregation with ReAE normalization.

    X_{k+1} = ReAE(D^{-1/2} (A + I) D^{-1/2} X_k)
    """
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    if add_loops:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    row, col = edge_index

    deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    out = x

    for _ in range(num_iterations):
        aggregated = torch.zeros_like(out)
        aggregated.index_add_(0, row, out[col] * norm.view(-1, 1))
        out = reae_normalize(aggregated)

    return out


# =========================================================
# 3. Dataset Loading
# =========================================================

def load_planetoid_dataset(dataset_name, root, device):
    """
    Load Cora or CiteSeer and create a link prediction split.

    The split follows the experimental logic:
    - train graph contains observed training links
    - test positives are held-out observed links
    - test negatives are sampled non-existing links
    """
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(
            num_val=0.10,
            num_test=0.20,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
            split_labels=True,
        ),
    ])

    dataset = Planetoid(
        root=os.path.join(root, dataset_name),
        name=dataset_name,
        transform=transform,
    )

    train_data, val_data, test_data = dataset[0]
    return dataset, train_data, val_data, test_data


# =========================================================
# 4. Conversion Utilities for SINE
# =========================================================

def pyg_edge_index_to_networkx(edge_index, num_nodes):
    """
    Convert PyG edge_index to a NetworkX graph for Karate Club SINE.

    SINE expects a NetworkX graph whose nodes are indexed from 0 to n-1.
    """
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edges = edge_index.detach().cpu().numpy().T

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from((int(u), int(v)) for u, v in edges if int(u) != int(v))

    return graph


def tensor_features_to_sparse_coo(x: torch.Tensor):
    """
    Convert node features to scipy sparse COO matrix.

    Karate Club SINE accepts numpy arrays or scipy sparse COO matrices.
    Cora and CiteSeer features are high-dimensional and sparse, so COO is preferable.
    """
    x_np = x.detach().cpu().numpy()
    return sp.coo_matrix(x_np)


# =========================================================
# 5. Node Embedding by SINE
# =========================================================

def train_sine_embeddings(
    graph,
    x_sparse,
    dimensions=128,
    walk_number=10,
    walk_length=80,
    window_size=5,
    epochs=1,
    workers=4,
    learning_rate=0.05,
    seed=42,
):
    """
    Train SINE node embeddings using Karate Club.

    This corresponds to the manuscript stage:
    Node Embedding by SINE.
    """
    model = SINE(
        walk_number=walk_number,
        walk_length=walk_length,
        dimensions=dimensions,
        workers=workers,
        window_size=window_size,
        epochs=epochs,
        learning_rate=learning_rate,
        min_count=1,
        seed=seed,
    )

    model.fit(graph, x_sparse)
    z = model.get_embedding()

    return z


# =========================================================
# 6. Link Representation Operators
# =========================================================

def build_link_features_np(
    z: np.ndarray,
    edge_index: torch.Tensor,
    operator: str = "merge",
) -> np.ndarray:
    """
    Build link representations from SINE node embeddings.

    Supported operators:
    - average
    - hadamard
    - weighted_l1
    - weighted_l2
    - merge
    """
    edge_np = edge_index.detach().cpu().numpy()
    src = edge_np[0]
    dst = edge_np[1]

    zu = z[src]
    zv = z[dst]

    operator = operator.lower()

    if operator == "average":
        return (zu + zv) / 2.0

    if operator == "hadamard":
        return zu * zv

    if operator in ["weighted_l1", "l1"]:
        return np.abs(zu - zv)

    if operator in ["weighted_l2", "l2"]:
        return (zu - zv) ** 2

    if operator == "merge":
        return np.concatenate([zu, zv], axis=1)

    raise ValueError(f"Unknown link representation operator: {operator}")


def get_edge_label_data(data):
    """
    Combine positive and negative edge-label indices.
    """
    pos_edge_index = data.pos_edge_label_index
    neg_edge_index = data.neg_edge_label_index

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    pos_y = torch.ones(pos_edge_index.size(1), device=edge_label_index.device)
    neg_y = torch.zeros(neg_edge_index.size(1), device=edge_label_index.device)

    edge_label = torch.cat([pos_y, neg_y], dim=0)

    return edge_label_index, edge_label


# =========================================================
# 7. SVDD / One-Class Kernel Model
# =========================================================

def train_svdd_model(
    x_train_links: np.ndarray,
    nu: float = 0.1,
    gamma="scale",
):
    """
    Train a one-class RBF kernel model on observed links only.

    OneClassSVM with RBF kernel is used here as a practical SVDD-style model.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_links)

    svdd = OneClassSVM(
        kernel="rbf",
        nu=nu,
        gamma=gamma,
    )

    svdd.fit(x_train_scaled)

    return svdd, scaler


def compute_metrics(y_true, scores):
    """
    Compute AUC, AP, and Precision.

    Precision is computed using the zero decision boundary of OneClassSVM.
    """
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    y_pred = (scores >= 0).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)

    return auc, ap, precision


# =========================================================
# 8. Full Pipeline
# =========================================================

def run_once(dataset_name, seed, args, device):
    set_seed(seed)

    dataset, train_data, val_data, test_data = load_planetoid_dataset(
        dataset_name=dataset_name,
        root=args.data_root,
        device=device,
    )

    num_nodes = train_data.num_nodes

    # -----------------------------------------------------
    # Stage 1: ReAE / WL-like Attribute Aggregation
    # -----------------------------------------------------
    aggregated_x = wl_reae_aggregation(
        x=train_data.x,
        edge_index=train_data.edge_index,
        num_nodes=num_nodes,
        num_iterations=args.reae_iterations,
    )

    # -----------------------------------------------------
    # Stage 2: Node Embedding by SINE
    # -----------------------------------------------------
    graph = pyg_edge_index_to_networkx(
        edge_index=train_data.edge_index,
        num_nodes=num_nodes,
    )

    x_sparse = tensor_features_to_sparse_coo(aggregated_x)

    z = train_sine_embeddings(
        graph=graph,
        x_sparse=x_sparse,
        dimensions=args.embedding_dim,
        walk_number=args.walk_number,
        walk_length=args.walk_length,
        window_size=args.window_size,
        epochs=args.sine_epochs,
        workers=args.workers,
        learning_rate=args.sine_lr,
        seed=seed,
    )

    # -----------------------------------------------------
    # Stage 3: Link Representation Operators
    # -----------------------------------------------------
    x_train_links = build_link_features_np(
        z=z,
        edge_index=train_data.pos_edge_label_index,
        operator=args.link_operator,
    )

    test_edge_label_index, test_edge_label = get_edge_label_data(test_data)

    x_test_links = build_link_features_np(
        z=z,
        edge_index=test_edge_label_index,
        operator=args.link_operator,
    )

    y_test = test_edge_label.detach().cpu().numpy()

    # -----------------------------------------------------
    # Stage 4: SVDD / One-Class Kernel Model
    # -----------------------------------------------------
    svdd, scaler = train_svdd_model(
        x_train_links=x_train_links,
        nu=args.svdd_nu,
        gamma=args.svdd_gamma,
    )

    x_test_scaled = scaler.transform(x_test_links)
    scores = svdd.decision_function(x_test_scaled)

    # -----------------------------------------------------
    # Stage 5: AUC / AP / Precision
    # -----------------------------------------------------
    auc, ap, precision = compute_metrics(y_test, scores)

    return {
        "Dataset": dataset_name,
        "Method": "Proposed-ReAE-SINE-SVDD",
        "Seed": seed,
        "LinkOperator": args.link_operator,
        "ReAEIterations": args.reae_iterations,
        "EmbeddingDim": args.embedding_dim,
        "AUC": auc,
        "AP": ap,
        "Precision": precision,
    }


# =========================================================
# 9. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output", type=str, default="proposed_reae_sine_svdd_results.csv")

    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])

    parser.add_argument("--reae_iterations", type=int, default=2)

    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--walk_number", type=int, default=10)
    parser.add_argument("--walk_length", type=int, default=80)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--sine_epochs", type=int, default=1)
    parser.add_argument("--sine_lr", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument(
        "--link_operator",
        type=str,
        default="merge",
        choices=["average", "hadamard", "weighted_l1", "weighted_l2", "merge"],
    )

    parser.add_argument("--svdd_nu", type=float, default=0.1)
    parser.add_argument("--svdd_gamma", type=str, default="scale")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Pipeline: Dataset -> ReAE/WL -> SINE -> {args.link_operator} -> SVDD -> metrics")

    all_results = []

    for dataset_name in args.datasets:
        for seed in args.seeds:
            print(f"\nRunning | Dataset={dataset_name} | Seed={seed}")

            result = run_once(
                dataset_name=dataset_name,
                seed=seed,
                args=args,
                device=device,
            )

            print(
                f"AUC={result['AUC']:.4f}, "
                f"AP={result['AP']:.4f}, "
                f"Precision={result['Precision']:.4f}"
            )

            all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)

    summary = (
        df.groupby(["Dataset", "Method", "LinkOperator"])
        .agg(
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", "std"),
            AP_mean=("AP", "mean"),
            AP_std=("AP", "std"),
            Precision_mean=("Precision", "mean"),
            Precision_std=("Precision", "std"),
        )
        .reset_index()
    )

    summary_output = args.output.replace(".csv", "_summary.csv")
    summary.to_csv(summary_output, index=False)

    print("\nDetailed results")
    print(df)

    print("\nSummary results")
    print(summary)

    print(f"\nSaved detailed results to: {args.output}")
    print(f"Saved summary results to: {summary_output}")


if __name__ == "__main__":
    main()