import os
import random
import argparse
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import to_undirected, add_self_loops, degree


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
# 2. ReAE / WL-like Information Aggregation
# =========================================================

def reae_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Rectified Aggregate Element (ReAE)-style normalization.

    The manuscript describes ReAE as a rectification step after aggregation
    to reduce the effect of insignificant propagated values and stabilize
    node attributes before embedding.

    Here:
    - negative values are removed using ReLU
    - each node feature vector is normalized by its row sum
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
    WL-inspired neighborhood aggregation with ReAE normalization.

    This approximates the information aggregation stage in the manuscript:
    node attributes are repeatedly updated using neighborhood information,
    then normalized using a ReAE-like operation.

    We use a signless normalized aggregation form:
        S = D^{-1/2} (A + I) D^{-1/2}
        X_{k+1} = ReAE(S X_k)

    This keeps the implementation stable and suitable for Cora/CiteSeer.
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
# 3. Node Embedding Model
# =========================================================

class GCNEncoderForEmbedding(nn.Module):
    """
    Unsupervised GCN encoder used inside a Graph Autoencoder.

    In the manuscript, the framework is model-agnostic and can work
    with different embedding models. For Cora/CiteSeer, this encoder is
    suitable because it uses both topology and node attributes.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_unsupervised_embedding_model(
    train_data,
    val_data,
    in_channels,
    hidden_channels,
    out_channels,
    device,
    epochs=300,
    lr=0.01,
    weight_decay=1e-4,
    patience=40,
):
    """
    Trains an unsupervised GAE encoder to obtain node embeddings.

    This corresponds to Stage 2 in the manuscript: Node Embedding.
    """
    encoder = GCNEncoderForEmbedding(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    ).to(device)

    model = GAE(encoder).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_auc = -1.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z = model.encode(train_data.x, train_data.edge_index)

        loss = model.recon_loss(
            z,
            train_data.pos_edge_label_index,
            train_data.neg_edge_label_index,
        )

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z_val = model.encode(val_data.x, val_data.edge_index)
            val_edge_index, val_label = get_edge_label_data(val_data)
            val_score = model.decoder(z_val, val_edge_index, sigmoid=True)

            val_auc = roc_auc_score(
                val_label.detach().cpu().numpy(),
                val_score.detach().cpu().numpy(),
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# =========================================================
# 4. Link Representation Operators
# =========================================================

def build_link_features(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    operator: str = "merge",
) -> torch.Tensor:
    """
    Implements link representation operators described in the manuscript:

    - average
    - hadamard
    - weighted_l1
    - weighted_l2
    - merge

    Merge corresponds to concatenating the two node embeddings.
    """
    src, dst = edge_index
    zu = z[src]
    zv = z[dst]

    operator = operator.lower()

    if operator == "average":
        return (zu + zv) / 2.0

    if operator == "hadamard":
        return zu * zv

    if operator in ["weighted_l1", "l1"]:
        return torch.abs(zu - zv)

    if operator in ["weighted_l2", "l2"]:
        return (zu - zv) ** 2

    if operator == "merge":
        return torch.cat([zu, zv], dim=1)

    raise ValueError(f"Unknown link representation operator: {operator}")


# =========================================================
# 5. SVDD-style Learning
# =========================================================

def train_svdd_model(
    x_train_links: np.ndarray,
    nu: float = 0.1,
    gamma="scale",
):
    """
    SVDD-style one-class learning.

    scikit-learn does not provide a class named SVDD directly.
    OneClassSVM with RBF kernel is commonly used as a practical
    implementation of one-class hypersphere/boundary learning.

    The model is trained only on observed positive links.
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


def svdd_decision_scores(
    svdd,
    scaler,
    x_test_links: np.ndarray,
):
    """
    Returns continuous SVDD decision scores.

    Higher score means the candidate link is more similar to the
    observed-link distribution learned by the SVDD model.
    """
    x_test_scaled = scaler.transform(x_test_links)
    return svdd.decision_function(x_test_scaled)


# =========================================================
# 6. Dataset and Evaluation Utilities
# =========================================================

def get_edge_label_data(data):
    """
    Combines positive and negative edges from RandomLinkSplit.
    """
    pos_edge_index = data.pos_edge_label_index
    neg_edge_index = data.neg_edge_label_index

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    pos_y = torch.ones(pos_edge_index.size(1), device=edge_label_index.device)
    neg_y = torch.zeros(neg_edge_index.size(1), device=edge_label_index.device)

    edge_label = torch.cat([pos_y, neg_y], dim=0)

    return edge_label_index, edge_label


def load_planetoid_dataset(dataset_name, root, device):
    """
    Loads Cora or CiteSeer and applies link prediction split.

    The split follows the paper logic:
    - observed links for training
    - positive test links
    - balanced negative links
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


def compute_metrics(y_true, scores):
    """
    Computes AUC, AP, and Precision.

    Precision is computed using zero as the decision threshold.
    """
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    y_pred = (scores >= 0).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)

    return auc, ap, precision


# =========================================================
# 7. Full Proposed Framework
# =========================================================

def run_proposed_framework_once(
    dataset_name,
    seed,
    args,
    device,
):
    set_seed(seed)

    dataset, train_data, val_data, test_data = load_planetoid_dataset(
        dataset_name=dataset_name,
        root=args.data_root,
        device=device,
    )

    num_nodes = train_data.num_nodes
    in_channels = dataset.num_features

    # -----------------------------------------------------
    # Stage 1: Information Aggregation with WL/ReAE
    # -----------------------------------------------------
    train_data.x = wl_reae_aggregation(
        x=train_data.x,
        edge_index=train_data.edge_index,
        num_nodes=num_nodes,
        num_iterations=args.reae_iterations,
    )

    val_data.x = wl_reae_aggregation(
        x=val_data.x,
        edge_index=val_data.edge_index,
        num_nodes=num_nodes,
        num_iterations=args.reae_iterations,
    )

    test_data.x = wl_reae_aggregation(
        x=test_data.x,
        edge_index=test_data.edge_index,
        num_nodes=num_nodes,
        num_iterations=args.reae_iterations,
    )

    # -----------------------------------------------------
    # Stage 2: Node Embedding
    # -----------------------------------------------------
    embedding_model = train_unsupervised_embedding_model(
        train_data=train_data,
        val_data=val_data,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.embedding_dim,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    embedding_model.eval()

    with torch.no_grad():
        z_train = embedding_model.encode(train_data.x, train_data.edge_index)
        z_test = embedding_model.encode(test_data.x, test_data.edge_index)

    # -----------------------------------------------------
    # Stage 3: Link Representation
    # -----------------------------------------------------
    x_train_links = build_link_features(
        z=z_train,
        edge_index=train_data.pos_edge_label_index,
        operator=args.link_operator,
    ).detach().cpu().numpy()

    test_edge_label_index, test_edge_label = get_edge_label_data(test_data)

    x_test_links = build_link_features(
        z=z_test,
        edge_index=test_edge_label_index,
        operator=args.link_operator,
    ).detach().cpu().numpy()

    y_test = test_edge_label.detach().cpu().numpy()

    # -----------------------------------------------------
    # Stage 4: SVDD
    # -----------------------------------------------------
    svdd, scaler = train_svdd_model(
        x_train_links=x_train_links,
        nu=args.svdd_nu,
        gamma=args.svdd_gamma,
    )

    scores = svdd_decision_scores(
        svdd=svdd,
        scaler=scaler,
        x_test_links=x_test_links,
    )

    auc, ap, precision = compute_metrics(y_test, scores)

    return {
        "Dataset": dataset_name,
        "Method": "Proposed-ReAE-GAE-SVDD",
        "Seed": seed,
        "LinkOperator": args.link_operator,
        "ReAEIterations": args.reae_iterations,
        "AUC": auc,
        "AP": ap,
        "Precision": precision,
    }


# =========================================================
# 8. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output", type=str, default="proposed_svdd_results.csv")

    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)

    parser.add_argument("--reae_iterations", type=int, default=2)

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
    print(f"Link operator: {args.link_operator}")
    print(f"ReAE iterations: {args.reae_iterations}")

    all_results = []

    for dataset_name in args.datasets:
        for seed in args.seeds:
            print(f"\nRunning Proposed Framework | Dataset={dataset_name} | Seed={seed}")

            result = run_proposed_framework_once(
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

    print("\nDetailed Results")
    print(df)

    print("\nSummary Results")
    print(summary)

    print(f"\nSaved detailed results to: {args.output}")
    print(f"Saved summary results to: {summary_output}")


if __name__ == "__main__":
    main()