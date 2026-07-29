"""
Phase 3.4 — Graph Metrics

Computes global and node-level graph metrics from the weighted adjacency
matrix using NetworkX.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def compute_graph_metrics(
    adj_count: np.ndarray,
    label_df: pd.DataFrame,
) -> dict:
    """
    Compute global graph metrics from a weighted streamline-count adjacency matrix.

    Returns a flat dict with:
      node_count, edge_count, graph_density,
      mean_degree, mean_weighted_degree,
      mean_clustering, connected_components,
      largest_component_size, characteristic_path_length (if connected).
    """
    n = adj_count.shape[0]

    # Build NetworkX undirected weighted graph (exclude self-loops + zero edges)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(adj_count[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)

    node_count = n
    edge_count = G.number_of_edges()

    # Density: 2|E| / (|V|(|V|-1))
    graph_density = float(nx.density(G))

    degrees = [d for _, d in G.degree()]
    mean_degree = float(np.mean(degrees)) if degrees else 0.0

    w_degrees = [d for _, d in G.degree(weight="weight")]
    mean_weighted_degree = float(np.mean(w_degrees)) if w_degrees else 0.0

    # Clustering coefficient (binary, then weighted)
    try:
        mean_clustering          = float(nx.average_clustering(G))
        mean_clustering_weighted = float(nx.average_clustering(G, weight="weight"))
    except Exception:
        mean_clustering          = 0.0
        mean_clustering_weighted = 0.0

    # Connected components
    components  = list(nx.connected_components(G))
    n_components = len(components)
    largest_size = max(len(c) for c in components) if components else 0

    # Characteristic path length (only for largest component if disconnected)
    cpl = None
    if edge_count > 0:
        try:
            if nx.is_connected(G):
                cpl = float(nx.average_shortest_path_length(G, weight=None))
            else:
                lcc  = G.subgraph(max(components, key=len)).copy()
                if lcc.number_of_edges() > 0:
                    cpl = float(nx.average_shortest_path_length(lcc, weight=None))
        except Exception:
            cpl = None

    return {
        "node_count":              node_count,
        "edge_count":              edge_count,
        "graph_density":           round(graph_density, 6),
        "mean_degree":             round(mean_degree, 3),
        "mean_weighted_degree":    round(mean_weighted_degree, 3),
        "mean_clustering":         round(mean_clustering, 6),
        "mean_clustering_weighted":round(mean_clustering_weighted, 6),
        "connected_components":    n_components,
        "largest_component_size":  largest_size,
        "characteristic_path_length": round(cpl, 4) if cpl is not None else None,
    }


def compute_node_metrics(
    adj_count: np.ndarray,
    label_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-node metrics: degree, weighted_degree, betweenness, clustering.
    Returns a DataFrame with one row per ROI.
    """
    n = adj_count.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(adj_count[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)

    degree   = dict(G.degree())
    w_degree = dict(G.degree(weight="weight"))
    try:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        betweenness = {i: 0.0 for i in range(n)}
    try:
        clustering = nx.clustering(G)
    except Exception:
        clustering = {i: 0.0 for i in range(n)}

    rows = []
    for i in range(n):
        label = i + 1
        name_row = label_df[label_df["index"] == label]
        name = str(name_row["name"].iloc[0]) if not name_row.empty else f"roi_{label:03d}"
        rows.append({
            "roi_index":        label,
            "name":             name,
            "degree":           degree.get(i, 0),
            "weighted_degree":  round(w_degree.get(i, 0.0), 3),
            "betweenness":      round(betweenness.get(i, 0.0), 6),
            "clustering":       round(clustering.get(i, 0.0), 6),
        })
    return pd.DataFrame(rows)
