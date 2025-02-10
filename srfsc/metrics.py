# -*- coding: utf-8 -*-
# Author: ChatGPT, manually checked.

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.metrics import pairwise_distances

import networkx as nx

def demographic_parity(clusters: dict, sensitive_attrs: dict):
    """
    Computes the demographic parity balance across clusters.
    
    clusters: dict, {node_id: cluster_id}
    sensitive_attrs: dict, {node_id: group_id} (e.g., 0 for one group, 1 for another)
    
    Returns: Balance score (1 = perfectly balanced, <1 = imbalanced)
    """
    cluster_groups = {}
    
    for node, cluster in clusters.items():
        if cluster not in cluster_groups:
            cluster_groups[cluster] = []
        cluster_groups[cluster].append(sensitive_attrs[node])
    
    min_ratio, max_ratio = float('inf'), 0
    
    for members in cluster_groups.values():
        group_counts = np.bincount(members, minlength=2)
        total = sum(group_counts)
        if total == 0:
            continue
        ratio = group_counts[0] / total  # Adjust index for multi-group scenarios
        min_ratio = min(min_ratio, ratio)
        max_ratio = max(max_ratio, ratio)
    
    return min_ratio / max_ratio if max_ratio > 0 else 1

def fairness_violation(clusters:dict, sensitive_attrs:dict):
    """
    Measures the deviation of group representation in clusters from the overall distribution.
    
    Returns: Fairness violation score (lower is better)
    """
    total_counts = np.bincount(list(sensitive_attrs.values()))
    total_ratio = total_counts / total_counts.sum()
    
    cluster_groups = {}
    
    for node, cluster in clusters.items():
        if cluster not in cluster_groups:
            cluster_groups[cluster] = []
        cluster_groups[cluster].append(sensitive_attrs[node])
    
    violations = []
    for members in cluster_groups.values():
        group_counts = np.bincount(members, minlength=len(total_counts))
        cluster_ratio = group_counts / group_counts.sum()
        violations.append(np.abs(cluster_ratio - total_ratio).sum())
    
    return np.mean(violations)

def consistency_score(graph: nx.Graph, clusters: dict, sensitive_attrs: dict):
    """
    Computes the consistency of clusters based on sensitive attributes.
    
    graph: networkx.Graph object
    clusters: dict {node_id: cluster_id}
    sensitive_attrs: dict {node_id: group_id}
    
    Returns: Consistency score (higher is better)
    """
    total_score = 0
    for u, v in graph.edges():
        if clusters[u] == clusters[v]:  # If nodes are in the same cluster
            total_score += (sensitive_attrs[u] == sensitive_attrs[v])
    
    return total_score / len(graph.edges())


def modularity(graph: nx.Graph, clusters: dict):
    """
    Computes modularity score for the given clustering.
    
    Returns: Modularity score (higher is better)
    """
    return nx.algorithms.community.quality.modularity(graph, 
                [{node for node, c in clusters.items() if c == cluster} for cluster in set(clusters.values())])

def nmi_score(true_labels: np.ndarray, predicted_labels: np.ndarray):
    """
    Computes Normalized Mutual Information (NMI) score.
    
    Returns: NMI score (higher is better)
    """
    if None in true_labels or None in predicted_labels:
        return 0
    return normalized_mutual_info_score(true_labels, predicted_labels)

def silhouette_score_custom(data, cluster_labels):
    """
    Computes silhouette score to measure clustering separation.
    
    Returns: Silhouette score (-1 to 1, higher is better)
    """
    if len(set(cluster_labels)) == 1:
        return -1
    return silhouette_score(pairwise_distances(data), cluster_labels)

def conductance(graph, clusters):
    """
    Computes conductance for each cluster and returns the average.
    
    Returns: Conductance score (lower is better)
    """
    conductance_values = []
    for cluster in set(clusters.values()):
        nodes = {node for node, c in clusters.items() if c == cluster}
        cut_edges = sum(1 for u, v in graph.edges() if (u in nodes) != (v in nodes))
        volume = sum(graph.degree(n) for n in nodes)
        conductance_values.append(cut_edges / volume if volume > 0 else 1)
    
    return np.mean(conductance_values)

def fairness_quality_tradeoff(fairness_score, quality_score, alpha=0.5):
    """
    Computes a tradeoff metric combining fairness and clustering quality.
    
    fairness_score: A normalized fairness metric (higher is better)
    quality_score: A normalized clustering quality metric (higher is better)
    alpha: Trade-off parameter (0.5 = equal weight, <0.5 favors fairness)
    
    Returns: Tradeoff score (higher is better)
    """
    return alpha * quality_score + (1 - alpha) * fairness_score

def convert_clusters_to_dict(clusters):
    """
    Converts cluster assignments to a dictionary.
    
    Returns: {node_id: cluster_id}
    """
    return {i: c for i, c in enumerate(clusters)}

# main
if __name__ == '__main__':
    graph = nx.karate_club_graph()  # Example graph
    clusters = {n: n % 2 for n in graph.nodes}  # Example binary clustering
    sensitive_attrs = {n: n % 2 for n in graph.nodes}  # Example binary sensitive attribute

    # Compute fairness metrics
    print("Demographic Parity:", demographic_parity(clusters, sensitive_attrs))
    print("Fairness Violation:", fairness_violation(clusters, sensitive_attrs))
    print("Consistency Score:", consistency_score(graph, clusters, sensitive_attrs))

    # Compute clustering quality metrics
    print("Modularity:", modularity(graph, clusters))
    print("Conductance:", conductance(graph, clusters))

    # Compute tradeoff
    fairness_score = demographic_parity(clusters, sensitive_attrs)
    quality_score = modularity(graph, clusters)
    print("Fairness-Quality Tradeoff Score:", fairness_quality_tradeoff(fairness_score, quality_score))
