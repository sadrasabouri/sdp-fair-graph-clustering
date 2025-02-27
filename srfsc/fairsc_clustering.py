# -*- coding: utf-8 -*-
from typing import Callable
import itertools
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt
import numpy as np


def _get_fair_normalized_adj_matrix(A: np.ndarray,
                                    s: np.ndarray,
                                    mu: float=0.1,
                                    y: float=0.1):
    """
    Adds fairness regulation and normalizes the adjacency matrix.

    :param A: Adjacency matrix
    :param s: Specifity vector
    :param mu: mu paramter in the equation
    :param y: lambda paramter in the equation
    :return: normalized adjacency matrix
    """
    assert A.shape[0] == s.shape[0] and A.shape[1] == s.shape[0], "s should be a vector of size A's rows and columns"

    n = A.shape[0]
    s_ = s.reshape((n, 1))

    ss_T = s_ @ s_.T
    ones = np.ones((n, n))
    A_ = A - y * ss_T - mu * ones

    A_norm = (A_ - A_.min()) / (A_.max() - A_.min())
    return A_norm


def fair_clustering_best_parameter(A: np.ndarray,
                                   s: np.ndarray,
                                   metric: Callable,
                                   mu_range: list=np.arange(0, 1, 0.2),
                                   y_range: list=np.arange(0, 10, 0.1),
                                   k: int=2):
    """
    Fair clustering with best parameters using greedy search.

    :param A: Adjacency matrix
    :param s: Specifity vector
    :param mu_range: mu range
    :param y_range: y range
    :param k: number of clusters
    :param metric: metric to use for evaluation
    :param x_act: actual cluster labels
    :return: cluster object with best parameters
    """
    best_score = -1
    best_params = None
    best_cluster = None

    scores = []
    for mu, y in tqdm(itertools.product(mu_range, y_range)):
        x = fair_clustering_wk(A, s, mu=mu, y=y, k=k)
        score = metric(x.labels_)
        scores.append({
            "score:":score,
            "mu": mu,
            "y": y
            })
        if score > best_score:
            best_score = score
            best_params = (float(mu), float(y))
            best_cluster = x
    return best_cluster, {"mu": best_params[0], "y": best_params[1]}, best_score, scores


def fair_clustering_wk(A: np.ndarray,
                       s: np.ndarray,
                       mu: float=0.1,
                       y: float=0.1,
                       k: int=2):
    """
    Cluster a graph given it's adjacancy matrix into k clusters.
    It's solving \max_x x^T (A - λ s s^T - \mu 1 1^T) x [s.t ~ x \in \{-1, 1\}^n] using SpectralClustering.
    https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.SpectralClustering.html

    :param A: Adjacency matrix
    :param s: Specifity vector
    :param mu: mu paramter in the equation
    :param y: lambda paramter in the equation
    :param k: number of clusters
    :return: cluster object
    """
    assert A.shape[0] == s.shape[0] and A.shape[1] == s.shape[0], "s should be a vector of size A's rows and columns"
    A_ = _get_fair_normalized_adj_matrix(A, s, mu, y)
    A_ = np.asarray(A_)
    x = SpectralClustering(n_clusters=k,
                           affinity='precomputed',
                           assign_labels='discretize',
                           random_state=0).fit(A_)
    return x


def find_best_k_cluster(A: np.ndarray,
                        s: np.ndarray,
                        mu: float=0.1,
                        y: float=0.1,
                        gamma: float=1,
                        stop_k: int=10):
    """
    Cluster a graph given it's adjacency matrix into k clusters automatically finding k from elbow in Silhouette score graph.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

    :param A: Adjacency matrix
    :param s: Specifity vector
    :param mu: mu paramter in the equation
    :param y: lambda paramter in the equation
    :param gamma: gamma paramter for converting affinity matrix to distance matrix
    :param stop_k: maximum number of clusters to try
    :return: the best cluster object
    """
    clusterings = {}
    scores = []

    for k in range(2, stop_k):
        x = fair_clustering_wk(A, s, mu=mu, y=y, k=k)
        A_ = _get_fair_normalized_adj_matrix(A, s, mu, y)
        D = np.exp(-gamma * A_)
        np.fill_diagonal(D, 0)

        score = silhouette_score(D, x.labels_, metric="precomputed")
        scores.append(score)
        clusterings[k] = x
    plt.plot(range(2, stop_k), scores, marker='o')
    plt.plot(range(2, stop_k), scores, '.-')
    plt.title(f"Clustering for $A' = A - {mu}I - {y} ss^T$")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.show()
    return clusterings, scores
