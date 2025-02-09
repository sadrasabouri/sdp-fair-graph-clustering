# -*- coding: utf-8 -*-
from typing import List, Tuple
from sklearn import metrics
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def clustering_eval(x: np.ndarray, y: np.ndarray):
    """
    Evaluate clustering results using different metrics.
    Metrics from https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

    :param x: clustering predicted vector
    :param y: clustering true vectors
    :return: dictionary of metrics
    """
    return {
        'rand': metrics.rand_score(y, x),
        'adjusted_rand': metrics.adjusted_rand_score(y, x),
        'adjusted_mutual_info': metrics.adjusted_mutual_info_score(y, x),
        'homogeneity': metrics.homogeneity_score(y, x),
        'completeness': metrics.completeness_score(y, x),
        'v_measure': metrics.v_measure_score(y, x),
        'fowlkes_mallows': metrics.fowlkes_mallows_score(y, x),
    }


def show_graph_with_colors(A: np.ndarray,
                           colors: List=[],
                           labels: List=[],
                           seed: int=0):
    """
    Show a graph with colors and labels.

    :param A: Adjacency matrix
    :param colors: list of colors
    :param labels: list of labels
    :param seed: seed for the layout
    :return: matplotlib.pyplot object
    """
    rows, cols = np.where(A == 1)
    edges = zip(rows.tolist(), cols.tolist())

    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=seed)  # positions for all nodes
    for color in set(colors):
        nx.draw_networkx_nodes(G,
                              pos,
                              nodelist=[i for i, c in enumerate(colors) if c==color],
                              node_color=color)
        nx.draw_networkx_nodes(G,
                              pos,
                              nodelist=[i for i, c in enumerate(colors) if c==color],
                              node_color=color)

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    labels = {i: label for i, label in enumerate(labels)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, alpha=0.5)
    return plt


def show_weighted_graph_with_colors(A: np.ndarray,
                                    colors: List=[],
                                    labels: List=[],
                                    seed: int=0):
    """
    Show a weighted graph with colors and labels.

    :param A: Adjacency matrix
    :param colors: list of colors
    :param labels: list of labels
    :param seed: seed for the layout
    :return: matplotlib.pyplot object
    """
    G = nx.Graph()
    n, m = A.shape
    for i in range(n):
        for j in range(i + 1, m):
            w = A[i, j]
            if w > 0:
                G.add_edge(i, j, weight=w, len=100/w)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    labels = {i: label for i, label in enumerate(labels)}

    pos = nx.spring_layout(G, seed=seed)
    fig, ax = plt.subplots()

    nx.draw(G, pos, width=weights, ax=ax)
    for color in set(colors):
        nx.draw_networkx_nodes(G,
                              pos,
                              ax=ax,
                              nodelist=[i for i, c in enumerate(colors) if c==color],
                              node_color=color)
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=15, font_color="whitesmoke")
    return plt


def get_random_sym_nonloop_adj(n: int, p: float, seed: int=0):
    """
    Return a random symmetric non-loop adjacency matrix.

    :param n: size of the matrix
    :param p: connection probability
    :return: adjacency matrix
    """
    np.random.seed(seed)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            is_connected = int((np.random.rand() < p) * 1)
            A[i, j] = is_connected
            A[j, i] = is_connected
    return A


def get_sparse_matrix(p: float, size: Tuple[int, int]):
    """
    Return a sparse matrix with connection probability p.

    :param p: inter-cluster connection probablity
    :param size: size of the matrix
    :return: sparse matrix
    """
    return np.random.binomial(
        1,
        [[p for i in range(size[1])] for j in range(size[0])],
        size=size)


def get_random_sym_nonloop_adj_p_range(n, p):
    """
    Return a random symmetric non-loop adjacency matrix with connection probabilities as a range.

    :param n: size of the matrix
    :param p: connection probability range
    :return: adjacency matrix
    """
    A = []
    for i in range(n):
        A.append([*[0]*i, *np.random.uniform(*p, (n-i,))])
    A = np.array(A)

    for i in range(n):
        for j in range(i+1, n):
            A[j, i] = A[i, j]
    return A


def get_random_inter_cluster(n, m, p):
    """
    Return a random inter-cluster connection matrix.

    :param n: number of rows
    :param m: number of columns
    :param p: connection probability range
    :return: matrix
    """
    return np.random.uniform(*p, (n, m))
