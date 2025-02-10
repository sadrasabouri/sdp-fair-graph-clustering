# -*- coding: utf-8 -*-
from typing import List
import networkx as nx
import numpy as np

def generate_synthetic_data_sbm_binary_s(n: List[int],
                                         p_a: List[List[float]],
                                         p_s: float=0.5,
                                         seed: int=0):
    """
    Generate synthetic data using networkx stochastic_block_model.

    :param n: list of number of nodes in each block
    :param p_a: node connection probability list
    :param p_s: probability for specify 
    :param seed: seed for random number generator
    :return: adjacency matrix, labels and node attributes
    """
    np.random.seed(seed)
    G = nx.stochastic_block_model(n, p_a, seed=seed)
    A = nx.adjacency_matrix(G).toarray()
    labels = np.array([l for l in nx.get_node_attributes(G, 'block').values()])
    s = np.random.binomial(1, p_s, len(labels))
    return G, A, labels, s

def generate_synthetic_data_sbm_binary_s_biased(n: List[int],
                                                p_a: List[List[float]],
                                                p_s_blocks: List[float],
                                                seed: int=0):
    """
    Generate synthetic data using networkx stochastic_block_model with biased s vector.

    :param n: list of number of nodes in each block
    :param p_a: node connection probability list
    :param p_s_blocks: list of probabilities for s=1 in each block
    :param seed: seed for random number generator
    :return: adjacency matrix, labels, and biased node attributes s
    """
    np.random.seed(seed)
    G = nx.stochastic_block_model(n, p_a, seed=seed)
    A = nx.adjacency_matrix(G).toarray()
    labels = np.array([l for l in nx.get_node_attributes(G, 'block').values()])
    
    # Generate s vector with different probabilities for each block
    s = np.zeros(len(labels), dtype=int)
    for block_idx, p_s in enumerate(p_s_blocks):
        block_nodes = np.where(labels == block_idx)[0]
        s[block_nodes] = np.random.binomial(1, p_s, len(block_nodes))
    
    return G, A, labels, s
