# -*- coding: utf-8 -*-
import numpy as np

def initialize_matrices(num_nodes: int) -> tuple:
    Z = np.zeros((num_nodes, num_nodes))
    P = np.zeros((num_nodes, num_nodes))
    Gamma = np.ones((num_nodes, num_nodes))
    return Z, P, Gamma

def update_Z(P: np.ndarray, Gamma: np.ndarray, balance_param: float, rho: float) -> np.ndarray:
    num_nodes = P.shape[0]
    Z = P + np.eye(num_nodes) - (1 / rho) * balance_param + (1 / rho) * Gamma
    np.fill_diagonal(Z, (np.diag(P) + 1 - (1 / rho) * balance_param + (1 / rho) * Gamma) / 2)
    return Z

def update_P(Z: np.ndarray, Gamma: np.ndarray, fairness_weight: float, rho: float) -> np.ndarray:
    U, S, Vt = np.linalg.svd(Z - Gamma / rho, full_matrices=False)
    S = np.maximum(0, S - fairness_weight / rho)
    P = U @ np.diag(S) @ Vt
    return P

def update_dual_variables(Z: np.ndarray, P: np.ndarray, Gamma: np.ndarray, alpha: float, rho: float) -> tuple:
    alpha += rho * (np.diag(Z) - 1)
    Gamma += rho * (P - Z)
    return alpha, Gamma

def fair_clustering_admm(adj_matrix: np.ndarray, sensitive_vector: np.ndarray, num_clusters: int, 
                          fairness_weight: float, balance_param: float, num_iterations: int) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    Z, P, Gamma = initialize_matrices(num_nodes)
    alpha = 1
    rho = 1.0  # ADMM parameter, can be tuned
    
    for _ in range(num_iterations):
        Z = update_Z(P, Gamma, balance_param, rho)
        P = update_P(Z, Gamma, fairness_weight, rho)
        alpha, Gamma = update_dual_variables(Z, P, Gamma, alpha, rho)
    
    return np.argmax(P, axis=1)  # Assign clusters based on P matrix

# Tests
def test_fair_clustering_admm():
    adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    sensitive_vector = np.array([0, 1, 0])
    num_clusters = 2
    fairness_weight = 0.5
    balance_param = 0.5
    num_iterations = 10
    
    cluster_assignments = fair_clustering_admm(adj_matrix, sensitive_vector, num_clusters, fairness_weight, balance_param, num_iterations)
    print(cluster_assignments)
    assert len(cluster_assignments) == adj_matrix.shape[0]
    assert set(cluster_assignments) <= set(range(num_clusters))

if __name__ == "__main__":
    test_fair_clustering_admm()
    print("All tests passed.")
