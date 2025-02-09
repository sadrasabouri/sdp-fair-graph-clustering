# This is the same implementation as the one in the original paper: https://arxiv.org/pdf/2402.10756
# we only replace pytorch with numpy for more accessibility and simplicity.
import numpy as np
import scipy as sp
from scipy.spatial import distance_matrix
from scipy.linalg import null_space, eigh, sqrtm


def cluster_graph(adj_matrix: np.ndarray,
                  sensitive_vector: np.ndarray,
                  num_clusters: int,
                  method: str = "individual_fair") -> np.ndarray:
    """
    Clusters the given graph using the specified fairness-aware spectral clustering method.

    :param adj_matrix: (num_nodes, num_nodes) Adjacency matrix of the graph
    :param sensitive_vector: (num_nodes,) Vector indicating sensitive group memberships
    :param num_clusters: Number of clusters to discover
    :param method: Clustering method to use ("individual_fair", "group_fair", "normal")
    :return: (num_nodes,) Cluster assignments
    """
    if method == "individual_fair":
        return ind_fair_sc(adj_matrix, sensitive_vector, num_clusters)
    elif method == "group_fair":
        return group_fair_sc(adj_matrix, sensitive_vector, num_clusters)
    elif method == "normal":
        return normal_sc(adj_matrix, num_clusters)
    else:
        raise ValueError("Invalid method. Choose from 'individual_fair', 'group_fair', or 'normal'.")

def similarity_constraint(A, F, k):
    n = A.shape[0]
    D = distance_matrix(F, F)
    R = D.copy()
    
    for i in range(n):
        R[i, :][np.argsort(R[i, :])[0:n - k]] = 0
        R[:, i][np.argsort(R[:, i])[0:n - k]] = 0
    return R

def compute_R0(sensitive):
    n = len(sensitive)
    sens_unique = np.unique(sensitive)
    h = len(sens_unique)
    group_one_hot = np.eye(h)[sensitive, :]
    similarity_matrix = np.matmul(group_one_hot, group_one_hot.T)
    R = similarity_matrix - np.eye(n)
    return R

def compute_RS(sensitive):
    R = compute_R0(sensitive)
    return R / R.sum(axis=1, keepdims=True)

def compute_RD(sensitive):
    n = len(sensitive)
    sens_unique = np.unique(sensitive)
    h = len(sens_unique)
    group_one_hot = np.eye(h)[sensitive, :]
    R = 1 - np.matmul(group_one_hot, group_one_hot.T)
    return R / R.sum(axis=1, keepdims=True)

def joint_Laplacian(groups):
    RS = compute_RS(groups)
    RD = compute_RD(groups)
    R = RD - RS
    L = np.diag(np.sum(R, axis=1)) - R
    Lp = (np.abs(L) + L) / 2
    Ln = (np.abs(L) - L) / 2
    return L, Ln, Lp

def compute_laplacian(A, normalize=False):
    D = np.diag(A.sum(axis=1))
    L = D - A
    if normalize:
        D_inv_sqrt = np.linalg.inv(sqrtm(D))
        L = np.matmul(D_inv_sqrt, np.matmul(L, D_inv_sqrt))
    return L

def compute_top_eigen(L, k):
    _, eigvecs = eigh(L, subset_by_index=[0, k - 1])
    return eigvecs

def kmeans(X, k, normalize):
    from sklearn.cluster import KMeans
    if normalize:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return KMeans(n_clusters=k, n_init=10).fit_predict(X)

def ind_fair_sc(A, groups, k, normalize_laplacian=False, normalize_evec=False):
    R = compute_RD(groups)
    Z = null_space(R)
    L = compute_laplacian(A, normalize_laplacian)
    LL = np.matmul(Z.T, np.matmul(L, Z))
    Y = compute_top_eigen(LL, k)
    YY = np.matmul(Z, Y)
    return kmeans(YY, k, normalize_evec)

def group_fair_sc(A, groups, k, normalize_laplacian=False, normalize_evec=False):
    Fair_Mat = compute_R0(groups)
    Z = null_space(Fair_Mat.T)
    L = compute_laplacian(A, normalize_laplacian)
    LL = np.matmul(Z.T, np.matmul(L, Z))
    LL = (LL + LL.T) / 2
    Y = compute_top_eigen(LL, k)
    YY = np.matmul(Z, Y)
    return kmeans(YY, k, normalize_evec)

def normal_sc(A, k, normalize_laplacian=False, normalize_evec=False):
    L = compute_laplacian(A, normalize_laplacian)
    Y = compute_top_eigen(L, k)
    return kmeans(Y, k, normalize_evec)

if __name__ == "__main__":
    # Test the implementation
    A = np.array([[0, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 1, 0, 1],
                  [0, 0, 1, 0]])
    sensitive_vector = np.array([0, 1, 1, 0])
    num_clusters = 2

    cluster_assignments = cluster_graph(A, sensitive_vector, num_clusters, method="individual_fair")
    print(cluster_assignments)
