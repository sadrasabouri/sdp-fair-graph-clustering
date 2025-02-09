# The original code is from https://github.com/matthklein/fair_spectral_clustering, 
# we convert it to Python script using LLMs and verify the final code by manually checking the code.
import numpy as np
import pandas as pd
import networkx as nx
import requests
from sklearn.cluster import KMeans

def null_space(A):
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    tol = max(A.shape) * np.spacing(np.max(s))
    rank = np.sum(s > tol)
    return Vh[rank:].T

def clustering_accuracy(labels, clustering):
    n = len(labels)

    # Reshape to vectors
    labels = np.array(labels).reshape(-1)
    clustering = np.array(clustering).reshape(-1)

    aa = np.unique(labels)
    J = len(aa)
    bb = np.unique(clustering)
    K = len(bb)

    # Remap labels if needed
    if not np.array_equal(np.sort(aa), np.arange(1, J+1)):
        labels_old = labels.copy()
        for i, ell in enumerate(aa, 1):
            labels[labels_old == ell] = i

    if not np.array_equal(np.sort(bb), np.arange(1, K+1)):
        clustering_old = clustering.copy()
        for i, ell in enumerate(bb, 1):
            clustering[clustering_old == ell] = i

    # Try all permutations
    min_error = float('inf')
    clustering_temp = clustering.copy()

    from itertools import permutations
    for perm in permutations(range(1, max(K, J) + 1)):
        for m in range(K):
            clustering_temp[clustering == m+1] = perm[m]

        error_temp = np.sum(clustering_temp != labels) / n
        min_error = min(min_error, error_temp)

    return min_error

def generate_adja_SB_model(n, a, b, c, d, k, h, block_sizes):
    """
    Exact implementation of their MATLAB generate_adja_SB_model function
    """
    # Input validation
    if np.sum(block_sizes) != n or len(block_sizes) != (k*h):
        raise ValueError('wrong input')

    # Initialize with probability d
    adja = np.random.binomial(1, d, (n, n))

    # Compute cumulative sums for indexing
    cum_block_sizes = np.cumsum(np.concatenate(([0], block_sizes)))

    # Fill in blocks according to the model
    for ell in range(1, k+1):
        for mmm in range(1, k+1):
            for ggg in range(1, h+1):
                for fff in range(1, h+1):
                    if ell == mmm:  # Same cluster
                        if ggg == fff:  # Same group
                            start_row = cum_block_sizes[(ell-1)*h+(ggg-1)]
                            end_row = cum_block_sizes[(ell-1)*h+ggg]
                            start_col = cum_block_sizes[(ell-1)*h+(ggg-1)]
                            end_col = cum_block_sizes[(ell-1)*h+ggg]

                            size_i = block_sizes[(ell-1)*h+(ggg-1)]
                            size_j = block_sizes[(ell-1)*h+(ggg-1)]

                            adja[start_row:end_row, start_col:end_col] = np.random.binomial(1, a, (size_i, size_j))

                        else:  # Same cluster, different group
                            start_row = cum_block_sizes[(ell-1)*h+(ggg-1)]
                            end_row = cum_block_sizes[(ell-1)*h+ggg]
                            start_col = cum_block_sizes[(ell-1)*h+(fff-1)]
                            end_col = cum_block_sizes[(ell-1)*h+fff]

                            size_i = block_sizes[(ell-1)*h+(ggg-1)]
                            size_j = block_sizes[(ell-1)*h+(fff-1)]

                            adja[start_row:end_row, start_col:end_col] = np.random.binomial(1, c, (size_i, size_j))

                    else:  # Different cluster
                        if ggg == fff:  # Same group
                            start_row = cum_block_sizes[(ell-1)*h+(ggg-1)]
                            end_row = cum_block_sizes[(ell-1)*h+ggg]
                            start_col = cum_block_sizes[(mmm-1)*h+(ggg-1)]
                            end_col = cum_block_sizes[(mmm-1)*h+ggg]

                            size_i = block_sizes[(ell-1)*h+(ggg-1)]
                            size_j = block_sizes[(mmm-1)*h+(ggg-1)]

                            adja[start_row:end_row, start_col:end_col] = np.random.binomial(1, b, (size_i, size_j))

    # Make symmetric using upper triangular part
    A = np.triu(adja, 1)
    A = A + A.T
    return A

def standard_SC_unnormalized(adj, k):
    n = adj.shape[0]
    degrees = np.sum(adj, axis=1)
    D = np.diag(degrees)
    L = D - adj
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[:k]
    H = eigvecs[:, idx]
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(H)
    return labels + 1

def fair_SC_unnormalized(adj, k, sensitive):
    n = adj.shape[0]
    sens_unique = np.unique(sensitive)
    h = len(sens_unique)

    sensitiveNEW = sensitive.copy()
    temp = 1
    for ell in sens_unique:
        sensitiveNEW[sensitive == ell] = temp
        temp += 1

    F = np.zeros((n, h-1))
    for ell in range(h-1):
        temp = (sensitiveNEW == ell+1)
        F[temp, ell] = 1
        groupSize = np.sum(temp)
        F[:, ell] = F[:, ell] - groupSize/n

    degrees = np.sum(adj, axis=1)
    D = np.diag(degrees)
    L = D - adj

    Z = null_space(F.T)
    Msymm = Z.T @ L @ Z
    Msymm = (Msymm + Msymm.T)/2

    eigvals, Y = np.linalg.eigh(Msymm)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    Y = Y[:, idx[:k]]

    for i in range(k):
        if abs(eigvals[i]) > 1e-10:
            Y[:, i] = Y[:, i] / np.sqrt(abs(eigvals[i]))

    H = Z @ Y
    norms = np.sqrt(np.sum(H * H, axis=1))
    H = H / norms[:, np.newaxis]

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(H)
    return labels + 1

def run_test(n_runs=10):
    """
    Run complete test with multiple trials
    """
    n = 1000
    k = 5
    h = 5
    a = 0.4
    b = 0.3
    c = 0.2
    d = 0.1

    fair_errors = []

    print(f"\nRunning {n_runs} trials...")
    print(f"Parameters: n={n}, k={k}, h={h}")
    print(f"Probabilities: a={a}, b={b}, c={c}, d={d}")

    for run in range(n_runs):
        print(f"\nTrial {run+1}/{n_runs}")

        # Generate block sizes
        block_size = n // (k * h)
        block_sizes = np.ones(k*h, dtype=int) * block_size

        # Generate ground truth labels and sensitive attributes
        sensitive = np.zeros(n, dtype=int)
        labels = np.zeros(n, dtype=int)
        for y in range(k):
            for z in range(h):
                start_idx = (n//k)*y + (n//(k*h))*z
                end_idx = (n//k)*y + (n//(k*h))*(z+1)
                sensitive[start_idx:end_idx] = z + 1
                labels[start_idx:end_idx] = y + 1

        # Generate adjacency matrix
        adj = generate_adja_SB_model(n, a, b, c, d, k, h, block_sizes)

        # Run clustering
        cluster_labels = fair_SC_unnormalized(adj, k, sensitive)

        # Calculate error
        error = clustering_accuracy(labels, cluster_labels)
        fair_errors.append(error)

        print(f"Error: {error:.3f}")

        # Print cluster sizes and group distributions
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        print("\nCluster distributions:")
        for cluster, count in zip(unique_clusters, counts):
            print(f"Cluster {cluster}: {count} nodes")
            for group in range(1, h+1):
                group_count = np.sum((cluster_labels == cluster) & (sensitive == group))
                print(f"  Group {group}: {group_count} nodes ({group_count/count*100:.1f}%)")

    avg_error = np.mean(fair_errors)
    std_error = np.std(fair_errors)
    print(f"\nFinal Results:")
    print(f"Average Error: {avg_error:.3f} ± {std_error:.3f}")

    return fair_errors

def run_test2():
    # Parameters
    n = 1000
    k = 5
    h = 5
    a = 0.4
    b = 0.3
    c = 0.2
    d = 0.1

    # Generate block sizes
    block_size = n // (k * h)
    block_sizes = np.ones(k*h, dtype=int) * block_size

    # Generate ground truth labels and sensitive attributes
    sensitive = np.zeros(n, dtype=int)
    labels = np.zeros(n, dtype=int)
    for y in range(k):
        for z in range(h):
            start_idx = (n//k)*y + (n//(k*h))*z
            end_idx = (n//k)*y + (n//(k*h))*(z+1)
            sensitive[start_idx:end_idx] = z + 1
            labels[start_idx:end_idx] = y + 1

    # Generate adjacency matrix
    adj = generate_adja_SB_model(n, a, b, c, d, k, h, block_sizes)

    # Run standard SC
    standard_labels = standard_SC_unnormalized(adj, k)
    standard_error = clustering_accuracy(labels, standard_labels)
    print(f"Standard SC Error: {standard_error:.3f}")

    # Run fair SC
    fair_labels = fair_SC_unnormalized(adj, k, sensitive)
    fair_error = clustering_accuracy(labels, fair_labels)
    print(f"Fair SC Error: {fair_error:.3f}")

def load_highschool_data():
    """
    Downloads and processes the high school friendship network data
    """
    metadata_url = "http://www.sociopatterns.org/wp-content/uploads/2015/09/metadata_2013.txt"
    friendship_url = "http://www.sociopatterns.org/wp-content/uploads/2015/07/Friendship-network_data_2013.csv.gz"

    print("Downloading metadata...")
    response = requests.get(metadata_url)
    metadata_text = response.text

    # Process metadata
    metadata_lines = metadata_text.strip().split('\n')
    metadata_list = []
    for line in metadata_lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] in ['M', 'F']:  # Only include M/F students
            metadata_list.append({
                'ID': int(parts[0]),
                'class': parts[1],
                'gender': parts[2]
            })

    metadata_df = pd.DataFrame(metadata_list)

    # Try to get friendship data
    try:
        print("Downloading friendship network data...")
        friendship_df = pd.read_csv(friendship_url, sep=' ', names=['student1', 'student2'])
    except:
        print("Could not download friendship data, using class-based connections")
        # Create edges based on class membership (fallback)
        G = create_class_based_network(metadata_df)
    else:
        # Create graph from actual friendship data
        G = create_friendship_network(metadata_df, friendship_df)

    # Get largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc)

    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()

    # Create sensitive attribute vector (gender)
    node_list = list(G.nodes())
    sensitive = np.array([1 if G.nodes[node]['gender'] == 'F' else 0 for node in node_list])

    return adj_matrix, sensitive, G

def create_class_based_network(metadata_df):
    G = nx.Graph()
    for _, row in metadata_df.iterrows():
        G.add_node(row['ID'], gender=row['gender'])
    for _, row1 in metadata_df.iterrows():
        for _, row2 in metadata_df.iterrows():
            if row1['class'] == row2['class'] and row1['ID'] != row2['ID']:
                G.add_edge(row1['ID'], row2['ID'])
    return G

def create_friendship_network(metadata_df, friendship_df):
    G = nx.Graph()
    for _, row in metadata_df.iterrows():
        G.add_node(row['ID'], gender=row['gender'])
    for _, row in friendship_df.iterrows():
        if G.has_node(row['student1']) and G.has_node(row['student2']):
            G.add_edge(row['student1'], row['student2'])
    return G

def run_test3():
    adj_matrix, sensitive, G = load_highschool_data()

    # Number of clusters
    k = 5

    # Run standard SC
    standard_labels = standard_SC_unnormalized(adj_matrix, k)
    standard_error = clustering_accuracy(sensitive, standard_labels)
    print(f"Standard SC Error: {standard_error:.3f}")

    # Run fair SC
    fair_labels = fair_SC_unnormalized(adj_matrix, k, sensitive)
    fair_error = clustering_accuracy(sensitive, fair_labels)
    print(f"Fair SC Error: {fair_error:.3f}")

if __name__ == "__main__":
    run_test()
    run_test2()
    run_test3()
