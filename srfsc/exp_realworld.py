# Experiment script for the SRFSC algorithm and comparison to others using real-world data.
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx

from metrics import modularity, conductance, nmi_score, silhouette_score_custom
from metrics import demographic_parity, fairness_violation, consistency_score
from metrics import fairness_quality_tradeoff
from metrics import convert_clusters_to_dict
from fairsc_clustering import fair_clustering_wk, find_best_k_cluster
from fairsc_clustering import fair_clustering_best_parameter

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from others.Ghodsi2024 import cluster_graph as ghodsi_cluster_graph
from others.kleindessner2019 import fair_SC_unnormalized as kleindessner_fair_SC_unnormalized

def plot_sil_score_for_number_of_clusters(A: np.ndarray,
                                          s: np.ndarray,
                                          num_clusters_range: Tuple[int]=range(2, 31),
                                          title: str="Silhouette score for number of clusters"):
    sil_scores = []
    for k in tqdm(num_clusters_range):
        x = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
        sil_scores.append(silhouette_score_custom(A, x.labels_))
    
    # mark the best k
    best_k = num_clusters_range[np.argmax(sil_scores)]
    best_score = max(sil_scores)
    plt.figure()
    plt.plot(best_k, best_score, 'ro')
    plt.text(best_k, best_score, f"Best k={best_k}", fontsize=12)
    plt.plot(num_clusters_range, sil_scores, marker='o')
    plt.title(title)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.grid()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
# run with ignoring UserWarning
import warnings
warnings.filterwarnings("ignore")

# On real data
from moreno_loader import load_moreno

print("Loading Moreno data...")
A, s = load_moreno()
G = nx.from_numpy_array(A)
print("Moreno data loaded.")
print("A shape:", A.shape)
print("s shape:", s.shape)

# plot_sil_score_for_number_of_clusters(A, s, title="Silhouette score for number of clusters on Moreno data")
best_k = 4
x_sc = SpectralClustering(n_clusters=best_k, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
x_kmeans = KMeans(n_clusters=best_k, n_init=10).fit(A)
x_ghodsi = ghodsi_cluster_graph(A, s, num_clusters=best_k)
x_kleindessner = kleindessner_fair_SC_unnormalized(A, k=2, sensitive=s)
# We assume spectral clustering is the best clustering method
x_act = x_sc.labels_

print(">> Modularity:")
print("Spectral Clustering:", modularity(G, convert_clusters_to_dict(x_sc.labels_)))
print("KMeans:", modularity(G, convert_clusters_to_dict(x_kmeans.labels_)))
print("Ghodsi:", modularity(G, convert_clusters_to_dict(x_ghodsi)))
print("Kleindessner:", modularity(G, convert_clusters_to_dict(x_kleindessner)))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: modularity(G, convert_clusters_to_dict(x)))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), modularity(G, convert_clusters_to_dict(x_srfsc_best.labels_)))

print(">> NMI:")
print("Spectral Clustering:", nmi_score(x_act, x_sc.labels_))
print("KMeans:", nmi_score(x_act, x_kmeans.labels_))
print("Ghodsi:", nmi_score(x_act, x_ghodsi))
print("Kleindessner:", nmi_score(x_act, x_kleindessner))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: nmi_score(x_act, x))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), nmi_score(x_act, x_srfsc_best.labels_))
    
print(">> Conductance:")
print("Spectral Clustering:", conductance(G, convert_clusters_to_dict(x_sc.labels_)))
print("KMeans:", conductance(G, convert_clusters_to_dict(x_kmeans.labels_)))
print("Ghodsi:", conductance(G, convert_clusters_to_dict(x_ghodsi)))
print("Kleindessner:", conductance(G, convert_clusters_to_dict(x_kleindessner)))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: conductance(G, convert_clusters_to_dict(x)))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), conductance(G, convert_clusters_to_dict(x_srfsc_best.labels_)))

print(">> Silhouette:")
print("Spectral Clustering:", silhouette_score_custom(A, x_sc.labels_))
print("KMeans:", silhouette_score_custom(A, x_kmeans.labels_))
print("Ghodsi:", silhouette_score_custom(A, x_ghodsi))
print("Kleindessner:", silhouette_score_custom(A, x_kleindessner))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: silhouette_score_custom(A, x))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), silhouette_score_custom(A, x_srfsc_best.labels_))

print(">> Demographic Parity:")
print("Spectral Clustering:", demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
print("KMeans:", demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
print("Ghodsi:", demographic_parity(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
print("Kleindessner:", demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: demographic_parity(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), demographic_parity(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

print(">> Fairness Violation:")
print("Spectral Clustering:", fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
print("KMeans:", fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
print("Ghodsi:", fairness_violation(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
print("Kleindessner:", fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: fairness_violation(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), fairness_violation(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

print(">> Consistency Score:")
print("Spectral Clustering:", consistency_score(G, x_sc.labels_, s))
print("KMeans:", consistency_score(G, x_kmeans.labels_, s))
print("Ghodsi:", consistency_score(G, x_ghodsi, s))
print("Kleindessner:", consistency_score(G, x_kleindessner, s))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: consistency_score(G, x, s))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), consistency_score(G, x_srfsc_best.labels_, s))


from facebook_loader import load_facebook

print("Loading Facebook data...")
A, s = load_facebook()
A = np.squeeze(np.asarray(A))
s = np.squeeze(np.asarray(s)).astype(int)
G = nx.from_numpy_array(A)
print("Facebook data loaded.")
print("A shape:", A.shape)
print("s shape:", s.shape)

# plot_sil_score_for_number_of_clusters(A, s, title="Silhouette score for number of clusters on Facebook data")
best_k = 2
x_sc = SpectralClustering(n_clusters=best_k, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
x_kmeans = KMeans(n_clusters=best_k, n_init=10).fit(A)
x_ghodsi = ghodsi_cluster_graph(A, s, num_clusters=best_k)
x_kleindessner = kleindessner_fair_SC_unnormalized(A, k=2, sensitive=s)
# We assume spectral clustering is the best clustering method
x_act = x_sc.labels_

print(">> Modularity:")
print("Spectral Clustering:", modularity(G, convert_clusters_to_dict(x_sc.labels_)))
print("KMeans:", modularity(G, convert_clusters_to_dict(x_kmeans.labels_)))
print("Ghodsi:", modularity(G, convert_clusters_to_dict(x_ghodsi)))
print("Kleindessner:", modularity(G, convert_clusters_to_dict(x_kleindessner)))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: modularity(G, convert_clusters_to_dict(x)))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), modularity(G, convert_clusters_to_dict(x_srfsc_best.labels_)))

print(">> NMI:")
print("Spectral Clustering:", nmi_score(x_act, x_sc.labels_))
print("KMeans:", nmi_score(x_act, x_kmeans.labels_))
print("Ghodsi:", nmi_score(x_act, x_ghodsi))
print("Kleindessner:", nmi_score(x_act, x_kleindessner))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: nmi_score(x_act, x))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), nmi_score(x_act, x_srfsc_best.labels_))
    
print(">> Conductance:")
print("Spectral Clustering:", conductance(G, convert_clusters_to_dict(x_sc.labels_)))
print("KMeans:", conductance(G, convert_clusters_to_dict(x_kmeans.labels_)))
print("Ghodsi:", conductance(G, convert_clusters_to_dict(x_ghodsi)))
print("Kleindessner:", conductance(G, convert_clusters_to_dict(x_kleindessner)))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: conductance(G, convert_clusters_to_dict(x)))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), conductance(G, convert_clusters_to_dict(x_srfsc_best.labels_)))

print(">> Silhouette:")
print("Spectral Clustering:", silhouette_score_custom(A, x_sc.labels_))
print("KMeans:", silhouette_score_custom(A, x_kmeans.labels_))
print("Ghodsi:", silhouette_score_custom(A, x_ghodsi))
print("Kleindessner:", silhouette_score_custom(A, x_kleindessner))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: silhouette_score_custom(A, x))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), silhouette_score_custom(A, x_srfsc_best.labels_))

print(">> Demographic Parity:")
print("Spectral Clustering:", demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
print("KMeans:", demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
print("Ghodsi:", demographic_parity(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
print("Kleindessner:", demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: demographic_parity(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), demographic_parity(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

print(">> Fairness Violation:")
print("Spectral Clustering:", fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
print("KMeans:", fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
print("Ghodsi:", fairness_violation(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
print("Kleindessner:", fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: fairness_violation(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), fairness_violation(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

print(">> Consistency Score:")
print("Spectral Clustering:", consistency_score(G, x_sc.labels_, s))
print("KMeans:", consistency_score(G, x_kmeans.labels_, s))
print("Ghodsi:", consistency_score(G, x_ghodsi, s))
print("Kleindessner:", consistency_score(G, x_kleindessner, s))
x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=best_k, metric=lambda x: consistency_score(G, x, s))
print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), consistency_score(G, x_srfsc_best.labels_, s))



# Very huge data, not recommended to run

# from pokec_loader import load_pokec
# print("Loading Pokec data...")
# A, s = load_pokec()
# print("Pokec data loaded.")
# print("A shape:", A.shape)
# print("s shape:", s.shape)

# plot_sil_score_for_number_of_clusters(A, s, title="Silhouette score for number of clusters on Pokec data")

