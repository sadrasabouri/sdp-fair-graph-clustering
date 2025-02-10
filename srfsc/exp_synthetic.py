# Experiment script for the SRFSC algorithm and comparison to others
import json

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

# On synthetic data
from synthetic_data import generate_synthetic_data_sbm_binary_s
from synthetic_data import generate_synthetic_data_sbm_binary_s_biased

#   Binary clustering uniform sensitive
for n in [100]:
    p_a = [[0.6, 0.4], [0.4, 0.6]]
    G, A, x_act, s = generate_synthetic_data_sbm_binary_s([n // 2, n // 2],
                                                          p_a, p_s=0.5, seed=10)
    # save all scores in a dictionary
    results = {}

    x_sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
    x_kmeans = KMeans(n_clusters=2, n_init=10).fit(A)
    x_ghodsi_if = ghodsi_cluster_graph(A, s, num_clusters=2, method="individual_fair")
    x_ghodsi_gf = ghodsi_cluster_graph(A, s, num_clusters=2, method="group_fair")
    x_kleindessner = kleindessner_fair_SC_unnormalized(A, k=2, sensitive=s)

    print(f"Setting - SBM BinaryClusters UniformS : n={n} p_a={p_a} p_s=0.5")
    print("---Metrics---")
    print(">> Modularity:")
    print("Spectral Clustering:", modularity(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", modularity(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi (IF):", modularity(G, convert_clusters_to_dict(x_ghodsi_if)))
    print("Ghodsi (GF):", modularity(G, convert_clusters_to_dict(x_ghodsi_gf)))
    print("Kleindessner:", modularity(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: modularity(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), modularity(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    results["Modularity"] = {
        "Spectral Clustering": modularity(G, convert_clusters_to_dict(x_sc.labels_)),
        "KMeans": modularity(G, convert_clusters_to_dict(x_kmeans.labels_)),
        "Ghodsi (IF)": modularity(G, convert_clusters_to_dict(x_ghodsi_if)),
        "Ghodsi (GF)": modularity(G, convert_clusters_to_dict(x_ghodsi_gf)),
        "Kleindessner": modularity(G, convert_clusters_to_dict(x_kleindessner)),
        "SRFSC": scores
    }

    print(">> NMI:")
    print("Spectral Clustering:", nmi_score(x_act, x_sc.labels_))
    print("KMeans:", nmi_score(x_act, x_kmeans.labels_))
    print("Ghodsi (IF):", nmi_score(x_act, x_ghodsi_if))
    print("Ghodsi (GF):", nmi_score(x_act, x_ghodsi_gf))
    print("Kleindessner:", nmi_score(x_act, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: nmi_score(x_act, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), nmi_score(x_act, x_srfsc_best.labels_))
    results["NMI"] = {
        "Spectral Clustering": nmi_score(x_act, x_sc.labels_),
        "KMeans": nmi_score(x_act, x_kmeans.labels_),
        "Ghodsi (IF)": nmi_score(x_act, x_ghodsi_if),
        "Ghodsi (GF)": nmi_score(x_act, x_ghodsi_gf),
        "Kleindessner": nmi_score(x_act, x_kleindessner),
        "SRFSC": scores
    }
    
    print(">> Conductance:")
    print("Spectral Clustering:", conductance(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", conductance(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi (IF):", conductance(G, convert_clusters_to_dict(x_ghodsi_if)))
    print("Ghodsi (GF):", conductance(G, convert_clusters_to_dict(x_ghodsi_gf)))
    print("Kleindessner:", conductance(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: conductance(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), conductance(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    results["Conductance"] = {
        "Spectral Clustering": conductance(G, convert_clusters_to_dict(x_sc.labels_)),
        "KMeans": conductance(G, convert_clusters_to_dict(x_kmeans.labels_)),
        "Ghodsi (IF)": conductance(G, convert_clusters_to_dict(x_ghodsi_if)),
        "Ghodsi (GF)": conductance(G, convert_clusters_to_dict(x_ghodsi_gf)),
        "Kleindessner": conductance(G, convert_clusters_to_dict(x_kleindessner)),
        "SRFSC": scores
    }

    print(">> Silhouette:")
    print("Spectral Clustering:", silhouette_score_custom(A, x_sc.labels_))
    print("KMeans:", silhouette_score_custom(A, x_kmeans.labels_))
    print("Ghodsi (IF):", silhouette_score_custom(A, x_ghodsi_if))
    print("Ghodsi (GF):", silhouette_score_custom(A, x_ghodsi_gf))
    print("Kleindessner:", silhouette_score_custom(A, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: silhouette_score_custom(A, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), silhouette_score_custom(A, x_srfsc_best.labels_))
    results["Silhouette"] = {
        "Spectral Clustering": silhouette_score_custom(A, x_sc.labels_),
        "KMeans": silhouette_score_custom(A, x_kmeans.labels_),
        "Ghodsi (IF)": silhouette_score_custom(A, x_ghodsi_if),
        "Ghodsi (GF)": silhouette_score_custom(A, x_ghodsi_gf),
        "Kleindessner": silhouette_score_custom(A, x_kleindessner),
        "SRFSC": scores
    }

    print(">> Demographic Parity:")
    print("Spectral Clustering:", demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi (IF):", demographic_parity(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)))
    print("Ghodsi (GF):", demographic_parity(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)))
    print("Kleindessner:", demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: demographic_parity(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), demographic_parity(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))
    results["Demographic Parity"] = {
        "Spectral Clustering": demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)),
        "KMeans": demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)),
        "Ghodsi (IF)": demographic_parity(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)),
        "Ghodsi (GF)": demographic_parity(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)),
        "Kleindessner": demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)),
        "SRFSC": scores
    }

    print(">> Fairness Violation:")
    print("Spectral Clustering:", fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi (IF):", fairness_violation(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)))
    print("Ghodsi (GF):", fairness_violation(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)))
    print("Kleindessner:", fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: -fairness_violation(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), fairness_violation(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))
    results["Fairness Violation"] = {
        "Spectral Clustering": fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)),
        "KMeans": fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)),
        "Ghodsi (IF)": fairness_violation(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)),
        "Ghodsi (GF)": fairness_violation(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)),
        "Kleindessner": fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)),
        "SRFSC": scores
    }

    print(">> Consistency Score:")
    print("Spectral Clustering:", consistency_score(G, x_sc.labels_, s))
    print("KMeans:", consistency_score(G, x_kmeans.labels_, s))
    print("Ghodsi (IF):", consistency_score(G, x_ghodsi_if, s))
    print("Ghodsi (GF):", consistency_score(G, x_ghodsi_gf, s))
    print("Kleindessner:", consistency_score(G, x_kleindessner, s))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: consistency_score(G, x, s))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), consistency_score(G, x_srfsc_best.labels_, s))
    results["Consistency Score"] = {
        "Spectral Clustering": consistency_score(G, x_sc.labels_, s),
        "KMeans": consistency_score(G, x_kmeans.labels_, s),
        "Ghodsi (IF)": consistency_score(G, x_ghodsi_if, s),
        "Ghodsi (GF)": consistency_score(G, x_ghodsi_gf, s),
        "Kleindessner": consistency_score(G, x_kleindessner, s),
        "SRFSC": scores
    }

    # save results
    with open("synthetic_data_results_BB.json", "w") as f:
        json.dump(results, f, indent=4)


#   Binary clustering with biased sensitive
for n in [100]:
    p_a = [[0.6, 0.4], [0.4, 0.6]]
    p_s_blocks = [0.25, 0.75]
    G, A, x_act, s = generate_synthetic_data_sbm_binary_s_biased([n // 2, n // 2],
                                                                p_a, p_s_blocks, seed=10)
    results = {}

    x_sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
    x_kmeans = KMeans(n_clusters=2, n_init=10).fit(A)
    x_ghodsi_if = ghodsi_cluster_graph(A, s, num_clusters=2, method="individual_fair")
    x_ghodsi_gf = ghodsi_cluster_graph(A, s, num_clusters=2, method="group_fair")
    x_kleindessner = kleindessner_fair_SC_unnormalized(A, k=2, sensitive=s)

    print("=====================================")
    print(f"Setting - SBM BinaryClusters BiasedS : n={n} p_a={p_a} p_s={p_s_blocks}")
    print("---Metrics---")
    print(">> Modularity:")
    print("Spectral Clustering:", modularity(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", modularity(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi (IF):", modularity(G, convert_clusters_to_dict(x_ghodsi_if)))
    print("Ghodsi (GF):", modularity(G, convert_clusters_to_dict(x_ghodsi_gf)))
    print("Kleindessner:", modularity(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: modularity(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), modularity(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    results["Modularity"] = {
        "Spectral Clustering": modularity(G, convert_clusters_to_dict(x_sc.labels_)),
        "KMeans": modularity(G, convert_clusters_to_dict(x_kmeans.labels_)),
        "Ghodsi (IF)": modularity(G, convert_clusters_to_dict(x_ghodsi_if)),
        "Ghodsi (GF)": modularity(G, convert_clusters_to_dict(x_ghodsi_gf)),
        "Kleindessner": modularity(G, convert_clusters_to_dict(x_kleindessner)),
        "SRFSC": scores
    }
    
    print(">> NMI:")
    print("Spectral Clustering:", nmi_score(x_act, x_sc.labels_))
    print("KMeans:", nmi_score(x_act, x_kmeans.labels_))
    print("Ghodsi (IF):", nmi_score(x_act, x_ghodsi_if))
    print("Ghodsi (GF):", nmi_score(x_act, x_ghodsi_gf))
    print("Kleindessner:", nmi_score(x_act, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: nmi_score(x_act, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), nmi_score(x_act, x_srfsc_best.labels_))
    results["NMI"] = {
        "Spectral Clustering": nmi_score(x_act, x_sc.labels_),
        "KMeans": nmi_score(x_act, x_kmeans.labels_),
        "Ghodsi (IF)": nmi_score(x_act, x_ghodsi_if),
        "Ghodsi (GF)": nmi_score(x_act, x_ghodsi_gf),
        "Kleindessner": nmi_score(x_act, x_kleindessner),
        "SRFSC": scores
    }
    
    print(">> Conductance:")
    print("Spectral Clustering:", conductance(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", conductance(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi (IF):", conductance(G, convert_clusters_to_dict(x_ghodsi_if)))
    print("Ghodsi (GF):", conductance(G, convert_clusters_to_dict(x_ghodsi_gf)))
    print("Kleindessner:", conductance(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: conductance(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), conductance(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    results["Conductance"] = {
        "Spectral Clustering": conductance(G, convert_clusters_to_dict(x_sc.labels_)),
        "KMeans": conductance(G, convert_clusters_to_dict(x_kmeans.labels_)),
        "Ghodsi (IF)": conductance(G, convert_clusters_to_dict(x_ghodsi_if)),
        "Ghodsi (GF)": conductance(G, convert_clusters_to_dict(x_ghodsi_gf)),
        "Kleindessner": conductance(G, convert_clusters_to_dict(x_kleindessner)),
        "SRFSC": scores
    }

    print(">> Silhouette:")
    print("Spectral Clustering:", silhouette_score_custom(A, x_sc.labels_))
    print("KMeans:", silhouette_score_custom(A, x_kmeans.labels_))
    print("Ghodsi (IF):", silhouette_score_custom(A, x_ghodsi_if))
    print("Ghodsi (GF):", silhouette_score_custom(A, x_ghodsi_gf))
    print("Kleindessner:", silhouette_score_custom(A, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: silhouette_score_custom(A, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), silhouette_score_custom(A, x_srfsc_best.labels_))
    results["Silhouette"] = {
        "Spectral Clustering": silhouette_score_custom(A, x_sc.labels_),
        "KMeans": silhouette_score_custom(A, x_kmeans.labels_),
        "Ghodsi (IF)": silhouette_score_custom(A, x_ghodsi_if),
        "Ghodsi (GF)": silhouette_score_custom(A, x_ghodsi_gf),
        "Kleindessner": silhouette_score_custom(A, x_kleindessner),
        "SRFSC": scores
    }

    print(">> Demographic Parity:")
    print("Spectral Clustering:", demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi (IF):", demographic_parity(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)))
    print("Ghodsi (GF):", demographic_parity(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)))
    print("Kleindessner:", demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: demographic_parity(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), demographic_parity(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))
    results["Demographic Parity"] = {
        "Spectral Clustering": demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)),
        "KMeans": demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)),
        "Ghodsi (IF)": demographic_parity(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)),
        "Ghodsi (GF)": demographic_parity(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)),
        "Kleindessner": demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)),
        "SRFSC": scores
    }

    print(">> Fairness Violation:")
    print("Spectral Clustering:", fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi (IF):", fairness_violation(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)))
    print("Ghodsi (GF):", fairness_violation(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)))
    print("Kleindessner:", fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: -fairness_violation(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), fairness_violation(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))
    results["Fairness Violation"] = {
        "Spectral Clustering": fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)),
        "KMeans": fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)),
        "Ghodsi (IF)": fairness_violation(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)),
        "Ghodsi (GF)": fairness_violation(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)),
        "Kleindessner": fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)),
        "SRFSC": scores
    }

    print(">> Consistency Score:")
    print("Spectral Clustering:", consistency_score(G, x_sc.labels_, s))
    print("KMeans:", consistency_score(G, x_kmeans.labels_, s))
    print("Ghodsi (IF):", consistency_score(G, x_ghodsi_if, s))
    print("Ghodsi (GF):", consistency_score(G, x_ghodsi_gf, s))
    print("Kleindessner:", consistency_score(G, x_kleindessner, s))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: consistency_score(G, x, s))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), consistency_score(G, x_srfsc_best.labels_, s))
    results["Consistency Score"] = {
        "Spectral Clustering": consistency_score(G, x_sc.labels_, s),
        "KMeans": consistency_score(G, x_kmeans.labels_, s),
        "Ghodsi (IF)": consistency_score(G, x_ghodsi_if, s),
        "Ghodsi (GF)": consistency_score(G, x_ghodsi_gf, s),
        "Kleindessner": consistency_score(G, x_kleindessner, s),
        "SRFSC": scores
    }

    # save results
    with open("synthetic_data_results_BU.json", "w") as f:
        json.dump(results, f, indent=4)

#   k classes clustering uniform sensitive
for n in [100]:

    p_a = [[0.6, 0.4, 0.3, 0.1, 0.01],
           [0.4, 0.6, 0.4, 0.1, 0.45],
           [0.3, 0.4, 0.6, 0.1, 0.01],
           [0.1, 0.1, 0.1, 0.6, 0.01],
           [0.01, 0.45, 0.01, 0.01, 0.6]]
    G, A, x_act, s = generate_synthetic_data_sbm_binary_s([2 * n // 5, n // 10, n // 10, n // 5, n // 5],
                                                          p_a, p_s=0.5, seed=10)
    results = {}
    x_sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
    x_kmeans = KMeans(n_clusters=2, n_init=10).fit(A)
    x_ghodsi_if = ghodsi_cluster_graph(A, s, num_clusters=2, method="individual_fair")
    x_ghodsi_gf = ghodsi_cluster_graph(A, s, num_clusters=2, method="group_fair")
    x_kleindessner = kleindessner_fair_SC_unnormalized(A, k=2, sensitive=s)

    print(f"Setting - SBM KClusters UniformS : n={n} p_a={p_a} p_s=0.5")
    print("---Metrics---")
    print(">> Modularity:")
    print("Spectral Clustering:", modularity(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", modularity(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi (IF):", modularity(G, convert_clusters_to_dict(x_ghodsi_if)))
    print("Ghodsi (GF):", modularity(G, convert_clusters_to_dict(x_ghodsi_gf)))
    print("Kleindessner:", modularity(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: modularity(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), modularity(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    results["Modularity"] = {
        "Spectral Clustering": modularity(G, convert_clusters_to_dict(x_sc.labels_)),
        "KMeans": modularity(G, convert_clusters_to_dict(x_kmeans.labels_)),
        "Ghodsi (IF)": modularity(G, convert_clusters_to_dict(x_ghodsi_if)),
        "Ghodsi (GF)": modularity(G, convert_clusters_to_dict(x_ghodsi_gf)),
        "Kleindessner": modularity(G, convert_clusters_to_dict(x_kleindessner)),
        "SRFSC": scores
    }
    
    print(">> NMI:")
    print("Spectral Clustering:", nmi_score(x_act, x_sc.labels_))
    print("KMeans:", nmi_score(x_act, x_kmeans.labels_))
    print("Ghodsi (IF):", nmi_score(x_act, x_ghodsi_if))
    print("Ghodsi (GF):", nmi_score(x_act, x_ghodsi_gf))
    print("Kleindessner:", nmi_score(x_act, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: nmi_score(x_act, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), nmi_score(x_act, x_srfsc_best.labels_))
    results["NMI"] = {
        "Spectral Clustering": nmi_score(x_act, x_sc.labels_),
        "KMeans": nmi_score(x_act, x_kmeans.labels_),
        "Ghodsi (IF)": nmi_score(x_act, x_ghodsi_if),
        "Ghodsi (GF)": nmi_score(x_act, x_ghodsi_gf),
        "Kleindessner": nmi_score(x_act, x_kleindessner),
        "SRFSC": scores
    }
    
    print(">> Conductance:")
    print("Spectral Clustering:", conductance(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", conductance(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi (IF):", conductance(G, convert_clusters_to_dict(x_ghodsi_if)))
    print("Ghodsi (GF):", conductance(G, convert_clusters_to_dict(x_ghodsi_gf)))
    print("Kleindessner:", conductance(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: conductance(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), conductance(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    results["Conductance"] = {
        "Spectral Clustering": conductance(G, convert_clusters_to_dict(x_sc.labels_)),
        "KMeans": conductance(G, convert_clusters_to_dict(x_kmeans.labels_)),
        "Ghodsi (IF)": conductance(G, convert_clusters_to_dict(x_ghodsi_if)),
        "Ghodsi (GF)": conductance(G, convert_clusters_to_dict(x_ghodsi_gf)),
        "Kleindessner": conductance(G, convert_clusters_to_dict(x_kleindessner)),
        "SRFSC": scores
    }

    print(">> Silhouette:")
    print("Spectral Clustering:", silhouette_score_custom(A, x_sc.labels_))
    print("KMeans:", silhouette_score_custom(A, x_kmeans.labels_))
    print("Ghodsi (IF):", silhouette_score_custom(A, x_ghodsi_if))
    print("Ghodsi (GF):", silhouette_score_custom(A, x_ghodsi_gf))
    print("Kleindessner:", silhouette_score_custom(A, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: silhouette_score_custom(A, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), silhouette_score_custom(A, x_srfsc_best.labels_))
    results["Silhouette"] = {
        "Spectral Clustering": silhouette_score_custom(A, x_sc.labels_),
        "KMeans": silhouette_score_custom(A, x_kmeans.labels_),
        "Ghodsi (IF)": silhouette_score_custom(A, x_ghodsi_if),
        "Ghodsi (GF)": silhouette_score_custom(A, x_ghodsi_gf),
        "Kleindessner": silhouette_score_custom(A, x_kleindessner),
        "SRFSC": scores
    }

    print(">> Demographic Parity:")
    print("Spectral Clustering:", demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi (IF):", demographic_parity(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)))
    print("Ghodsi (GF):", demographic_parity(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)))
    print("Kleindessner:", demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: demographic_parity(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), demographic_parity(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))
    results["Demographic Parity"] = {
        "Spectral Clustering": demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)),
        "KMeans": demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)),
        "Ghodsi (IF)": demographic_parity(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)),
        "Ghodsi (GF)": demographic_parity(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)),
        "Kleindessner": demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)),
        "SRFSC": scores
    }

    print(">> Fairness Violation:")
    print("Spectral Clustering:", fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi (IF):", fairness_violation(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)))
    print("Ghodsi (GF):", fairness_violation(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)))
    print("Kleindessner:", fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: -fairness_violation(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), fairness_violation(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))
    results["Fairness Violation"] = {
        "Spectral Clustering": fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)),
        "KMeans": fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)),
        "Ghodsi (IF)": fairness_violation(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)),
        "Ghodsi (GF)": fairness_violation(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)),
        "Kleindessner": fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)),
        "SRFSC": scores
    }

    print(">> Consistency Score:")
    print("Spectral Clustering:", consistency_score(G, x_sc.labels_, s))
    print("KMeans:", consistency_score(G, x_kmeans.labels_, s))
    print("Ghodsi (IF):", consistency_score(G, x_ghodsi_if, s))
    print("Ghodsi (GF):", consistency_score(G, x_ghodsi_gf, s))
    print("Kleindessner:", consistency_score(G, x_kleindessner, s))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: consistency_score(G, x, s))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), consistency_score(G, x_srfsc_best.labels_, s))
    results["Consistency Score"] = {
        "Spectral Clustering": consistency_score(G, x_sc.labels_, s),
        "KMeans": consistency_score(G, x_kmeans.labels_, s),
        "Ghodsi (IF)": consistency_score(G, x_ghodsi_if, s),
        "Ghodsi (GF)": consistency_score(G, x_ghodsi_gf, s),
        "Kleindessner": consistency_score(G, x_kleindessner, s),
        "SRFSC": scores
    }

    # save results
    with open("synthetic_data_results_5U.json", "w") as f:
        json.dump(results, f, indent=4)


#   k classes clustering with biased sensitive
for n in [100]:

    p_a = [[0.6, 0.4, 0.3, 0.1, 0.01],
           [0.4, 0.6, 0.4, 0.1, 0.45],
           [0.3, 0.4, 0.6, 0.1, 0.01],
           [0.1, 0.1, 0.1, 0.6, 0.01],
           [0.01, 0.45, 0.01, 0.01, 0.6]]
    p_s_blocks = [0.25, 0.75, 0.5, 0.25, 0.75]
    G, A, x_act, s = generate_synthetic_data_sbm_binary_s_biased([2 * n // 5, n // 10, n // 10, n // 5, n // 5],
                                                          p_a, p_s_blocks=p_s_blocks, seed=10)
    results = {}
    x_sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
    x_kmeans = KMeans(n_clusters=2, n_init=10).fit(A)
    x_ghodsi_if = ghodsi_cluster_graph(A, s, num_clusters=2, method="individual_fair")
    x_ghodsi_gf = ghodsi_cluster_graph(A, s, num_clusters=2, method="group_fair")
    x_kleindessner = kleindessner_fair_SC_unnormalized(A, k=2, sensitive=s)

    print(f"Setting - SBM KClusters UniformS : n={n} p_a={p_a} p_s={p_s_blocks}")
    print("---Metrics---")
    print(">> Modularity:")
    print("Spectral Clustering:", modularity(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", modularity(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi (IF):", modularity(G, convert_clusters_to_dict(x_ghodsi_if)))
    print("Ghodsi (GF):", modularity(G, convert_clusters_to_dict(x_ghodsi_gf)))
    print("Kleindessner:", modularity(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: modularity(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), modularity(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    results["Modularity"] = {
        "Spectral Clustering": modularity(G, convert_clusters_to_dict(x_sc.labels_)),
        "KMeans": modularity(G, convert_clusters_to_dict(x_kmeans.labels_)),
        "Ghodsi (IF)": modularity(G, convert_clusters_to_dict(x_ghodsi_if)),
        "Ghodsi (GF)": modularity(G, convert_clusters_to_dict(x_ghodsi_gf)),
        "Kleindessner": modularity(G, convert_clusters_to_dict(x_kleindessner)),
        "SRFSC": scores
    }
    
    print(">> NMI:")
    print("Spectral Clustering:", nmi_score(x_act, x_sc.labels_))
    print("KMeans:", nmi_score(x_act, x_kmeans.labels_))
    print("Ghodsi (IF):", nmi_score(x_act, x_ghodsi_if))
    print("Ghodsi (GF):", nmi_score(x_act, x_ghodsi_gf))
    print("Kleindessner:", nmi_score(x_act, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: nmi_score(x_act, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), nmi_score(x_act, x_srfsc_best.labels_))
    results["NMI"] = {
        "Spectral Clustering": nmi_score(x_act, x_sc.labels_),
        "KMeans": nmi_score(x_act, x_kmeans.labels_),
        "Ghodsi (IF)": nmi_score(x_act, x_ghodsi_if),
        "Ghodsi (GF)": nmi_score(x_act, x_ghodsi_gf),
        "Kleindessner": nmi_score(x_act, x_kleindessner),
        "SRFSC": scores
    }
    
    print(">> Conductance:")
    print("Spectral Clustering:", conductance(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", conductance(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi (IF):", conductance(G, convert_clusters_to_dict(x_ghodsi_if)))
    print("Ghodsi (GF):", conductance(G, convert_clusters_to_dict(x_ghodsi_gf)))
    print("Kleindessner:", conductance(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: conductance(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), conductance(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    results["Conductance"] = {
        "Spectral Clustering": conductance(G, convert_clusters_to_dict(x_sc.labels_)),
        "KMeans": conductance(G, convert_clusters_to_dict(x_kmeans.labels_)),
        "Ghodsi (IF)": conductance(G, convert_clusters_to_dict(x_ghodsi_if)),
        "Ghodsi (GF)": conductance(G, convert_clusters_to_dict(x_ghodsi_gf)),
        "Kleindessner": conductance(G, convert_clusters_to_dict(x_kleindessner)),
        "SRFSC": scores
    }

    print(">> Silhouette:")
    print("Spectral Clustering:", silhouette_score_custom(A, x_sc.labels_))
    print("KMeans:", silhouette_score_custom(A, x_kmeans.labels_))
    print("Ghodsi (IF):", silhouette_score_custom(A, x_ghodsi_if))
    print("Ghodsi (GF):", silhouette_score_custom(A, x_ghodsi_gf))
    print("Kleindessner:", silhouette_score_custom(A, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: silhouette_score_custom(A, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), silhouette_score_custom(A, x_srfsc_best.labels_))
    results["Silhouette"] = {
        "Spectral Clustering": silhouette_score_custom(A, x_sc.labels_),
        "KMeans": silhouette_score_custom(A, x_kmeans.labels_),
        "Ghodsi (IF)": silhouette_score_custom(A, x_ghodsi_if),
        "Ghodsi (GF)": silhouette_score_custom(A, x_ghodsi_gf),
        "Kleindessner": silhouette_score_custom(A, x_kleindessner),
        "SRFSC": scores
    }

    print(">> Demographic Parity:")
    print("Spectral Clustering:", demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi (IF):", demographic_parity(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)))
    print("Ghodsi (GF):", demographic_parity(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)))
    print("Kleindessner:", demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: demographic_parity(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), demographic_parity(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))
    results["Demographic Parity"] = {
        "Spectral Clustering": demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)),
        "KMeans": demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)),
        "Ghodsi (IF)": demographic_parity(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)),
        "Ghodsi (GF)": demographic_parity(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)),
        "Kleindessner": demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)),
        "SRFSC": scores
    }

    print(">> Fairness Violation:")
    print("Spectral Clustering:", fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi (IF):", fairness_violation(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)))
    print("Ghodsi (GF):", fairness_violation(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)))
    print("Kleindessner:", fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: -fairness_violation(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), fairness_violation(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))
    results["Fairness Violation"] = {
        "Spectral Clustering": fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)),
        "KMeans": fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)),
        "Ghodsi (IF)": fairness_violation(convert_clusters_to_dict(x_ghodsi_if), convert_clusters_to_dict(s)),
        "Ghodsi (GF)": fairness_violation(convert_clusters_to_dict(x_ghodsi_gf), convert_clusters_to_dict(s)),
        "Kleindessner": fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)),
        "SRFSC": scores
    }

    print(">> Consistency Score:")
    print("Spectral Clustering:", consistency_score(G, x_sc.labels_, s))
    print("KMeans:", consistency_score(G, x_kmeans.labels_, s))
    print("Ghodsi (IF):", consistency_score(G, x_ghodsi_if, s))
    print("Ghodsi (GF):", consistency_score(G, x_ghodsi_gf, s))
    print("Kleindessner:", consistency_score(G, x_kleindessner, s))
    x_srfsc_best, x_srfsc_params, x_srfsc_score, scores = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: consistency_score(G, x, s))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), consistency_score(G, x_srfsc_best.labels_, s))
    results["Consistency Score"] = {
        "Spectral Clustering": consistency_score(G, x_sc.labels_, s),
        "KMeans": consistency_score(G, x_kmeans.labels_, s),
        "Ghodsi (IF)": consistency_score(G, x_ghodsi_if, s),
        "Ghodsi (GF)": consistency_score(G, x_ghodsi_gf, s),
        "Kleindessner": consistency_score(G, x_kleindessner, s),
        "SRFSC": scores
    }

    # save results
    with open("synthetic_data_results_5B.json", "w") as f:
        json.dump(results, f, indent=4)
