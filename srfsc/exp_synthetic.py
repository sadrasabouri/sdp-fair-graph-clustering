# Experiment script for the SRFSC algorithm and comparison to others
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
    x_sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
    x_kmeans = KMeans(n_clusters=2, n_init=10).fit(A)
    x_ghodsi = ghodsi_cluster_graph(A, s, num_clusters=2)
    x_kleindessner = kleindessner_fair_SC_unnormalized(A, k=2, sensitive=s)

    print(f"Setting - SBM BinaryClusters UniformS : n={n} p_a={p_a} p_s=0.5")
    print("---Metrics---")
    print(">> Modularity:")
    print("Spectral Clustering:", modularity(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", modularity(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi:", modularity(G, convert_clusters_to_dict(x_ghodsi)))
    print("Kleindessner:", modularity(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: modularity(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), modularity(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    
    print(">> NMI:")
    print("Spectral Clustering:", nmi_score(x_act, x_sc.labels_))
    print("KMeans:", nmi_score(x_act, x_kmeans.labels_))
    print("Ghodsi:", nmi_score(x_act, x_ghodsi))
    print("Kleindessner:", nmi_score(x_act, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: nmi_score(x_act, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), nmi_score(x_act, x_srfsc_best.labels_))
    
    print(">> Conductance:")
    print("Spectral Clustering:", conductance(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", conductance(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi:", conductance(G, convert_clusters_to_dict(x_ghodsi)))
    print("Kleindessner:", conductance(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: conductance(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), conductance(G, convert_clusters_to_dict(x_srfsc_best.labels_)))

    print(">> Silhouette:")
    print("Spectral Clustering:", silhouette_score_custom(A, x_sc.labels_))
    print("KMeans:", silhouette_score_custom(A, x_kmeans.labels_))
    print("Ghodsi:", silhouette_score_custom(A, x_ghodsi))
    print("Kleindessner:", silhouette_score_custom(A, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: silhouette_score_custom(A, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), silhouette_score_custom(A, x_srfsc_best.labels_))

    print(">> Demographic Parity:")
    print("Spectral Clustering:", demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi:", demographic_parity(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
    print("Kleindessner:", demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: demographic_parity(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), demographic_parity(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

    print(">> Fairness Violation:")
    print("Spectral Clustering:", fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi:", fairness_violation(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
    print("Kleindessner:", fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: fairness_violation(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), fairness_violation(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

    print(">> Consistency Score:")
    print("Spectral Clustering:", consistency_score(G, x_sc.labels_, s))
    print("KMeans:", consistency_score(G, x_kmeans.labels_, s))
    print("Ghodsi:", consistency_score(G, x_ghodsi, s))
    print("Kleindessner:", consistency_score(G, x_kleindessner, s))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: consistency_score(G, x, s))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), consistency_score(G, x_srfsc_best.labels_, s))

#   Binary clustering with biased sensitive
for n in [100]:
    p_a = [[0.6, 0.4], [0.4, 0.6]]
    p_s_blocks = [0.25, 0.75]
    G, A, x_act, s = generate_synthetic_data_sbm_binary_s_biased([n // 2, n // 2],
                                                                p_a, p_s_blocks, seed=10)
    x_sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
    x_kmeans = KMeans(n_clusters=2, n_init=10).fit(A)
    x_ghodsi = ghodsi_cluster_graph(A, s, num_clusters=2)
    x_kleindessner = kleindessner_fair_SC_unnormalized(A, k=2, sensitive=s)

    print("=====================================")
    print(f"Setting - SBM BinaryClusters BiasedS : n={n} p_a={p_a} p_s={p_s_blocks}")
    print("---Metrics---")
    print(">> Modularity:")
    print("Spectral Clustering:", modularity(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", modularity(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi:", modularity(G, convert_clusters_to_dict(x_ghodsi)))
    print("Kleindessner:", modularity(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: modularity(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), modularity(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    
    print(">> NMI:")
    print("Spectral Clustering:", nmi_score(x_act, x_sc.labels_))
    print("KMeans:", nmi_score(x_act, x_kmeans.labels_))
    print("Ghodsi:", nmi_score(x_act, x_ghodsi))
    print("Kleindessner:", nmi_score(x_act, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: nmi_score(x_act, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), nmi_score(x_act, x_srfsc_best.labels_))
    
    print(">> Conductance:")
    print("Spectral Clustering:", conductance(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", conductance(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi:", conductance(G, convert_clusters_to_dict(x_ghodsi)))
    print("Kleindessner:", conductance(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: conductance(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), conductance(G, convert_clusters_to_dict(x_srfsc_best.labels_)))

    print(">> Silhouette:")
    print("Spectral Clustering:", silhouette_score_custom(A, x_sc.labels_))
    print("KMeans:", silhouette_score_custom(A, x_kmeans.labels_))
    print("Ghodsi:", silhouette_score_custom(A, x_ghodsi))
    print("Kleindessner:", silhouette_score_custom(A, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: silhouette_score_custom(A, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), silhouette_score_custom(A, x_srfsc_best.labels_))

    print(">> Demographic Parity:")
    print("Spectral Clustering:", demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi:", demographic_parity(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
    print("Kleindessner:", demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: demographic_parity(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), demographic_parity(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

    print(">> Fairness Violation:")
    print("Spectral Clustering:", fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi:", fairness_violation(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
    print("Kleindessner:", fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: fairness_violation(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), fairness_violation(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

    print(">> Consistency Score:")
    print("Spectral Clustering:", consistency_score(G, x_sc.labels_, s))
    print("KMeans:", consistency_score(G, x_kmeans.labels_, s))
    print("Ghodsi:", consistency_score(G, x_ghodsi, s))
    print("Kleindessner:", consistency_score(G, x_kleindessner, s))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: consistency_score(G, x, s))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), consistency_score(G, x_srfsc_best.labels_, s))

#   k classes clustering uniform sensitive
for n in [100]:

    p_a = [[0.6, 0.4, 0.3, 0.1, 0.01],
           [0.4, 0.6, 0.4, 0.1, 0.45],
           [0.3, 0.4, 0.6, 0.1, 0.01],
           [0.1, 0.1, 0.1, 0.6, 0.01],
           [0.01, 0.45, 0.01, 0.01, 0.6]]
    G, A, x_act, s = generate_synthetic_data_sbm_binary_s([2 * n // 5, n // 10, n // 10, n // 5, n // 5],
                                                          p_a, p_s=0.5, seed=10)
    x_sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
    x_kmeans = KMeans(n_clusters=2, n_init=10).fit(A)
    x_ghodsi = ghodsi_cluster_graph(A, s, num_clusters=2)
    x_kleindessner = kleindessner_fair_SC_unnormalized(A, k=2, sensitive=s)

    print(f"Setting - SBM KClusters UniformS : n={n} p_a={p_a} p_s=0.5")
    print("---Metrics---")
    print(">> Modularity:")
    print("Spectral Clustering:", modularity(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", modularity(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi:", modularity(G, convert_clusters_to_dict(x_ghodsi)))
    print("Kleindessner:", modularity(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: modularity(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), modularity(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    
    print(">> NMI:")
    print("Spectral Clustering:", nmi_score(x_act, x_sc.labels_))
    print("KMeans:", nmi_score(x_act, x_kmeans.labels_))
    print("Ghodsi:", nmi_score(x_act, x_ghodsi))
    print("Kleindessner:", nmi_score(x_act, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: nmi_score(x_act, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), nmi_score(x_act, x_srfsc_best.labels_))
    
    print(">> Conductance:")
    print("Spectral Clustering:", conductance(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", conductance(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi:", conductance(G, convert_clusters_to_dict(x_ghodsi)))
    print("Kleindessner:", conductance(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: conductance(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), conductance(G, convert_clusters_to_dict(x_srfsc_best.labels_)))

    print(">> Silhouette:")
    print("Spectral Clustering:", silhouette_score_custom(A, x_sc.labels_))
    print("KMeans:", silhouette_score_custom(A, x_kmeans.labels_))
    print("Ghodsi:", silhouette_score_custom(A, x_ghodsi))
    print("Kleindessner:", silhouette_score_custom(A, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: silhouette_score_custom(A, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), silhouette_score_custom(A, x_srfsc_best.labels_))

    print(">> Demographic Parity:")
    print("Spectral Clustering:", demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi:", demographic_parity(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
    print("Kleindessner:", demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: demographic_parity(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), demographic_parity(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

    print(">> Fairness Violation:")
    print("Spectral Clustering:", fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi:", fairness_violation(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
    print("Kleindessner:", fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: fairness_violation(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), fairness_violation(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

    print(">> Consistency Score:")
    print("Spectral Clustering:", consistency_score(G, x_sc.labels_, s))
    print("KMeans:", consistency_score(G, x_kmeans.labels_, s))
    print("Ghodsi:", consistency_score(G, x_ghodsi, s))
    print("Kleindessner:", consistency_score(G, x_kleindessner, s))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: consistency_score(G, x, s))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), consistency_score(G, x_srfsc_best.labels_, s))

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
    x_sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize', random_state=0).fit(A)
    x_kmeans = KMeans(n_clusters=2, n_init=10).fit(A)
    x_ghodsi = ghodsi_cluster_graph(A, s, num_clusters=2)
    x_kleindessner = kleindessner_fair_SC_unnormalized(A, k=2, sensitive=s)

    print(f"Setting - SBM KClusters UniformS : n={n} p_a={p_a} p_s={p_s_blocks}")
    print("---Metrics---")
    print(">> Modularity:")
    print("Spectral Clustering:", modularity(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", modularity(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi:", modularity(G, convert_clusters_to_dict(x_ghodsi)))
    print("Kleindessner:", modularity(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: modularity(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), modularity(G, convert_clusters_to_dict(x_srfsc_best.labels_)))
    
    print(">> NMI:")
    print("Spectral Clustering:", nmi_score(x_act, x_sc.labels_))
    print("KMeans:", nmi_score(x_act, x_kmeans.labels_))
    print("Ghodsi:", nmi_score(x_act, x_ghodsi))
    print("Kleindessner:", nmi_score(x_act, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: nmi_score(x_act, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), nmi_score(x_act, x_srfsc_best.labels_))
    
    print(">> Conductance:")
    print("Spectral Clustering:", conductance(G, convert_clusters_to_dict(x_sc.labels_)))
    print("KMeans:", conductance(G, convert_clusters_to_dict(x_kmeans.labels_)))
    print("Ghodsi:", conductance(G, convert_clusters_to_dict(x_ghodsi)))
    print("Kleindessner:", conductance(G, convert_clusters_to_dict(x_kleindessner)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: conductance(G, convert_clusters_to_dict(x)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), conductance(G, convert_clusters_to_dict(x_srfsc_best.labels_)))

    print(">> Silhouette:")
    print("Spectral Clustering:", silhouette_score_custom(A, x_sc.labels_))
    print("KMeans:", silhouette_score_custom(A, x_kmeans.labels_))
    print("Ghodsi:", silhouette_score_custom(A, x_ghodsi))
    print("Kleindessner:", silhouette_score_custom(A, x_kleindessner))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: silhouette_score_custom(A, x))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), silhouette_score_custom(A, x_srfsc_best.labels_))

    print(">> Demographic Parity:")
    print("Spectral Clustering:", demographic_parity(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", demographic_parity(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi:", demographic_parity(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
    print("Kleindessner:", demographic_parity(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: demographic_parity(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), demographic_parity(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

    print(">> Fairness Violation:")
    print("Spectral Clustering:", fairness_violation(convert_clusters_to_dict(x_sc.labels_), convert_clusters_to_dict(s)))
    print("KMeans:", fairness_violation(convert_clusters_to_dict(x_kmeans.labels_), convert_clusters_to_dict(s)))
    print("Ghodsi:", fairness_violation(convert_clusters_to_dict(x_ghodsi), convert_clusters_to_dict(s)))
    print("Kleindessner:", fairness_violation(convert_clusters_to_dict(x_kleindessner), convert_clusters_to_dict(s)))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: fairness_violation(convert_clusters_to_dict(x), convert_clusters_to_dict(s)))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), fairness_violation(convert_clusters_to_dict(x_srfsc_best.labels_), convert_clusters_to_dict(s)))

    print(">> Consistency Score:")
    print("Spectral Clustering:", consistency_score(G, x_sc.labels_, s))
    print("KMeans:", consistency_score(G, x_kmeans.labels_, s))
    print("Ghodsi:", consistency_score(G, x_ghodsi, s))
    print("Kleindessner:", consistency_score(G, x_kleindessner, s))
    x_srfsc_best, x_srfsc_params, x_srfsc_score = fair_clustering_best_parameter(A, s, k=2, metric=lambda x: consistency_score(G, x, s))
    print("SRFSC (mu={mu},lambda={y}):".format(**x_srfsc_params), consistency_score(G, x_srfsc_best.labels_, s))
