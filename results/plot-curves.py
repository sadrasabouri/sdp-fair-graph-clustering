import itertools
import json
import matplotlib.pyplot as plt

METRIC_FILES = ["synthetic_data_results_BU.json",
                "synthetic_data_results_BB.json",
                "synthetic_data_results_5U.json",
                "synthetic_data_results_5B.json",
                "realworld_data_results_moreno.json",
                "realworld_data_results_facebook.json"]

CLUSTERING_METRICS = ["Modularity", "NMI", "Conductance", "Silhouette"]
FAIRNESS_METRICS = ["Demographic Parity", "Fairness Violation", "Consistency Score"]
COLOR_LIST = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
MARKER_LIST = ['o', 's', 'D', '^', 'v', 'p', 'P']

for metric_file in METRIC_FILES:
    with open(f"{metric_file}") as f:
        data = json.load(f)
    for cmetri, fmetric in itertools.product(CLUSTERING_METRICS, FAIRNESS_METRICS):
        cmetric_data = data[cmetri]
        fmetric_data = data[fmetric]
        print(f"Plotting {cmetri} vs {fmetric} for {metric_file.replace('.json', '').replace('_', ' ').capitalize()}")
        # Print a scatter plot of the clustering metric vs fairness metric
        plt.figure()
        other_methods = list(cmetric_data.keys())
        other_methods.remove("SRFSC")
        for i, method in enumerate(other_methods):
            plt.scatter(cmetric_data[method], fmetric_data[method], label=method, color=COLOR_LIST[i], marker=MARKER_LIST[i])
        i += 1
        ours_cmetric = cmetric_data["SRFSC"]
        ours_fmetric = fmetric_data["SRFSC"]
        our_points_x = []
        our_points_y = []
        for ocm, ofm in zip(ours_cmetric, ours_fmetric):
            if ocm["mu"] != ofm["mu"] or ocm["y"] != ofm["y"]:
                print("Mismatched mu and y")
                print(ocm, ofm)
                continue
            our_points_x.append(ocm["score:"])
            our_points_y.append(ofm["score:"])
        plt.scatter(our_points_x, our_points_y, label="SRFSC - SC (ours)", color=COLOR_LIST[i], marker=MARKER_LIST[i])
        plt.xlabel(cmetri)
        if fmetric == "Demographic Parity":
            plt.ylabel("Clustering Balance")
        else:
            plt.ylabel(fmetric)
        plt.title(f"{fmetric} vs {cmetri} for {metric_file.replace('.json', '').replace('_', ' ').capitalize()}")
        plt.legend()
        plt.savefig(f"{metric_file}_{cmetri}_{fmetric}.pdf", bbox_inches='tight')
        plt.close()
