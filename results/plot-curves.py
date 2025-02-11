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
GOOD_DIRECTION = {"Modularity": "↑", "NMI": "↑", "Conductance": "↓", "Silhouette": "↑"}
FAIRNESS_METRICS = ["Demographic Parity", "Fairness Violation", "Consistency Score"]
GOOD_DIRECTION.update({"Demographic Parity": "↑", "Fairness Violation": "↓", "Consistency Score": "↓"})
COLOR_LIST = ['b', 'g', 'r', 'c', 'y', 'k']
MARKER_LIST = ['o', 's', 'D', '^', 'p', 'P']

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
        other_methods.remove("Ghodsi (GF)")
        for k, method in enumerate(other_methods):
            label = method
            if method == "Ghodsi (IF)":
                label = "iFairNMT"
            plt.scatter(cmetric_data[method], fmetric_data[method], label=label, color=COLOR_LIST[k], marker=MARKER_LIST[k])
        k += 1
        ours_cmetric = cmetric_data["SRFSC"]
        ours_fmetric = fmetric_data["SRFSC"]
        our_points_x = []
        our_points_y = []
        for ocm, ofm in zip(ours_cmetric, ours_fmetric):
            if ocm["mu"] != ofm["mu"] or ocm["y"] != ofm["y"]:
                print("Mismatched mu and y")
                print(ocm, ofm)
                continue
            if fmetric == "Fairness Violation": # due to a fault in implementation
                our_points_y.append(-ofm["score:"])
            else:
                our_points_y.append(ofm["score:"])
            our_points_x.append(ocm["score:"])
        # remove points which are dominated by other points in ocm and ofm (based on GOOD_DIRECTION)
        # dominated_clustering_idx = []
        # dominated_fairness_idx = []
        # for i, (x, y) in enumerate(zip(our_points_x, our_points_y)):
        #     for j, (x_, y_) in enumerate(zip(our_points_x, our_points_y)):
        #         if i == j:
        #             continue
        #         if (x_ > x and GOOD_DIRECTION[cmetri] == "↑") or (x_ < x and GOOD_DIRECTION[cmetri] == "↓"):
        #             dominated_clustering_idx.append(i)
        #         if (y_ > y and GOOD_DIRECTION[fmetric] == "↑") or (y_ < y and GOOD_DIRECTION[fmetric] == "↓"):
        #             dominated_fairness_idx.append(i)
        # our_points_x = [x for i, x in enumerate(our_points_x) if i not in dominated_clustering_idx or i not in dominated_fairness_idx]
        # our_points_y = [y for i, y in enumerate(our_points_y) if i not in dominated_clustering_idx or i not in dominated_fairness_idx]
        plt.scatter(our_points_x, our_points_y, label="Ours", color=COLOR_LIST[k], marker=MARKER_LIST[k])
        plt.xlabel(cmetri + GOOD_DIRECTION[cmetri])
        if fmetric == "Demographic Parity":
            plt.ylabel("Clustering Balance" + GOOD_DIRECTION[fmetric])
        else:
            plt.ylabel(fmetric + GOOD_DIRECTION[fmetric])
        plt.title(f"{metric_file.split('_')[-1].replace('.json', '').replace('_', ' ').capitalize()}")
        plt.legend()
        plt.savefig(f"{metric_file}_{cmetri}_{fmetric}.pdf", bbox_inches='tight')
        plt.close()
