import json
from typing import List, Dict, Tuple
import numpy as np
import itertools

DATA_FILES = ["synthetic_data_results_BU",
                "synthetic_data_results_BB",
                "synthetic_data_results_5U",
                "synthetic_data_results_5B",
                "realworld_data_results_moreno",
                "realworld_data_results_facebook"]
data_files = {}
for data_file in DATA_FILES:
    with open(f"{data_file}.json") as f:
        data_files[data_file] = json.load(f)

LATEX_TERM2_METRIC = {
    "M↑": "Modularity",
    "NMI↑": "NMI",
    "C↓": "Conductance",
    "S↑": "Silhouette",
    "CB↑": "Demographic Parity",
    "FV↓": "Fairness Violation",
    "CS↓": "Consistency Score"
}
LATEX_TERM2_DATASET = {
    "BU": data_files["synthetic_data_results_BU"],
    "BB": data_files["synthetic_data_results_BB"],
    "5U": data_files["synthetic_data_results_5U"],
    "5B": data_files["synthetic_data_results_5B"],
    "Moreno": data_files["realworld_data_results_moreno"],
    "Facebook": data_files["realworld_data_results_facebook"]
}

def get_srfsc_scores(data_scores) -> Dict[Tuple[float, float], float]:
    score_map = {}
    for score_instance in data_scores:
        score_map[(score_instance["mu"], score_instance["y"])] = score_instance["score:"]
    return score_map


def generate_latex_table(synth_dataset_names: List,
                         real_dataset_names: List,
                         metrics: List) -> str:
    # Define dataset names
    num_synth_datasets = len(synth_dataset_names)
    num_real_datasets = len(real_dataset_names)
    
    # Define metric names
    num_metrics = len(metrics)
    
    # Start building LaTeX string
    latex = r"""
    \begin{table*}[h]
        \centering
        \caption{Comparison of clustering methods across different datasets using different metrics.}
        \small
        \resizebox{\linewidth}{!}{
        \begin{tabular}{l | """
    
    # Column formatting
    col_format = "|".join(["c" * num_metrics] * (num_synth_datasets + num_real_datasets))
    latex += f"{col_format} }}\n"
    
    # Header rows
    latex += "    \\multirow{3}{*}{Methods} & \\multicolumn{" + str(num_synth_datasets * num_metrics) + "}{c|}{Synthetic Data} "
    latex += f"& \\multicolumn{{{num_real_datasets * num_metrics}}}{{c}}{{Real-world Data}} \\\\ \n"
    
    latex += "    & " + " & ".join([f"\\multicolumn{{{num_metrics}}}{{c}}{{\\textbf{{{name}}}}}" for name in synth_dataset_names + real_dataset_names]) + "\\\\ \n"
    
    latex += "    & " + " & ".join(metrics * (num_synth_datasets + num_real_datasets)) + "\\\\ \n    \\hline\n"
    
    # Data rows
    OTHER_METHODS = list(LATEX_TERM2_DATASET["BU"]['Modularity'].keys())
    OTHER_METHODS.remove("SRFSC")
    for method_name in OTHER_METHODS:
        row = f"    {method_name} & "
        for dataset_name in synth_dataset_names:
            for metric_name in metrics:
                row += f"{LATEX_TERM2_DATASET[dataset_name][LATEX_TERM2_METRIC[metric_name]][method_name]:.3f} & "
            row += "   "
        for dataset_name in real_dataset_names:
            for metric_name in metrics:
                row += f"{LATEX_TERM2_DATASET[dataset_name][LATEX_TERM2_METRIC[metric_name]][method_name]:.3f} & "
            row += "   "
        row = row[:-6] + " \\\\ \n"
        latex += row

    # SRFSC rows
    for mu, y in itertools.product(np.arange(0, 1, 0.2), np.arange(0, 1, 0.2)):
        row = f"    SRFSC ($\mu={mu:.2f}, \lambda={y:.2f}$) & "
        for dataset_name in synth_dataset_names:
            for metric_name in metrics:
                score = get_srfsc_scores(LATEX_TERM2_DATASET[dataset_name][LATEX_TERM2_METRIC[metric_name]]['SRFSC'])
                row += f"{score[(mu, y)]:.3f} & "
            row += "   "
        for dataset_name in real_dataset_names:
            for metric_name in metrics:
                score = get_srfsc_scores(LATEX_TERM2_DATASET[dataset_name][LATEX_TERM2_METRIC[metric_name]]['SRFSC'])
                row += f"{score[((mu, y))]:.3f} & "
            row += "   "
        row = row[:-6] + " \\\\ \n"
        latex += row
    
    
    latex += "    \\hline\n"
    latex += """
        \end{tabular}
        }
        \label{tab:clustering_results}
    \end{table*}
    """
    
    return latex

# Example usage
print(generate_latex_table(["BU", "BB", "5U", "5B"],
                           ["Moreno", "Facebook"],
                           ["M↑", "NMI↑", "C↓", "S↑", "CB↑", "FV↓", "CS↓"]))
