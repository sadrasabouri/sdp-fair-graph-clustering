# -*- coding: utf-8 -*-
"""https://networkrepository.com/soc-highschool-moreno.php
"""

import numpy as np
import scipy.sparse as sp
import os
import zipfile
import os

# Path to the uploaded ZIP file
zip_path = "/content/soc-highschool-moreno.zip"
extract_path = "/content/facebook_data"

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete. Files available:")
print(os.listdir(extract_path))


def load_facebook():
    """
    Load the FacebookNet (High School) dataset from extracted files.
    """
    path = "/content/facebook_data"
    file_name = "soc-highschool-moreno.edges"

    # Check if file exists
    file_path = os.path.join(path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found. Please check extraction.")

    # Load edges, skipping commented lines
    edges = np.loadtxt(file_path, dtype=int, comments='%')

    num_nodes = edges.max() + 1  # Assuming nodes are zero-indexed

    # Create adjacency matrix
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                      shape=(num_nodes, num_nodes), dtype=int)
    A = A + A.T  # Ensure the matrix is symmetric

    # **Sensitive Attribute (Gender)**
    # Assigning gender randomly since the dataset lacks it
    np.random.seed(42)
    sensitivity_vector = np.random.choice([0, 1], size=num_nodes)  # 0 = Male, 1 = Female

    return A, sensitivity_vector

if __name__ == "__main__":
    A, s = load_facebook()
    print(f"Adjacency matrix shape: {A.shape}")
    print(f"Sensitive attribute vector shape: {s.shape}")
