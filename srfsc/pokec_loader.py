import numpy as np
import scipy.sparse as sp
import os
import urllib.request

def download_and_extract_data():
    """Download and extract both profiles and relationships data"""
    url_profiles = "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz"
    url_relationships = "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"

    # Download data if not already downloaded
    if not os.path.exists('soc-pokec-profiles.txt.gz'):
        print("Downloading profiles data...")
        urllib.request.urlretrieve(url_profiles, 'soc-pokec-profiles.txt.gz')
    if not os.path.exists('soc-pokec-relationships.txt.gz'):
        print("Downloading relationships data...")
        urllib.request.urlretrieve(url_relationships, 'soc-pokec-relationships.txt.gz')

    # Extract feature data
    print("Extracting feature data...")
    os.makedirs('pokec_data', exist_ok=True)
    os.system(f"gunzip -c soc-pokec-profiles.txt.gz > pokec_data/soc-pokec-profiles.txt")
    os.system(f"gunzip -c soc-pokec-relationships.txt.gz > pokec_data/soc-pokec-relationships.txt")
    

def load_pokec(path: str='pokec_data'):
    """
    Load the Pokec social network dataset.

    :param path: Path to the dataset directory
    """
    # Load edges
    edges = np.loadtxt(os.path.join(path, 'soc-pokec-relationships.txt'), dtype=int)
    num_nodes = edges.max() + 1  # Assuming nodes are zero-indexed

    # Create adjacency matrix
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                      shape=(num_nodes, num_nodes), dtype=int)
    A = A + A.T  # Ensure the matrix is symmetric

    # Load sensitive attribute (gender)
    sensitive_attrs = {}
    with open(os.path.join(path, 'soc-pokec-profiles.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            # based on https://snap.stanford.edu/data/soc-pokec-readme.txt
            node_id = int(parts[0])
            gender = parts[3]
            if gender != "null":
                gender = int(gender)
            gender = None
            sensitive_attrs[node_id] = gender

    # Create sensitivity vector
    sensitivity_vector = np.array([sensitive_attrs.get(i, -1) for i in range(num_nodes)])

    return A, sensitivity_vector

if __name__ == "__main__":
    download_and_extract_data()
    A, s = load_pokec('pokec_data')
    print(f"Adjacency matrix shape: {A.shape}")
    print(f"Sensitive attribute vector shape: {s.shape}")
