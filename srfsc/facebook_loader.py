import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import gzip
import urllib.request
import os
import tarfile

def download_and_extract_data():
    """Download and extract both the combined file and feature data"""
    # Download combined file
    url_combined = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    url_features = "https://snap.stanford.edu/data/facebook.tar.gz"

    if not os.path.exists('facebook_combined.txt.gz'):
        print("Downloading network data...")
        urllib.request.urlretrieve(url_combined, 'facebook_combined.txt.gz')
    if not os.path.exists('facebook.tar.gz'):
        print("Downloading feature data...")
        urllib.request.urlretrieve(url_features, 'facebook.tar.gz')

    # Extract feature data
    print("Extracting feature data...")
    with tarfile.open('facebook.tar.gz', 'r:gz') as tar:
        tar.extractall(path='./facebook_data')

def process_all_networks(data_dir):
    """Process all networks to get gender information"""
    results = {}

    # Iterate through all files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.featnames'):
            network_id = filename.split('.')[0]

            # Read feature names file
            gender_indices = []
            with open(os.path.join(data_dir, filename), 'r') as f:
                for i, line in enumerate(f):
                    if 'gender' in line.lower():
                        gender_indices.append(i)

            if gender_indices:
                # print(f"\nProcessing network {network_id}")
                # print(f"Gender features found at indices: {gender_indices}")

                # Read corresponding feature file
                feat_file = os.path.join(data_dir, f"{network_id}.feat")
                if os.path.exists(feat_file):
                    network_data = {}
                    with open(feat_file, 'r') as f:
                        for line in f:
                            values = list(map(int, line.strip().split()))
                            node_id = values[0]
                            features = {f'gender_{i}': values[idx+1]
                                      for i, idx in enumerate(gender_indices)}
                            network_data[node_id] = features
                    results[network_id] = network_data

    return results

def get_gender_data():
    """Get gender information from feature files"""
    data_dir = "./facebook_data/facebook"

    if not os.path.exists(data_dir):
        # print("Downloading and extracting data...")
        download_and_extract_data()

    # Process networks to get gender information
    results = process_all_networks(data_dir)
    return results

def create_network_with_gender():
    """Create network with gender information"""
    # Get gender data
    gender_data = get_gender_data()

    # Create network from combined file
    G = nx.Graph()

    # print("\nReading edge data...")
    with gzip.open('facebook_combined.txt.gz', 'rt') as f:
        for line in f:
            node1, node2 = map(int, line.strip().split())
            G.add_edge(node1, node2)

    # Add gender attributes to nodes
    # print("Adding gender attributes...")
    for network_id, network_data in gender_data.items():
        for node_id, gender_features in network_data.items():
            node_id = int(node_id)
            if node_id in G.nodes():
                # Add these features as node attributes
                G.nodes[node_id].update(gender_features)

    # print(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, gender_data

def create_filtered_network():
    """Create network excluding nodes with pattern (0,0)"""
    G, gender_data = create_network_with_gender()

    # Find nodes to remove (those with pattern 0,0)
    nodes_to_remove = []
    for node in G.nodes():
        if G.nodes[node]:  # if node has gender data
            values = tuple(G.nodes[node].values())[:2]  # get first two values
            if values == (0, 0):
                nodes_to_remove.append(node)

    # Remove nodes with pattern (0,0)
    G.remove_nodes_from(nodes_to_remove)
    # print(f"Removed {len(nodes_to_remove)} nodes with pattern (0,0)")
    # print(f"Network now has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G

def analyze_gender_patterns(G):
    """Analyze gender pattern distribution in the network"""
    patterns = {}
    node_pattern_mapping = {}

    for node in G.nodes():
        if G.nodes[node]:  # if node has gender data
            values = tuple(G.nodes[node].values())[:2]
            patterns[values] = patterns.get(values, 0) + 1
            node_pattern_mapping[node] = values

    # Sort patterns by count
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)

    # print("\nDetailed Gender Pattern Analysis:")
    # print("--------------------------------")
    total_nodes = G.number_of_nodes()
    for pattern, count in sorted_patterns:
        print(f"Pattern {pattern}: {count} nodes ({count/total_nodes*100:.2f}%)")

    return patterns, node_pattern_mapping

def visualize_gender_patterns_pie(patterns):
    """Create interactive pie chart of gender patterns"""
    labels = [f"Pattern {pattern}" for pattern, _ in patterns.items()]
    values = [count for _, count in patterns.items()]
    colors = ['lightblue', 'lightpink', 'lightgray']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=colors),
        texttemplate="%{label}<br>%{value} nodes<br>(%{percent})"
    )])

    fig.update_layout(
        title="Gender Pattern Distribution in Network",
        width=800,
        height=800
    )

    fig.show()
    fig.write_html("gender_patterns_pie.html")

def visualize_gender_patterns_bar(patterns):
    """Create interactive bar chart of gender patterns"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    patterns_list = list(patterns.items())
    labels = [f"Pattern {pattern}" for pattern, _ in patterns_list]
    values = [count for _, count in patterns_list]
    total = sum(values)
    percentages = [count/total*100 for count in values]
    colors = ['lightblue', 'lightpink', 'lightgray']

    fig.add_trace(
        go.Bar(x=labels, y=values, name="Node Count", marker_color=colors),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=labels, y=percentages, name="Percentage",
                  line=dict(color='red', width=2), mode='lines+markers'),
        secondary_y=True
    )

    fig.update_layout(
        title="Gender Pattern Distribution - Counts and Percentages",
        xaxis_title="Pattern Type",
        width=800,
        height=500
    )

    fig.show()
    fig.write_html("gender_patterns_bar.html")

def create_and_visualize_adjacency_matrix(G):
    """Create and visualize the adjacency matrix with improved visibility"""
    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()

    # Create figure with a white background
    plt.figure(figsize=(20, 20), facecolor='white')

    # Create heatmap with improved visibility
    sns.heatmap(adj_matrix,
                cmap='RdBu_r',  # Red-Blue diverging colormap
                square=True,
                xticklabels=100,  # Show fewer tick labels for clarity
                yticklabels=100,
                cbar_kws={
                    'label': 'Connection',
                    'orientation': 'vertical',
                    'shrink': 0.8,
                    'pad': 0.02
                })

    # Customize the plot
    plt.title('Facebook Network Connectivity Matrix\n(Node Connections Visualization)',
              fontsize=16, pad=20)
    plt.xlabel('Node ID', fontsize=12, labelpad=10)
    plt.ylabel('Node ID', fontsize=12, labelpad=10)

    # Add informative text annotation
    plt.figtext(1.02, 0.7,
                "How to read this plot:\n" +
                "- Each point represents a connection\n" +
                "- Dark red indicates connected nodes\n" +
                "- Dark blue indicates no connection\n" +
                f"- Matrix size: {adj_matrix.shape[0]}×{adj_matrix.shape[1]}\n" +
                f"- Total connections: {int(np.sum(adj_matrix))}\n" +
                f"- Network density: {np.sum(adj_matrix)/(adj_matrix.shape[0]**2):.4f}",
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Adjust layout to prevent text cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Save with high resolution
    plt.savefig('adjacency_matrix_improved.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

    return adj_matrix

def analyze_network_metrics(G):
    """Analyze various network metrics"""
    # print("\nNetwork Metrics Analysis:")
    # print("------------------------")

    # Basic metrics
    # print(f"Number of nodes: {G.number_of_nodes()}")
    # print(f"Number of edges: {G.number_of_edges()}")

    # Density
    density = nx.density(G)
    # print(f"Network density: {density:.4f}")

    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G)
    # print(f"Average clustering coefficient: {avg_clustering:.4f}")

    # Average shortest path length
    avg_path = nx.average_shortest_path_length(G)
    # print(f"Average shortest path length: {avg_path:.4f}")

    # Degree statistics
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)
    # print(f"Average degree: {avg_degree:.2f}")
    # print(f"Maximum degree: {max(degrees)}")
    # print(f"Minimum degree: {min(degrees)}")

def visualize_degree_distribution(G):
    """Visualize the degree distribution"""
    degrees = [d for n, d in G.degree()]

    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50, alpha=0.75, color='blue')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.yscale('log')

    plt.show()
    plt.savefig('degree_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def extract_adjacency_and_features(G):
    """Extract adjacency matrix and sensitive attribute vector"""
    # Create mapping of old node IDs to new sequential indices
    node_to_idx = {node: idx for idx, node in enumerate(sorted(G.nodes()))}

    # Extract adjacency matrix using the new indices
    A = nx.adjacency_matrix(G).todense()

    # Create sensitive feature vector using the new indices
    s = np.zeros(G.number_of_nodes())
    for node in G.nodes():
        if G.nodes[node]:  # if node has gender data
            values = list(G.nodes[node].values())
            if values:  # if there are any values
                idx = node_to_idx[node]  # use mapped index
                s[idx] = values[0]  # use first gender feature as sensitive attribute

    # Print statistics about sensitive attribute distribution
    # print("\nSensitive Attribute Distribution:")
    # print("-" * 30)
    # print(f"Total nodes: {len(s)}")
    # print(f"Nodes with s=0: {np.sum(s == 0)}")
    # print(f"Nodes with s=1: {np.sum(s == 1)}")
    # print(f"Ratio s=1/total: {np.sum(s == 1)/len(s):.4f}")

    return A, s

def analyze_network_by_sensitive(G, s):
    """Analyze network characteristics by sensitive attribute"""
    # print("\nNetwork Analysis by Sensitive Attribute:")
    # print("-" * 35)

    # Create mapping of old node IDs to new sequential indices
    node_to_idx = {node: idx for idx, node in enumerate(sorted(G.nodes()))}

    # Analyze degrees by sensitive attribute
    degrees = dict(G.degree())
    s0_degrees = [degrees[node] for node, idx in node_to_idx.items() if s[idx] == 0]
    s1_degrees = [degrees[node] for node, idx in node_to_idx.items() if s[idx] == 1]

    # print(f"Average degree (s=0): {np.mean(s0_degrees):.2f}")
    # print(f"Average degree (s=1): {np.mean(s1_degrees):.2f}")
    # print(f"Degree ratio (s=1/s=0): {np.mean(s1_degrees)/np.mean(s0_degrees):.2f}")

def visualize_sensitive_distribution(G, s):
    """Visualize distribution of sensitive attribute"""
    plt.figure(figsize=(12, 6))

    # Create subplot for sensitive attribute distribution
    plt.subplot(1, 2, 1)
    values, counts = np.unique(s, return_counts=True)
    plt.bar(['s=0', 's=1'], counts, color=['lightblue', 'lightpink'])
    plt.title('Distribution of Sensitive Attribute')
    plt.ylabel('Count')

    # Create subplot for degree distribution by sensitive attribute
    plt.subplot(1, 2, 2)
    degrees = np.array([d for n, d in G.degree()])
    s0_degrees = degrees[s == 0]
    s1_degrees = degrees[s == 1]

    plt.hist([s0_degrees, s1_degrees], label=['s=0', 's=1'],
             bins=30, alpha=0.6, density=True)
    plt.title('Degree Distribution by Sensitive Attribute')
    plt.xlabel('Degree')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.savefig('sensitive_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def load_facebook():
    """Load the Facebook social network dataset"""
    # Download and extract data
    download_and_extract_data()

    # Create filtered network
    G_filtered = create_filtered_network()
    return extract_adjacency_and_features(G_filtered)

def main():
    # Create filtered network
    G_filtered = create_filtered_network()

    # Extract adjacency matrix and sensitive attributes
    A, s = extract_adjacency_and_features(G_filtered)
    print(f"Adjacency matrix shape: {A.shape}")
    print(f"Sensitive attribute vector shape: {s.shape}")


if __name__ == "__main__":
    main()
