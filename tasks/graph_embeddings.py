import json
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
from node2vec import Node2Vec
import gensim.models
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import AffinityPropagation  # Import Affinity Propagation


def generate_graph_embeddings(graphs, embedding_dim=8):
    """
    Generate graph embeddings using Node2Vec.
    
    Args:
        graphs (list): List of NetworkX graphs
        embedding_dim (int): Dimensionality of embeddings
    
    Returns:
        np.array: Array of graph embeddings
    """
    graph_embeddings = []
    
    for G in graphs:
        # Prepare Node2Vec
        # Convert graph to undirected for Node2Vec
        undirected_G = G.to_undirected()
        
        # Node2Vec parameters
        node2vec = Node2Vec(undirected_G, 
                            dimensions=embedding_dim, 
                            walk_length=20, 
                            num_walks=15, 
                            workers=8)
        
        # Fit Word2Vec model
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Get node embeddings
        node_embeddings = [model.wv[str(node)] for node in G.nodes()]
        
        # Aggregate node embeddings (e.g., mean)
        graph_embedding = np.mean(node_embeddings, axis=0)
        graph_embeddings.append(graph_embedding)
    
    return np.array(graph_embeddings)

import plotly.express as px
import plotly.graph_objects as go

def visualize_graph_embeddings(embeddings, robot_files, path):
    """
    Visualize graph embeddings using PCA with interactive hover effects and cluster medians.
    Args:
    embeddings (np.array): Graph embeddings
    robot_files (list): List of corresponding robot files
    """
    print("embeddings: ", embeddings)
    
    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Perform Affinity Propagation clustering
    affinity_propagation = AffinityPropagation(max_iter=1000)
    kmeans_labels = affinity_propagation.fit_predict(reduced_embeddings)
    
    # Create a DataFrame for Plotly
    df = pd.DataFrame(reduced_embeddings, columns=['PC1', 'PC2'])
    df['Cluster'] = kmeans_labels
    df['File Name'] = robot_files
    
    # Find the median point for each cluster
    cluster_medians = []
    median_files = []
    
    for cluster in np.unique(kmeans_labels):
        cluster_data = df[df['Cluster'] == cluster]
        
        # Calculate the point closest to the median of PC1 and PC2
        median_pc1 = cluster_data['PC1'].median()
        median_pc2 = cluster_data['PC2'].median()
        
        # Find the point closest to the median coordinates
        median_index = ((cluster_data['PC1'] - median_pc1)**2 + 
                        (cluster_data['PC2'] - median_pc2)**2).argmin()
        
        median_point = cluster_data.iloc[median_index]
        cluster_medians.append(median_point)
        median_files.append(median_point['File Name'])
    
    # Create the main scatter plot
    fig = px.scatter(df, x='PC1', y='PC2', color='Cluster',
                     hover_name='File Name',
                     title='PCA of Graph Embeddings with Affinity Propagation Clustering',
                     labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'})
    
    # Add median points with a distinct marker style
    median_df = pd.DataFrame(cluster_medians)
    fig.add_trace(
        go.Scatter(
            x=median_df['PC1'], 
            y=median_df['PC2'],
            mode='markers',
            name='Cluster Medians',
            marker=dict(
                symbol='circle',  # Circle-shaped marker
                size=12,          # Adjusted size
                color='white',    # White fill
                line=dict(        # Black outline
                    color='black',
                    width=2
                )
            ),
            text=[f'Median File: {file}' for file in median_files],
            hoverinfo='text'
        )
    )
    
    # Adjust the legend spacing
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    
    # Show and save the plot
    fig.show()
    fig.write_html(path + 'affinitypropagation_clusters_with_medians.html')
    
    # Print out the median files for reference
    print("Median Files for Each Cluster:")
    for i, (point, file) in enumerate(zip(cluster_medians, median_files)):
        print(f"Cluster {point['Cluster']}: {file}")
    
    return cluster_medians, median_files


def find_parent(root, target_element):
    """
    Find the parent of a given element in the XML tree.
    
    Parameters:
    -----------
    root : ET.Element
        Root of the XML tree
    target_element : ET.Element
        Element whose parent we want to find
    
    Returns:
    --------
    ET.Element or None
        Parent element of the target, or None if not found
    """
    for parent in root.iter():
        for child in parent:
            if child == target_element:
                return parent
    return None


def parse_mujoco_xml_to_graph(xml_path):
    """
    Parse a MuJoCo XML file and create a graph representation.
    
    Parameters:
    -----------
    xml_path : str
        Path to the MuJoCo XML file
    
    Returns:
    --------
    nx.DiGraph
        A directed graph where nodes represent bodies and 
        edges represent kinematic connections
    """
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Dictionary to track body-to-parent mapping
    body_parent_map = {}
    
    # First pass: Collect body information
    for body in root.findall('.//body'):
        body_name = body.get('name', 'unnamed_body')
        
        # Collect geom attributes as node features
        geom_features = []
        for geom in body.findall('geom'):
            geom_info = {
                'size': geom.get('size', ''),
                'fromto': geom.get('fromto', '')
            }
            # Calculate length from 'fromto' if it exists
            if geom_info['fromto']:
                fromto_values = list(map(float, geom_info['fromto'].split()))
                length = ((fromto_values[3] - fromto_values[0]) ** 2 + 
                           (fromto_values[4] - fromto_values[1]) ** 2 + 
                           (fromto_values[5] - fromto_values[2]) ** 2) ** 0.5
                geom_info['length'] = length
            else:
                geom_info['length'] = 0.0  # Default length if 'fromto' is not provided
            
            geom_features.append(geom_info)
        
        # Collect joint information
        joint_features = []
        for joint in body.findall('joint'):
            joint_info = {
                'name': joint.get('name', 'unnamed_joint'),
                'gear': joint.get('gear', '')  # Extract gear value from joint
            }
            joint_features.append(joint_info)
        
        # Store node with features
        G.add_node(body_name, 
                   geoms=geom_features, 
                   joints=joint_features,
                   pos=body.get('pos', ''),
        )
        
        # Find parent body
        parent_element = find_parent(root, body)
        if parent_element is not None and parent_element.tag == 'body':
            parent_name = parent_element.get('name', 'unnamed_parent_body')
            body_parent_map[body_name] = parent_name
    
    # Second pass: Add edges based on kinematic hierarchy
    for body, parent in body_parent_map.items():
        if parent in G.nodes and body in G.nodes:
            G.add_edge(parent, body)
    
    return G

def extract_body_gear_mapping(xml_path):
    """
    Extract a mapping of body names to their motor gear values.
    
    Parameters:
    -----------
    xml_path : str
        Path to the MuJoCo XML file
    
    Returns:
    --------
    dict
        Dictionary mapping body names to motor gear values
    """
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Dictionary to store body name to gear mapping
    body_gear_map = {}
    
    # Find all bodies in the XML
    for body in root.findall('.//body'):
        body_name = body.get('name')
        
        # Find the joint for this body
        joint = body.find('joint')
        if joint is not None:
            joint_name = joint.get('name')
            
            # Find the corresponding motor in the actuator section
            motor = root.find(f".//motor[@joint='{joint_name}']")
            if motor is not None:
                gear_value = motor.get('gear')
                if gear_value:
                    body_gear_map[body_name] = int(gear_value)
    
    return body_gear_map

def extract_graph_features(G, xml_path):
    """
    Extract comprehensive features from the graph and XML.
    
    Parameters:
    -----------
    G : nx.DiGraph
        Graph representation of the MuJoCo model
    xml_path : str
        Path to the MuJoCo XML file
    
    Returns:
    --------
    dict
        Dictionary of graph and model features
    """
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract body-to-gear mapping
    body_gear_map = extract_body_gear_mapping(xml_path)
    
    # Initialize features dictionary
    features = {
        'num_bodies': G.number_of_nodes(),
        'num_connections': G.number_of_edges(),
        'is_connected': nx.is_strongly_connected(G),
        'body_geom_stats': {},
        'actuator_gears': []
    }
    
    # Collect actuator gear values
    # for actuator in root.findall('.//actuator/motor'):
    #     gear_value = actuator.get('gear', '')
    #     features['actuator_gears'].append(gear_value)
    
    # Process each node in the graph
    for node in G.nodes():
        # Get geoms for the node
        geoms = G.nodes[node].get('geoms', [])
        
        if geoms:
            # Assuming first geom for simplicity, adjust if needed
            geom = geoms[0]
            
            # Get gear value for this body
            gear_value = body_gear_map.get(node, None)
            
            features['body_geom_stats'][node] = {
                'length': geom.get('length', 0.0),
                'size': geom.get('size', ''),
                'gear_value': gear_value
            }

    return features

def process_xml_files(robot_files, all_graph_data):
    for xml_path in robot_files:
        # Create and process graph
        mujoco_graph = parse_mujoco_xml_to_graph(xml_path)
        graph_features = extract_graph_features(mujoco_graph, xml_path)
        
        # Update node attributes with features
        for node, features in graph_features['body_geom_stats'].items():
            mujoco_graph.nodes[node].update(features)
            
        all_graph_data.append(mujoco_graph)
    
    return all_graph_data

def load_graph_data(json_file):
    """
    Load graph data from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing graph data
    
    Returns:
        list: Parsed graph data
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def create_graph(item):
    """
    Create a NetworkX graph from a graph data item with string nodes and specific feature structure.
    
    Args:
        item (dict): Dictionary containing graph information with nodes, edges, and node_features
    
    Returns:
        nx.Graph: Constructed graph
    """
    G = nx.Graph()
    
    # Convert nodes to integers and add them to the graph
    for node_str, features in item['node_features'].items():
        node_id = int(node_str)  # Convert node to integer
        G.add_node(node_id, features=features)
    
    # Add edges, converting node strings to integers
    for edge in item['edges']:
        # Convert edge nodes to integers
        src = int(edge[0])
        dst = int(edge[1])
        G.add_edge(src, dst)
    
    return G

def perform_pca(G):
    """
    Perform PCA on node features and return reduced features.
    
    Args:
        G (nx.Graph): Input graph
    
    Returns:
        tuple: Reduced features and their categories
    """
    # Extract node features, handling potential missing values
    node_features = []
    for node in G.nodes():
        # Convert features to a list, handling potential None values
        # print("G.nodes[node]: ", G.nodes[node])
        node_feat = G.nodes[node]
        feat_list = [
            node_feat.get('length', 0),
            float(node_feat.get('size', 0)),
            node_feat.get('gear_value', 0) or 0
        ]
        node_features.append(feat_list)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(node_features)
    
    # Perform PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    
    return reduced_features, np.ones(len(reduced_features))  # Default single category

def plot_graph_clusters(reduced_features, categories):
    """
    Plot graph clusters using dimensionality-reduced features.
    
    Args:
        reduced_features (np.array): PCA-reduced node features
        categories (np.array): Category labels for each node
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                          c=categories, cmap='viridis', 
                          alpha=0.7, s=50)
    plt.title('Graph Clustering Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Category')
    plt.tight_layout()
    plt.show()
    plt.savefig('clustering.png')

def get_robot_files(folder_path):
    return [os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.endswith('.xml')]


def main():
    """
    Main function to load graph data, create graph, perform clustering, and visualize.
    """

    folder_path = "/home/knagiredla/robonet/logs/exp_GSCA_5_flat_base_hopper_100_000_134_1735392360/xmlrobots/gen_500_steps/valid/"

    all_graph_data = []
    # Process XML files and build graphs
    robot_files = get_robot_files(folder_path)
    all_graph_data = process_xml_files(robot_files, all_graph_data)

    # Generate graph embeddings
    graph_embeddings = generate_graph_embeddings(all_graph_data)
    
    cluster_medians, median_files = visualize_graph_embeddings(graph_embeddings, robot_files, folder_path)
    # print("cluster_medians: ", cluster_medians)
    # print("median_files: ", median_files)

if __name__ == "__main__":
    main()