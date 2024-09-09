import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid
import numpy as np

# Load the data from CSV
input_file = 'input.csv'  # Replace with your actual file path
data = pd.read_csv(input_file)

# Define columns for clustering
ecfp_cols = [col for col in data.columns if col.startswith('ECFP4_')]
X = data[ecfp_cols].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to find the best number of clusters for Spectral Clustering
def find_best_spectral_clustering_params(X):
    best_score = -1
    best_n_clusters = None
    
    for n_clusters in range(2, 21):  # Test from 2 to 20 clusters
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10)
        labels = spectral.fit_predict(X)
        
        score = silhouette_score(X, labels)
        print(f"Number of clusters: {n_clusters}, Silhouette Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    
    return best_n_clusters

# Find the optimal number of clusters
best_n_clusters = find_best_spectral_clustering_params(X_scaled)

# Apply Spectral Clustering with the optimal number of clusters
spectral_clustering = SpectralClustering(n_clusters=best_n_clusters, affinity='nearest_neighbors', n_neighbors=10)
data['Spectral_Cluster'] = spectral_clustering.fit_predict(X_scaled)

# Save the data with Spectral Clustering results
output_file = 'output.csv'
data.to_csv(output_file, index=False)
print(f"Data with Spectral Clustering results saved to {output_file}")
