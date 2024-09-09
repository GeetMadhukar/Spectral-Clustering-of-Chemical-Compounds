# Spectral-Clustering-of-Chemical-Compounds

**Spectral Clustering of Chemical Compounds Based on ECFP4 Fingerprints**
This Python script performs spectral clustering on chemical compounds using Extended Connectivity Fingerprints (ECFP4) to identify groups of structurally similar molecules. The script reads compound data from a CSV file, processes the ECFP4 fingerprints, and finds the optimal number of clusters using silhouette scores. It then applies spectral clustering and saves the results to CSV files, including both Spectral and K-means clustering results.

**Key Features:**

1. Data Preprocessing: Reads compound data from a CSV file and extracts ECFP4 fingerprint columns for clustering.

2. Spectral Clustering: Automatically determines the optimal number of clusters (2 to 20) by maximizing the silhouette score and applies spectral clustering.

**Cluster Results:**

The script outputs a CSV file:
output.csv: Contains the compounds along with their spectral cluster assignments.

**Standardization:**
Scales the ECFP4 fingerprints using StandardScaler for better clustering performance.

**Dependencies:**

pandas
scikit-learn
numpy
