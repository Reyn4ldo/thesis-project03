"""
Clustering module for antibiogram analysis.

Provides multiple clustering algorithms:
- K-means clustering
- Hierarchical clustering
- DBSCAN for density-based clustering

Includes stability analysis and cluster quality metrics.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class AntibiogramClusterer:
    """
    Clustering analysis for antibiogram patterns.
    
    Identifies antibiotypes (groups of isolates with similar resistance patterns)
    using multiple clustering algorithms.
    """
    
    def __init__(self, n_clusters=5, random_state=42):
        """
        Initialize clusterer.
        
        Parameters
        ----------
        n_clusters : int, default=5
            Number of clusters for k-means and hierarchical clustering
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.kmeans_model = None
        self.hierarchical_model = None
        self.dbscan_model = None
        
        self.scaler = StandardScaler()
        self.X_scaled = None
        
        self.results = {}
    
    def prepare_data(self, df):
        """
        Prepare antibiogram data for clustering.
        
        Parameters
        ----------
        df : pd.DataFrame
            Processed dataframe with antibiogram features
            
        Returns
        -------
        np.ndarray
            Feature matrix for clustering
        """
        # Use antibiogram features (resistance vectors)
        antibiogram_cols = [col for col in df.columns if 'antibiogram_' in col]
        
        if not antibiogram_cols:
            # Fallback to binary resistance indicators
            antibiogram_cols = [col for col in df.columns 
                               if '_resistant' in col and 'total' not in col]
        
        X = df[antibiogram_cols].values
        
        # Remove any NaN values
        X = np.nan_to_num(X, nan=-1)
        
        # Scale the features
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return self.X_scaled, antibiogram_cols
    
    def fit_kmeans(self, X, n_clusters=None):
        """
        Fit K-means clustering.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        n_clusters : int, optional
            Number of clusters (overrides default)
            
        Returns
        -------
        dict
            Clustering results with labels and metrics
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        print(f"\nFitting K-means with {n_clusters} clusters...")
        
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        labels = self.kmeans_model.fit_predict(X)
        
        # Compute metrics
        silhouette = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        
        results = {
            'method': 'kmeans',
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_index': db_index,
            'calinski_harabasz_score': ch_score,
            'cluster_sizes': np.bincount(labels),
            'centroids': self.kmeans_model.cluster_centers_
        }
        
        self.results['kmeans'] = results
        
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Index: {db_index:.4f}")
        print(f"  Calinski-Harabasz Score: {ch_score:.2f}")
        print(f"  Cluster sizes: {results['cluster_sizes']}")
        
        return results
    
    def fit_hierarchical(self, X, n_clusters=None, linkage_method='ward'):
        """
        Fit hierarchical clustering.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        n_clusters : int, optional
            Number of clusters
        linkage_method : str, default='ward'
            Linkage method ('ward', 'complete', 'average', 'single')
            
        Returns
        -------
        dict
            Clustering results
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        print(f"\nFitting Hierarchical clustering ({linkage_method}) with {n_clusters} clusters...")
        
        self.hierarchical_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        
        labels = self.hierarchical_model.fit_predict(X)
        
        # Compute metrics
        silhouette = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        
        results = {
            'method': 'hierarchical',
            'linkage': linkage_method,
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_index': db_index,
            'calinski_harabasz_score': ch_score,
            'cluster_sizes': np.bincount(labels)
        }
        
        self.results['hierarchical'] = results
        
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Index: {db_index:.4f}")
        print(f"  Calinski-Harabasz Score: {ch_score:.2f}")
        print(f"  Cluster sizes: {results['cluster_sizes']}")
        
        return results
    
    def fit_dbscan(self, X, eps=0.5, min_samples=5):
        """
        Fit DBSCAN clustering.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        eps : float, default=0.5
            Maximum distance between samples
        min_samples : int, default=5
            Minimum samples in neighborhood
            
        Returns
        -------
        dict
            Clustering results
        """
        print(f"\nFitting DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        
        labels = self.dbscan_model.fit_predict(X)
        
        # Number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Compute metrics (excluding noise points)
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X[mask], labels[mask])
                db_index = davies_bouldin_score(X[mask], labels[mask])
                ch_score = calinski_harabasz_score(X[mask], labels[mask])
            else:
                silhouette = db_index = ch_score = np.nan
        else:
            silhouette = db_index = ch_score = np.nan
        
        results = {
            'method': 'dbscan',
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette,
            'davies_bouldin_index': db_index,
            'calinski_harabasz_score': ch_score,
            'cluster_sizes': np.bincount(labels[labels >= 0]) if n_clusters > 0 else np.array([])
        }
        
        self.results['dbscan'] = results
        
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        if not np.isnan(silhouette):
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Davies-Bouldin Index: {db_index:.4f}")
            print(f"  Calinski-Harabasz Score: {ch_score:.2f}")
        print(f"  Cluster sizes: {results['cluster_sizes']}")
        
        return results
    
    def fit_all(self, X):
        """
        Fit all clustering methods.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
            
        Returns
        -------
        dict
            Results from all methods
        """
        self.fit_kmeans(X)
        self.fit_hierarchical(X)
        self.fit_dbscan(X)
        
        return self.results
    
    def find_optimal_k(self, X, k_range=range(2, 11)):
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        k_range : range
            Range of k values to test
            
        Returns
        -------
        dict
            Metrics for each k value
        """
        print("\nFinding optimal number of clusters...")
        
        inertias = []
        silhouettes = []
        db_indices = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
            db_indices.append(davies_bouldin_score(X, labels))
        
        results = {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouettes,
            'davies_bouldin_indices': db_indices
        }
        
        # Find optimal k (highest silhouette)
        optimal_k = k_range[np.argmax(silhouettes)]
        print(f"Optimal k (by silhouette): {optimal_k}")
        
        return results, optimal_k
    
    def plot_dendrogram(self, X, save_path=None):
        """Plot hierarchical clustering dendrogram."""
        print("\nGenerating dendrogram...")
        
        linkage_matrix = linkage(X, method='ward')
        
        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, no_labels=True)
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.title('Hierarchical Clustering Dendrogram')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_elbow_curve(self, elbow_results, save_path=None):
        """Plot elbow curve for optimal k selection."""
        k_values = elbow_results['k_values']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Inertia plot
        axes[0].plot(k_values, elbow_results['inertias'], 'o-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Curve')
        axes[0].grid(alpha=0.3)
        
        # Silhouette plot
        axes[1].plot(k_values, elbow_results['silhouette_scores'], 'o-', color='green')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        axes[1].grid(alpha=0.3)
        
        # Davies-Bouldin plot
        axes[2].plot(k_values, elbow_results['davies_bouldin_indices'], 'o-', color='red')
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Davies-Bouldin Index')
        axes[2].set_title('Davies-Bouldin Index (lower is better)')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_cluster_profiles(self, df, labels, method_name='kmeans'):
        """
        Generate cluster profiles.
        
        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe with metadata
        labels : array-like
            Cluster labels
        method_name : str
            Name of clustering method
            
        Returns
        -------
        dict
            Cluster profiles with statistics
        """
        df_with_clusters = df.copy()
        df_with_clusters[f'{method_name}_cluster'] = labels
        
        profiles = {}
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_data = df_with_clusters[df_with_clusters[f'{method_name}_cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'prevalence': len(cluster_data) / len(df) * 100
            }
            
            # Species distribution
            if 'bacterial_species' in df.columns:
                profile['species_distribution'] = cluster_data['bacterial_species'].value_counts().to_dict()
            
            # Region distribution
            if 'administrative_region' in df.columns:
                profile['region_distribution'] = cluster_data['administrative_region'].value_counts().to_dict()
            
            # Resistance profile (mean resistance rates)
            resistance_cols = [col for col in df.columns if '_resistant' in col and 'total' not in col]
            if resistance_cols:
                profile['mean_resistance_rates'] = cluster_data[resistance_cols].mean().to_dict()
            
            # MAR index statistics
            if 'mar_index' in df.columns:
                profile['mar_index_mean'] = cluster_data['mar_index'].mean()
                profile['mar_index_std'] = cluster_data['mar_index'].std()
            
            profiles[f'Cluster {cluster_id}'] = profile
        
        return profiles
