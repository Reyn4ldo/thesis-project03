"""
Unsupervised outlier detection methods.

Provides multiple algorithms for detecting anomalous isolates:
- Isolation Forest
- Local Outlier Factor (LOF)
- DBSCAN-based outlier detection
- Mahalanobis distance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
import warnings
warnings.filterwarnings('ignore')


class OutlierDetector:
    """
    Multi-method outlier detection for antibiotic resistance data.
    
    Identifies rare, extreme, or inconsistent isolates using multiple
    unsupervised algorithms.
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        """
        Initialize outlier detector.
        
        Parameters
        ----------
        contamination : float, default=0.05
            Expected proportion of outliers (0-0.5)
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.X_scaled = None
        
        # Models
        self.isolation_forest = None
        self.lof = None
        self.dbscan = None
        self.mahalanobis_cov = None
        
        # Results
        self.results = {}
    
    def prepare_data(self, df):
        """
        Prepare data for outlier detection.
        
        Parameters
        ----------
        df : pd.DataFrame
            Processed dataframe
            
        Returns
        -------
        np.ndarray
            Scaled feature matrix
        list
            Feature names
        """
        # Select numeric features for outlier detection
        # Exclude binary flags and imputation indicators
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remove imputation flags
        feature_cols = [col for col in numeric_cols 
                       if not col.endswith('_imputed') 
                       and not col.endswith('_missing')]
        
        # Prefer antibiogram and aggregate features
        priority_patterns = ['antibiogram_', 'resistant', 'mar_index', 
                           'total_', 'ratio_', 'who_']
        
        feature_cols = [col for col in feature_cols 
                       if any(pattern in col for pattern in priority_patterns)]
        
        X = df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0)
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return self.X_scaled, feature_cols
    
    def fit_isolation_forest(self, X):
        """
        Fit Isolation Forest for anomaly detection.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
            
        Returns
        -------
        dict
            Detection results with scores and labels
        """
        print(f"\nFitting Isolation Forest (contamination={self.contamination})...")
        
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        # Fit and predict (-1 for outliers, 1 for inliers)
        labels = self.isolation_forest.fit_predict(X)
        
        # Get anomaly scores (lower is more anomalous)
        scores = self.isolation_forest.score_samples(X)
        
        # Convert to outlier probability (higher is more anomalous)
        outlier_scores = -scores  # Negate so higher is more anomalous
        
        n_outliers = (labels == -1).sum()
        
        results = {
            'method': 'isolation_forest',
            'labels': labels,
            'scores': outlier_scores,
            'n_outliers': n_outliers,
            'outlier_indices': np.where(labels == -1)[0],
            'outlier_rate': n_outliers / len(labels)
        }
        
        self.results['isolation_forest'] = results
        
        print(f"  Outliers detected: {n_outliers} ({results['outlier_rate']:.2%})")
        print(f"  Outlier score range: [{outlier_scores.min():.3f}, {outlier_scores.max():.3f}]")
        
        return results
    
    def fit_lof(self, X, n_neighbors=20):
        """
        Fit Local Outlier Factor.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        n_neighbors : int, default=20
            Number of neighbors for LOF
            
        Returns
        -------
        dict
            Detection results
        """
        print(f"\nFitting Local Outlier Factor (n_neighbors={n_neighbors})...")
        
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination
        )
        
        # Fit and predict
        labels = self.lof.fit_predict(X)
        
        # Get negative outlier factor
        # Note: More negative values indicate outliers
        # We negate to make higher scores more anomalous for consistency
        scores = -self.lof.negative_outlier_factor_
        
        n_outliers = (labels == -1).sum()
        
        results = {
            'method': 'lof',
            'labels': labels,
            'scores': scores,
            'n_outliers': n_outliers,
            'outlier_indices': np.where(labels == -1)[0],
            'outlier_rate': n_outliers / len(labels)
        }
        
        self.results['lof'] = results
        
        print(f"  Outliers detected: {n_outliers} ({results['outlier_rate']:.2%})")
        print(f"  LOF score range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        return results
    
    def fit_dbscan_outliers(self, X, eps=0.5, min_samples=5):
        """
        Use DBSCAN to identify outliers (noise points).
        
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
            Detection results
        """
        print(f"\nFitting DBSCAN outlier detection (eps={eps}, min_samples={min_samples})...")
        
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        
        labels = self.dbscan.fit_predict(X)
        
        # Label -1 indicates noise/outliers
        outlier_mask = labels == -1
        n_outliers = outlier_mask.sum()
        
        # Create binary scores (1 for outliers, 0 for inliers)
        scores = outlier_mask.astype(float)
        
        results = {
            'method': 'dbscan',
            'labels': labels,
            'scores': scores,
            'n_outliers': n_outliers,
            'outlier_indices': np.where(outlier_mask)[0],
            'outlier_rate': n_outliers / len(labels),
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
        }
        
        self.results['dbscan'] = results
        
        print(f"  Outliers detected: {n_outliers} ({results['outlier_rate']:.2%})")
        print(f"  Clusters found: {results['n_clusters']}")
        
        return results
    
    def compute_mahalanobis_distance(self, X):
        """
        Compute Mahalanobis distance for each sample.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
            
        Returns
        -------
        dict
            Distance scores and outlier labels
        """
        print(f"\nComputing Mahalanobis distances...")
        
        # Compute covariance matrix
        self.mahalanobis_cov = EmpiricalCovariance().fit(X)
        
        # Compute Mahalanobis distance
        distances = self.mahalanobis_cov.mahalanobis(X)
        
        # Determine threshold (e.g., 97.5th percentile)
        threshold = np.percentile(distances, 100 * (1 - self.contamination))
        
        # Label outliers
        labels = np.where(distances > threshold, -1, 1)
        n_outliers = (labels == -1).sum()
        
        results = {
            'method': 'mahalanobis',
            'labels': labels,
            'scores': distances,
            'threshold': threshold,
            'n_outliers': n_outliers,
            'outlier_indices': np.where(labels == -1)[0],
            'outlier_rate': n_outliers / len(labels)
        }
        
        self.results['mahalanobis'] = results
        
        print(f"  Outliers detected: {n_outliers} ({results['outlier_rate']:.2%})")
        print(f"  Distance threshold: {threshold:.3f}")
        print(f"  Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
        
        return results
    
    def fit_all(self, X):
        """
        Fit all outlier detection methods.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
            
        Returns
        -------
        dict
            Results from all methods
        """
        self.fit_isolation_forest(X)
        self.fit_lof(X)
        self.fit_dbscan_outliers(X)
        self.compute_mahalanobis_distance(X)
        
        return self.results
    
    def get_consensus_outliers(self, min_methods=2):
        """
        Get outliers identified by multiple methods.
        
        Parameters
        ----------
        min_methods : int, default=2
            Minimum number of methods that must agree
            
        Returns
        -------
        np.ndarray
            Indices of consensus outliers
        dict
            Outlier counts per sample
        """
        if not self.results:
            raise ValueError("Must run outlier detection first")
        
        # Count how many methods label each sample as outlier
        n_samples = len(self.results['isolation_forest']['labels'])
        outlier_counts = np.zeros(n_samples, dtype=int)
        
        for method_name, results in self.results.items():
            outlier_mask = results['labels'] == -1
            outlier_counts += outlier_mask.astype(int)
        
        # Get consensus outliers
        consensus_mask = outlier_counts >= min_methods
        consensus_indices = np.where(consensus_mask)[0]
        
        print(f"\nConsensus Outliers (≥{min_methods} methods):")
        print(f"  Found {len(consensus_indices)} consensus outliers")
        print(f"  Distribution of votes:")
        for i in range(len(self.results), 0, -1):
            count = (outlier_counts == i).sum()
            if count > 0:
                print(f"    {i} methods: {count} samples")
        
        return consensus_indices, outlier_counts
    
    def get_top_outliers(self, n=20, method='isolation_forest'):
        """
        Get top N most anomalous samples by a specific method.
        
        Parameters
        ----------
        n : int, default=20
            Number of top outliers to return
        method : str, default='isolation_forest'
            Method to use for ranking
            
        Returns
        -------
        np.ndarray
            Indices of top outliers
        np.ndarray
            Scores of top outliers
        """
        if method not in self.results:
            raise ValueError(f"Method {method} not fitted")
        
        scores = self.results[method]['scores']
        
        # Get indices of top n scores (highest are most anomalous)
        top_indices = np.argsort(scores)[-n:][::-1]
        top_scores = scores[top_indices]
        
        return top_indices, top_scores
    
    def get_summary(self):
        """
        Get summary of outlier detection results.
        
        Returns
        -------
        dict
            Summary statistics for all methods
        """
        summary = {}
        
        for method_name, results in self.results.items():
            summary[method_name] = {
                'n_outliers': results['n_outliers'],
                'outlier_rate': results['outlier_rate'],
                'score_mean': results['scores'].mean(),
                'score_std': results['scores'].std(),
                'score_min': results['scores'].min(),
                'score_max': results['scores'].max()
            }
        
        return summary
