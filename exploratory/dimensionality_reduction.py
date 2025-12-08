"""
Dimensionality reduction and visualization module.

Provides:
- PCA for variance analysis
- t-SNE for 2D visualization
- UMAP for topological visualization
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class DimensionalityReducer:
    """
    Dimensionality reduction for antibiogram visualization.
    
    Supports PCA, t-SNE, and UMAP for different visualization needs.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize reducer.
        
        Parameters
        ----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        
        self.pca_model = None
        self.tsne_model = None
        self.umap_model = None
        
        self.results = {}
    
    def fit_pca(self, X, n_components=None):
        """
        Fit PCA for variance analysis.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        n_components : int, optional
            Number of components (default: min(n_samples, n_features))
            
        Returns
        -------
        dict
            PCA results including transformed data and variance explained
        """
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1], 50)  # Cap at 50 for efficiency
        
        print(f"\nFitting PCA with {n_components} components...")
        
        self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca_model.fit_transform(X)
        
        # Calculate cumulative variance explained
        cumsum_variance = np.cumsum(self.pca_model.explained_variance_ratio_)
        
        # Find number of components for 95% variance
        n_95 = np.argmax(cumsum_variance >= 0.95) + 1
        
        results = {
            'method': 'pca',
            'X_reduced': X_pca[:, :2],  # First 2 components for visualization
            'X_all_components': X_pca,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'cumulative_variance': cumsum_variance,
            'n_components_95': n_95,
            'components': self.pca_model.components_
        }
        
        self.results['pca'] = results
        
        print(f"  Variance explained by first 2 components: {cumsum_variance[1]:.2%}")
        print(f"  Components needed for 95% variance: {n_95}")
        
        return results
    
    def fit_tsne(self, X, n_components=2, perplexity=30):
        """
        Fit t-SNE for 2D visualization.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        n_components : int, default=2
            Number of dimensions for embedding
        perplexity : float, default=30
            t-SNE perplexity parameter
            
        Returns
        -------
        dict
            t-SNE results
        """
        print(f"\nFitting t-SNE (perplexity={perplexity})...")
        
        # Use PCA preprocessing if high-dimensional
        if X.shape[1] > 50:
            print("  Applying PCA preprocessing (50 components)...")
            pca_pre = PCA(n_components=50, random_state=self.random_state)
            X = pca_pre.fit_transform(X)
        
        self.tsne_model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=self.random_state,
            n_iter=1000
        )
        
        X_tsne = self.tsne_model.fit_transform(X)
        
        results = {
            'method': 'tsne',
            'X_reduced': X_tsne,
            'perplexity': perplexity
        }
        
        self.results['tsne'] = results
        
        print(f"  t-SNE embedding completed")
        
        return results
    
    def fit_umap(self, X, n_components=2, n_neighbors=15, min_dist=0.1):
        """
        Fit UMAP for topological visualization.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        n_components : int, default=2
            Number of dimensions for embedding
        n_neighbors : int, default=15
            Number of neighbors for local structure
        min_dist : float, default=0.1
            Minimum distance between points
            
        Returns
        -------
        dict
            UMAP results
        """
        print(f"\nFitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        
        self.umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=self.random_state
        )
        
        X_umap = self.umap_model.fit_transform(X)
        
        results = {
            'method': 'umap',
            'X_reduced': X_umap,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist
        }
        
        self.results['umap'] = results
        
        print(f"  UMAP embedding completed")
        
        return results
    
    def fit_all(self, X):
        """
        Fit all dimensionality reduction methods.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
            
        Returns
        -------
        dict
            Results from all methods
        """
        self.fit_pca(X)
        self.fit_tsne(X)
        self.fit_umap(X)
        
        return self.results
    
    def plot_pca_variance(self, save_path=None):
        """Plot PCA variance explained."""
        if 'pca' not in self.results:
            raise ValueError("PCA must be fitted first")
        
        pca_results = self.results['pca']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Individual variance
        n_components = len(pca_results['explained_variance_ratio'])
        axes[0].bar(range(1, min(21, n_components+1)), 
                   pca_results['explained_variance_ratio'][:20])
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Variance Explained')
        axes[0].set_title('Variance Explained by Component')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Cumulative variance
        axes[1].plot(range(1, n_components+1), 
                    pca_results['cumulative_variance'], 'o-')
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Variance Explained')
        axes[1].set_title('Cumulative Variance Explained')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_embedding(self, method='umap', color_by=None, df=None, 
                      title=None, save_path=None):
        """
        Plot 2D embedding colored by metadata.
        
        Parameters
        ----------
        method : str
            Method to plot ('pca', 'tsne', 'umap')
        color_by : str, optional
            Column name to color points by
        df : pd.DataFrame, optional
            Dataframe with metadata for coloring
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
        """
        if method not in self.results:
            raise ValueError(f"{method} must be fitted first")
        
        X_reduced = self.results[method]['X_reduced']
        
        plt.figure(figsize=(10, 8))
        
        if color_by and df is not None:
            # Color by categorical variable
            if df[color_by].dtype == 'object' or df[color_by].nunique() < 20:
                categories = df[color_by].unique()
                colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
                
                for i, cat in enumerate(categories):
                    mask = df[color_by] == cat
                    plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                              c=[colors[i]], label=str(cat), alpha=0.6, s=30)
                
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          fontsize='small')
            else:
                # Color by continuous variable
                scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=df[color_by], cmap='viridis', 
                                    alpha=0.6, s=30)
                plt.colorbar(scatter, label=color_by)
        else:
            # No coloring
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=30)
        
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'{method.upper()} Visualization' + 
                     (f' (colored by {color_by})' if color_by else ''))
        
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_all_embeddings(self, df, color_by='bacterial_species', save_dir=None):
        """
        Plot all embedding methods side by side.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with metadata
        color_by : str
            Column to color by
        save_dir : str, optional
            Directory to save figures
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        methods = ['pca', 'tsne', 'umap']
        
        for idx, method in enumerate(methods):
            if method not in self.results:
                continue
            
            X_reduced = self.results[method]['X_reduced']
            
            if color_by in df.columns:
                categories = df[color_by].unique()
                colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
                
                for i, cat in enumerate(categories):
                    mask = df[color_by] == cat
                    axes[idx].scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                                    c=[colors[i]], label=str(cat), alpha=0.6, s=20)
            else:
                axes[idx].scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=20)
            
            axes[idx].set_xlabel(f'{method.upper()} Dimension 1')
            axes[idx].set_ylabel(f'{method.upper()} Dimension 2')
            axes[idx].set_title(f'{method.upper()} Visualization')
            axes[idx].grid(alpha=0.3)
        
        # Add legend to the right
        if color_by in df.columns and df[color_by].nunique() < 15:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),
                      fontsize='small')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f'all_embeddings_{color_by}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
