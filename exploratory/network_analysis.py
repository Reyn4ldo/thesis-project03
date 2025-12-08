"""
Network analysis module for co-resistance relationships.

Creates and analyzes co-resistance networks where:
- Nodes are antibiotics
- Edges represent co-resistance relationships
- Edge weights reflect co-resistance frequency or strength
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class CoResistanceNetwork:
    """
    Co-resistance network analysis.
    
    Builds networks from resistance data and analyzes:
    - Network topology (centrality, density, clustering)
    - Community structure (Louvain algorithm)
    - Hub antibiotics
    """
    
    def __init__(self, min_edge_weight=0.1):
        """
        Initialize network analyzer.
        
        Parameters
        ----------
        min_edge_weight : float, default=0.1
            Minimum edge weight to include in network
        """
        self.min_edge_weight = min_edge_weight
        
        self.graph = None
        self.communities = None
        self.centrality_measures = {}
    
    def build_network(self, transactions, method='frequency', compute_pvalues=False):
        """
        Build co-resistance network from transaction data.
        
        Parameters
        ----------
        transactions : pd.DataFrame
            Binary transaction dataframe
        method : str, default='frequency'
            Method to compute edge weights:
            - 'frequency': co-resistance frequency
            - 'odds_ratio': adjusted odds ratio
            - 'correlation': correlation coefficient
        compute_pvalues : bool, default=False
            Whether to compute p-values using Fisher's exact test
            
        Returns
        -------
        networkx.Graph
            Co-resistance network
        """
        print(f"\nBuilding co-resistance network (method={method})...")
        
        antibiotics = list(transactions.columns)
        n_samples = len(transactions)
        
        # Initialize graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(antibiotics)
        
        # Compute edge weights
        edges_added = 0
        
        for i, ab1 in enumerate(antibiotics):
            for j in range(i+1, len(antibiotics)):
                ab2 = antibiotics[j]
                
                # Count co-occurrences
                both_resistant = (transactions[ab1] & transactions[ab2]).sum()
                ab1_only = (transactions[ab1] & ~transactions[ab2]).sum()
                ab2_only = (~transactions[ab1] & transactions[ab2]).sum()
                neither = (~transactions[ab1] & ~transactions[ab2]).sum()
                
                # Calculate edge weight based on method
                if method == 'frequency':
                    weight = both_resistant / n_samples
                
                elif method == 'odds_ratio':
                    # Adjusted odds ratio (add 0.5 to avoid division by zero)
                    odds_ratio = ((both_resistant + 0.5) * (neither + 0.5)) / \
                                ((ab1_only + 0.5) * (ab2_only + 0.5))
                    weight = np.log(odds_ratio)  # Log odds ratio
                
                elif method == 'correlation':
                    # Phi coefficient (correlation for binary data)
                    numerator = both_resistant * neither - ab1_only * ab2_only
                    denominator = np.sqrt((both_resistant + ab1_only) * 
                                         (both_resistant + ab2_only) *
                                         (ab2_only + neither) * 
                                         (ab1_only + neither))
                    weight = numerator / denominator if denominator > 0 else 0
                
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Add edge if above threshold
                if weight >= self.min_edge_weight:
                    edge_attrs = {'weight': weight, 'co_resistance_count': both_resistant}
                    
                    # Optionally compute p-value
                    if compute_pvalues:
                        contingency = [[both_resistant, ab1_only],
                                      [ab2_only, neither]]
                        _, pvalue = fisher_exact(contingency, alternative='greater')
                        edge_attrs['pvalue'] = pvalue
                    
                    self.graph.add_edge(ab1, ab2, **edge_attrs)
                    edges_added += 1
        
        print(f"  Network created: {len(antibiotics)} nodes, {edges_added} edges")
        print(f"  Network density: {nx.density(self.graph):.4f}")
        
        return self.graph
    
    def compute_centrality(self):
        """
        Compute various centrality measures.
        
        Returns
        -------
        dict
            Dictionary of centrality measures
        """
        if self.graph is None:
            raise ValueError("Must build network first")
        
        print("\nComputing centrality measures...")
        
        self.centrality_measures = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph, weight='weight'),
            'closeness': nx.closeness_centrality(self.graph, distance='weight'),
            'eigenvector': nx.eigenvector_centrality(self.graph, weight='weight', max_iter=1000)
        }
        
        # Convert to dataframe for easier analysis
        centrality_df = pd.DataFrame(self.centrality_measures)
        centrality_df = centrality_df.sort_values('degree', ascending=False)
        
        print(f"  Computed {len(self.centrality_measures)} centrality measures")
        
        return centrality_df
    
    def detect_communities(self, resolution=1.0):
        """
        Detect communities using Louvain algorithm.
        
        Parameters
        ----------
        resolution : float, default=1.0
            Resolution parameter for community detection
            
        Returns
        -------
        dict
            Community assignments
        """
        if self.graph is None:
            raise ValueError("Must build network first")
        
        print(f"\nDetecting communities (resolution={resolution})...")
        
        # Use Louvain algorithm
        try:
            import community as community_louvain
            self.communities = community_louvain.best_partition(
                self.graph, 
                weight='weight',
                resolution=resolution
            )
        except ImportError:
            # Fallback to greedy modularity communities
            print("  Warning: python-louvain not installed, using greedy modularity")
            communities_gen = nx.community.greedy_modularity_communities(
                self.graph, weight='weight'
            )
            self.communities = {}
            for i, community in enumerate(communities_gen):
                for node in community:
                    self.communities[node] = i
        
        n_communities = len(set(self.communities.values()))
        print(f"  Found {n_communities} communities")
        
        # Print community composition
        for comm_id in range(n_communities):
            members = [node for node, comm in self.communities.items() if comm == comm_id]
            print(f"    Community {comm_id}: {len(members)} antibiotics - {', '.join(members[:5])}" +
                  (f" and {len(members)-5} more" if len(members) > 5 else ""))
        
        return self.communities
    
    def get_hub_antibiotics(self, n=10, centrality_measure='degree'):
        """
        Identify hub antibiotics based on centrality.
        
        Parameters
        ----------
        n : int, default=10
            Number of hubs to return
        centrality_measure : str, default='degree'
            Centrality measure to use
            
        Returns
        -------
        pd.Series
            Top hub antibiotics
        """
        if centrality_measure not in self.centrality_measures:
            raise ValueError(f"Must compute centrality first. Available: {list(self.centrality_measures.keys())}")
        
        centrality = pd.Series(self.centrality_measures[centrality_measure])
        return centrality.nlargest(n)
    
    def plot_network(self, layout='spring', node_color_by='degree', 
                    node_size_by='degree', show_labels=True,
                    save_path=None):
        """
        Visualize the co-resistance network.
        
        Parameters
        ----------
        layout : str, default='spring'
            Layout algorithm ('spring', 'circular', 'kamada_kawai')
        node_color_by : str, default='degree'
            Node attribute to color by ('degree', 'betweenness', 'community')
        node_size_by : str, default='degree'
            Node attribute to size by
        show_labels : bool, default=True
            Whether to show node labels
        save_path : str, optional
            Path to save figure
        """
        if self.graph is None:
            raise ValueError("Must build network first")
        
        print(f"\nPlotting network with {layout} layout...")
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, weight='weight', seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph, weight='weight')
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        # Determine node colors
        if node_color_by == 'community' and self.communities:
            node_colors = [self.communities[node] for node in self.graph.nodes()]
            cmap = plt.cm.tab20
        elif node_color_by in self.centrality_measures:
            node_colors = [self.centrality_measures[node_color_by][node] 
                          for node in self.graph.nodes()]
            cmap = plt.cm.viridis
        else:
            node_colors = 'skyblue'
            cmap = None
        
        # Determine node sizes
        if node_size_by in self.centrality_measures:
            node_sizes = [self.centrality_measures[node_size_by][node] * 3000 
                         for node in self.graph.nodes()]
        else:
            node_sizes = 500
        
        # Plot
        plt.figure(figsize=(14, 10))
        
        # Draw edges
        edge_weights = [self.graph[u][v]['weight'] for u, v in self.graph.edges()]
        nx.draw_networkx_edges(self.graph, pos, width=edge_weights, 
                              alpha=0.3, edge_color='gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors,
                              node_size=node_sizes, cmap=cmap, alpha=0.8)
        
        # Draw labels
        if show_labels:
            nx.draw_networkx_labels(self.graph, pos, font_size=8)
        
        plt.title('Co-Resistance Network')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_network_statistics(self):
        """
        Get comprehensive network statistics.
        
        Returns
        -------
        dict
            Network statistics
        """
        if self.graph is None:
            raise ValueError("Must build network first")
        
        stats = {
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'average_clustering': nx.average_clustering(self.graph, weight='weight'),
            'n_connected_components': nx.number_connected_components(self.graph)
        }
        
        # Add community stats if available
        if self.communities:
            stats['n_communities'] = len(set(self.communities.values()))
            
            # Modularity
            communities_list = []
            for comm_id in set(self.communities.values()):
                members = [node for node, c in self.communities.items() if c == comm_id]
                communities_list.append(members)
            stats['modularity'] = nx.community.modularity(self.graph, communities_list, weight='weight')
        
        return stats
    
    def export_network(self, filepath, format='gexf'):
        """
        Export network to file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        format : str, default='gexf'
            Output format ('gexf', 'graphml', 'gml')
        """
        if self.graph is None:
            raise ValueError("Must build network first")
        
        if format == 'gexf':
            nx.write_gexf(self.graph, filepath)
        elif format == 'graphml':
            nx.write_graphml(self.graph, filepath)
        elif format == 'gml':
            nx.write_gml(self.graph, filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"\nNetwork exported to {filepath}")
