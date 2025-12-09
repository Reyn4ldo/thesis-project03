#!/usr/bin/env python3
"""
Phase 3 - Unsupervised & Exploratory Analyses

Performs clustering, dimensionality reduction, association rule mining,
and network analysis to discover antibiotic resistance patterns.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from exploratory import (
    AntibiogramClusterer,
    DimensionalityReducer,
    AssociationRuleMiner,
    CoResistanceNetwork
)
from pathlib import Path


def load_processed_data():
    """Load preprocessed data."""
    print("="*80)
    print("LOADING PROCESSED DATA")
    print("="*80)
    
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    
    # Combine for exploratory analysis
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"\nTotal samples: {len(full_df)}")
    print(f"Total features: {len(full_df.columns)}")
    
    return full_df


def run_clustering_analysis(df):
    """Run clustering analysis on antibiograms."""
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS")
    print("="*80)
    
    # Initialize clusterer
    clusterer = AntibiogramClusterer(n_clusters=5, random_state=42)
    
    # Prepare data
    X, feature_names = clusterer.prepare_data(df)
    
    # Find optimal k
    elbow_results, optimal_k = clusterer.find_optimal_k(X, k_range=range(2, 11))
    
    # Fit with optimal k
    print(f"\n\nUsing optimal k={optimal_k} for final clustering...")
    clusterer.n_clusters = optimal_k
    
    # Fit all methods
    results = clusterer.fit_all(X)
    
    # Generate cluster profiles for k-means
    profiles = clusterer.get_cluster_profiles(df, results['kmeans']['labels'], 'kmeans')
    
    # Print cluster profiles
    print("\n" + "-"*80)
    print("CLUSTER PROFILES (K-means)")
    print("-"*80)
    
    for cluster_name, profile in profiles.items():
        print(f"\n{cluster_name}:")
        print(f"  Size: {profile['size']} ({profile['prevalence']:.1f}%)")
        
        if 'species_distribution' in profile:
            print(f"  Top species:")
            for species, count in list(profile['species_distribution'].items())[:3]:
                print(f"    - {species}: {count}")
        
        if 'mar_index_mean' in profile:
            print(f"  MAR index: {profile['mar_index_mean']:.3f} ± {profile['mar_index_std']:.3f}")
    
    return clusterer, results


def run_dimensionality_reduction(df, clusterer):
    """Run dimensionality reduction and visualization."""
    print("\n" + "="*80)
    print("DIMENSIONALITY REDUCTION")
    print("="*80)
    
    # Initialize reducer
    reducer = DimensionalityReducer(random_state=42)
    
    # Use same data as clustering
    X = clusterer.X_scaled
    
    # Fit all methods
    results = reducer.fit_all(X)
    
    # Print PCA variance info
    pca_results = results['pca']
    print(f"\nPCA Results:")
    print(f"  First 2 components explain: {pca_results['cumulative_variance'][1]:.2%} variance")
    print(f"  Components for 95% variance: {pca_results['n_components_95']}")
    
    return reducer, results


def run_association_rules(df):
    """Run association rule mining."""
    print("\n" + "="*80)
    print("ASSOCIATION RULE MINING")
    print("="*80)
    
    # Initialize miner
    miner = AssociationRuleMiner(min_support=0.1, min_confidence=0.5, min_lift=1.5)
    
    # Prepare transactions
    transactions = miner.prepare_transactions(df)
    
    # Mine frequent itemsets using FP-growth (faster)
    frequent_itemsets = miner.mine_fpgrowth(transactions, min_support=0.1)
    
    print(f"\nTop 10 Frequent Itemsets:")
    for idx, row in frequent_itemsets.head(10).iterrows():
        items = ', '.join(sorted(row['itemsets']))
        print(f"  {idx+1}. {items} (support={row['support']:.3f})")
    
    # Generate rules
    rules = miner.generate_rules(metric='lift', min_threshold=1.5)
    
    # Print top rules
    miner.print_top_rules(n=15, sort_by='lift')
    
    # Get summary statistics
    summary = miner.get_summary_statistics()
    print(f"\n\nSummary Statistics:")
    print(f"  Frequent itemsets: {summary['n_frequent_itemsets']}")
    print(f"  Association rules: {summary['n_rules']}")
    if 'rule_stats' in summary:
        print(f"  Mean lift: {summary['rule_stats']['mean_lift']:.3f}")
        print(f"  Max lift: {summary['rule_stats']['max_lift']:.3f}")
    
    return miner, transactions


def run_network_analysis(transactions):
    """Run co-resistance network analysis."""
    print("\n" + "="*80)
    print("NETWORK ANALYSIS")
    print("="*80)
    
    # Initialize network
    network = CoResistanceNetwork(min_edge_weight=0.15)
    
    # Build network using frequency method
    graph = network.build_network(transactions, method='frequency')
    
    # Compute centrality
    centrality_df = network.compute_centrality()
    
    print(f"\nTop 10 Hub Antibiotics (by degree centrality):")
    for idx, (antibiotic, degree) in enumerate(centrality_df['degree'].head(10).items(), 1):
        print(f"  {idx}. {antibiotic}: {degree:.4f}")
    
    # Detect communities
    communities = network.detect_communities(resolution=1.0)
    
    # Get network statistics
    stats = network.get_network_statistics()
    
    print(f"\n\nNetwork Statistics:")
    print(f"  Nodes: {stats['n_nodes']}")
    print(f"  Edges: {stats['n_edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Average degree: {stats['average_degree']:.2f}")
    print(f"  Average clustering: {stats['average_clustering']:.4f}")
    print(f"  Communities: {stats.get('n_communities', 'N/A')}")
    if 'modularity' in stats:
        print(f"  Modularity: {stats['modularity']:.4f}")
    
    return network, centrality_df


def generate_phase3_report(clusterer, reducer, miner, network):
    """Generate comprehensive Phase 3 summary report."""
    print("\n" + "="*80)
    print("GENERATING PHASE 3 SUMMARY REPORT")
    print("="*80)
    
    report = []
    report.append("# Phase 3 - Unsupervised & Exploratory Analyses Summary\n")
    report.append("="*80 + "\n\n")
    
    report.append("## Analyses Completed\n\n")
    report.append("1. ✅ Clustering Analysis (K-means, Hierarchical, DBSCAN)\n")
    report.append("2. ✅ Dimensionality Reduction (PCA, t-SNE, UMAP)\n")
    report.append("3. ✅ Association Rule Mining (FP-growth)\n")
    report.append("4. ✅ Network Analysis (Co-resistance graphs)\n\n")
    
    # Clustering results
    report.append("## Clustering Results\n\n")
    if clusterer.results:
        for method, results in clusterer.results.items():
            report.append(f"### {method.upper()}\n")
            report.append(f"- Clusters: {results['n_clusters']}\n")
            report.append(f"- Silhouette Score: {results['silhouette_score']:.4f}\n")
            report.append(f"- Davies-Bouldin Index: {results['davies_bouldin_index']:.4f}\n")
            report.append(f"- Cluster sizes: {list(results['cluster_sizes'])}\n\n")
    
    # Dimensionality reduction
    report.append("## Dimensionality Reduction\n\n")
    if 'pca' in reducer.results:
        pca = reducer.results['pca']
        report.append(f"### PCA\n")
        report.append(f"- First 2 components variance: {pca['cumulative_variance'][1]:.2%}\n")
        report.append(f"- Components for 95% variance: {pca['n_components_95']}\n\n")
    
    # Association rules
    report.append("## Association Rules\n\n")
    if miner.rules is not None:
        report.append(f"- Frequent itemsets found: {len(miner.frequent_itemsets)}\n")
        report.append(f"- Association rules generated: {len(miner.rules)}\n")
        
        report.append(f"\n### Top 5 Rules (by lift):\n")
        for idx, row in miner.get_top_rules(5).iterrows():
            report.append(f"{idx+1}. {miner.format_rule(row)}\n")
        report.append("\n")
    
    # Network analysis
    report.append("## Network Analysis\n\n")
    if network.graph:
        stats = network.get_network_statistics()
        report.append(f"- Nodes (antibiotics): {stats['n_nodes']}\n")
        report.append(f"- Edges (co-resistance links): {stats['n_edges']}\n")
        report.append(f"- Network density: {stats['density']:.4f}\n")
        report.append(f"- Average degree: {stats['average_degree']:.2f}\n")
        report.append(f"- Communities detected: {stats.get('n_communities', 'N/A')}\n")
        if 'modularity' in stats:
            report.append(f"- Modularity: {stats['modularity']:.4f}\n")
        report.append("\n")
    
    report.append("## Deliverables\n\n")
    report.append("- Cluster profiles and antibiotypes identified\n")
    report.append("- Dimensionality reduction embeddings (PCA, t-SNE, UMAP)\n")
    report.append("- Association rules ranked by support, confidence, and lift\n")
    report.append("- Co-resistance network with centrality measures\n")
    report.append("- Community structure identified\n\n")
    
    report.append("## Validation\n\n")
    report.append("- Cluster quality assessed via silhouette score and Davies-Bouldin index\n")
    report.append("- Network modularity computed for community validation\n")
    report.append("- Association rules filtered by minimum thresholds\n\n")
    
    report.append("## Next Steps\n\n")
    report.append("- Create interactive visualizations (UMAP/t-SNE plots)\n")
    report.append("- Bootstrap stability tests for clusters\n")
    report.append("- Expert review of identified antibiotypes\n")
    report.append("- Integrate findings with supervised learning results\n")
    
    report_text = "".join(report)
    
    with open('PHASE3_SUMMARY.md', 'w') as f:
        f.write(report_text)
    
    print("\n✅ Summary report saved to 'PHASE3_SUMMARY.md'")
    print(report_text)


def main():
    """Main execution function for Phase 3."""
    print("\n" + "="*80)
    print("PHASE 3 - UNSUPERVISED & EXPLORATORY ANALYSES")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path('exploratory_results')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df = load_processed_data()
    
    # Run analyses
    clusterer, clustering_results = run_clustering_analysis(df)
    reducer, reduction_results = run_dimensionality_reduction(df, clusterer)
    miner, transactions = run_association_rules(df)
    network, centrality_df = run_network_analysis(transactions)
    
    # Generate report
    generate_phase3_report(clusterer, reducer, miner, network)
    
    print("\n" + "="*80)
    print("PHASE 3 COMPLETE")
    print("="*80)
    print("\nAll exploratory analyses completed successfully!")
    print(f"Results saved to {output_dir}/")
    print("\n")


if __name__ == "__main__":
    main()
