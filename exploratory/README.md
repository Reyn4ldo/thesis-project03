# Exploratory Analysis Module

Unsupervised learning and pattern discovery for antibiotic resistance data.

## Overview

This module provides comprehensive exploratory analysis tools:
- **Clustering** - Identify antibiotypes (groups with similar resistance patterns)
- **Dimensionality Reduction** - Visualize high-dimensional resistance data
- **Association Rules** - Discover co-resistance patterns
- **Network Analysis** - Map co-resistance relationships

## Components

### 1. Clustering (`clustering.py`)

Identifies antibiotypes using multiple algorithms.

**Algorithms**:
- K-means clustering
- Hierarchical clustering (Ward linkage)
- DBSCAN (density-based)

**Features**:
- Optimal k selection (elbow method, silhouette analysis)
- Cluster quality metrics (silhouette score, Davies-Bouldin index)
- Cluster profiles with metadata distributions
- Dendrogram visualization

**Usage**:
```python
from exploratory import AntibiogramClusterer

clusterer = AntibiogramClusterer(n_clusters=5)
X, features = clusterer.prepare_data(df)

# Find optimal number of clusters
elbow_results, optimal_k = clusterer.find_optimal_k(X, k_range=range(2, 11))

# Fit all methods
results = clusterer.fit_all(X)

# Get cluster profiles
profiles = clusterer.get_cluster_profiles(df, results['kmeans']['labels'])
```

### 2. Dimensionality Reduction (`dimensionality_reduction.py`)

Reduces high-dimensional resistance data to 2D for visualization.

**Methods**:
- **PCA** - Linear dimensionality reduction, variance analysis
- **t-SNE** - Non-linear, preserves local structure
- **UMAP** - Non-linear, preserves topological structure

**Features**:
- Variance explained analysis (PCA)
- Interactive plots colored by metadata
- Multiple layouts and coloring options

**Usage**:
```python
from exploratory import DimensionalityReducer

reducer = DimensionalityReducer()
X = df[antibiogram_cols].values

# Fit all methods
results = reducer.fit_all(X)

# Plot UMAP colored by species
reducer.plot_embedding(
    method='umap',
    color_by='bacterial_species',
    df=df,
    save_path='umap_by_species.png'
)

# Plot PCA variance
reducer.plot_pca_variance(save_path='pca_variance.png')
```

### 3. Association Rule Mining (`association_rules.py`)

Discovers frequent co-resistance patterns.

**Algorithms**:
- Apriori algorithm
- FP-growth (faster, recommended)

**Metrics**:
- **Support** - Frequency of pattern
- **Confidence** - Conditional probability
- **Lift** - Strength of association (>1 means positive correlation)

**Features**:
- Frequent itemset mining
- Association rule generation
- Co-resistance frequency matrix
- Rule filtering by thresholds

**Usage**:
```python
from exploratory import AssociationRuleMiner

miner = AssociationRuleMiner(
    min_support=0.1,
    min_confidence=0.5,
    min_lift=1.5
)

# Prepare transactions
transactions = miner.prepare_transactions(df)

# Mine itemsets using FP-growth
frequent_itemsets = miner.mine_fpgrowth(transactions)

# Generate rules
rules = miner.generate_rules(metric='lift', min_threshold=1.5)

# Get top rules
top_rules = miner.get_top_rules(n=20, sort_by='lift')

# Print rules
miner.print_top_rules(n=10)
```

### 4. Network Analysis (`network_analysis.py`)

Creates and analyzes co-resistance networks.

**Network Construction**:
- Nodes = antibiotics
- Edges = co-resistance relationships
- Edge weights = co-resistance frequency/strength

**Methods**:
- Frequency-based (co-resistance rate)
- Odds ratio-based (adjusted for marginal frequencies)
- Correlation-based (phi coefficient)

**Analysis**:
- Centrality measures (degree, betweenness, closeness, eigenvector)
- Community detection (Louvain algorithm)
- Hub identification
- Network statistics (density, clustering, modularity)

**Usage**:
```python
from exploratory import CoResistanceNetwork

network = CoResistanceNetwork(min_edge_weight=0.15)

# Build network
graph = network.build_network(
    transactions, 
    method='frequency',
    compute_pvalues=False
)

# Compute centrality
centrality_df = network.compute_centrality()

# Detect communities
communities = network.detect_communities(resolution=1.0)

# Get hub antibiotics
hubs = network.get_hub_antibiotics(n=10, centrality_measure='degree')

# Plot network
network.plot_network(
    layout='spring',
    node_color_by='community',
    save_path='network.png'
)

# Get statistics
stats = network.get_network_statistics()
```

## Running Phase 3

### Quick Start

```bash
# Run complete exploratory analysis
python phase3_exploratory_analysis.py
```

This will:
1. Load preprocessed data
2. Run clustering analysis
3. Perform dimensionality reduction
4. Mine association rules
5. Build co-resistance network
6. Generate comprehensive report

### Output Files

- `PHASE3_SUMMARY.md` - Summary report
- `exploratory_results/` - Visualizations and artifacts

## Detailed Workflow

### 1. Clustering Workflow

```python
import pandas as pd
from exploratory import AntibiogramClusterer

# Load data
df = pd.read_csv('processed_data.csv')

# Initialize
clusterer = AntibiogramClusterer(n_clusters=5)
X, features = clusterer.prepare_data(df)

# Find optimal k
elbow_results, optimal_k = clusterer.find_optimal_k(X, range(2, 11))

# Plot elbow curve
clusterer.plot_elbow_curve(elbow_results, 'elbow_curve.png')

# Fit with optimal k
clusterer.n_clusters = optimal_k
results = clusterer.fit_all(X)

# Generate and analyze profiles
profiles = clusterer.get_cluster_profiles(df, results['kmeans']['labels'])
for cluster, profile in profiles.items():
    print(f"{cluster}: {profile['size']} isolates ({profile['prevalence']:.1f}%)")
```

### 2. Visualization Workflow

```python
from exploratory import DimensionalityReducer

reducer = DimensionalityReducer()
results = reducer.fit_all(X)

# Create multiple visualizations
for method in ['pca', 'tsne', 'umap']:
    for color_by in ['bacterial_species', 'administrative_region', 'esbl']:
        reducer.plot_embedding(
            method=method,
            color_by=color_by,
            df=df,
            save_path=f'{method}_{color_by}.png'
        )
```

### 3. Association Rules Workflow

```python
from exploratory import AssociationRuleMiner

miner = AssociationRuleMiner(min_support=0.1, min_confidence=0.5)
transactions = miner.prepare_transactions(df)

# Mine patterns
itemsets = miner.mine_fpgrowth(transactions)
rules = miner.generate_rules(metric='lift', min_threshold=1.5)

# Analyze specific antibiotic
ampicillin_rules = miner.get_rules_for_antibiotic('ampicillin')

# Get co-resistance matrix
co_resistance_matrix = miner.get_co_resistance_matrix(transactions)
```

### 4. Network Analysis Workflow

```python
from exploratory import CoResistanceNetwork

network = CoResistanceNetwork(min_edge_weight=0.15)
graph = network.build_network(transactions, method='frequency')

# Analyze structure
centrality = network.compute_centrality()
communities = network.detect_communities()
hubs = network.get_hub_antibiotics(n=10)

# Visualize
network.plot_network(
    layout='spring',
    node_color_by='community',
    node_size_by='degree',
    save_path='network.png'
)

# Export for Gephi
network.export_network('network.gexf', format='gexf')
```

## Validation

### Cluster Stability

```python
# Bootstrap stability test
from sklearn.utils import resample

stability_scores = []
for i in range(100):
    X_boot = resample(X, random_state=i)
    clusterer_boot = AntibiogramClusterer(n_clusters=optimal_k)
    results_boot = clusterer_boot.fit_kmeans(X_boot)
    stability_scores.append(results_boot['silhouette_score'])

print(f"Mean silhouette: {np.mean(stability_scores):.3f} ± {np.std(stability_scores):.3f}")
```

### Network Robustness

```python
# Test network stability with different thresholds
thresholds = [0.1, 0.15, 0.2, 0.25]
modularity_scores = []

for threshold in thresholds:
    net = CoResistanceNetwork(min_edge_weight=threshold)
    net.build_network(transactions)
    net.detect_communities()
    stats = net.get_network_statistics()
    modularity_scores.append(stats['modularity'])
```

## Interpretation Guidelines

### Clustering Results

- **Silhouette Score**: [-1, 1], higher is better
  - >0.7: Strong structure
  - 0.5-0.7: Reasonable structure
  - <0.5: Weak/overlapping clusters

- **Davies-Bouldin Index**: Lower is better
  - <1.0: Good separation

### Association Rules

- **Lift > 1**: Positive association (co-occur more than expected)
- **Lift = 1**: Independent
- **Lift < 1**: Negative association

Example rule interpretation:
```
{ampicillin} -> {amoxicillin/clavulanic_acid} 
[support=0.35, confidence=0.82, lift=2.1]
```
- 35% of isolates resistant to both
- 82% of ampicillin-resistant isolates also resistant to amoxicillin/clavulanic_acid
- 2.1x more likely than random chance

### Network Metrics

- **Degree Centrality**: Number of connections (hub antibiotics)
- **Betweenness Centrality**: Bridges between communities
- **Modularity**: Quality of community structure (>0.3 is good)

## Troubleshooting

### Issue: DBSCAN finds no clusters
**Solution**: Adjust `eps` parameter based on data scale

```python
# Estimate good eps from nearest neighbor distances
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=10).fit(X)
distances, _ = nbrs.kneighbors(X)
eps = np.percentile(distances[:, -1], 90)
```

### Issue: t-SNE/UMAP takes too long
**Solution**: Use PCA preprocessing

```python
# Reduce to 50 dimensions first
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
# Then apply t-SNE/UMAP on X_reduced
```

### Issue: Too few/many association rules
**Solution**: Adjust thresholds

```python
# More rules: lower thresholds
miner = AssociationRuleMiner(min_support=0.05, min_confidence=0.3, min_lift=1.0)

# Fewer, stronger rules: higher thresholds
miner = AssociationRuleMiner(min_support=0.2, min_confidence=0.7, min_lift=2.0)
```

## Dependencies

```
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
umap-learn >= 0.5
mlxtend >= 0.22
networkx >= 3.0
python-louvain >= 0.16
matplotlib >= 3.5
seaborn >= 0.12
```

## References

- Clustering: [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- t-SNE: [van der Maaten & Hinton, 2008](http://jmlr.org/papers/v9/vandermaaten08a.html)
- UMAP: [McInnes et al., 2018](https://arxiv.org/abs/1802.03426)
- Association Rules: [Agrawal & Srikant, 1994](http://www.vldb.org/conf/1994/P487.PDF)
- Louvain: [Blondel et al., 2008](https://arxiv.org/abs/0803.0476)
