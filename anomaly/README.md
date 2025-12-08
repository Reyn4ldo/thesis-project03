# Anomaly Detection Module

Identifies rare, extreme, or inconsistent isolates in antibiotic resistance data.

## Overview

This module provides comprehensive anomaly detection using:
- **Unsupervised Methods** - Isolation Forest, LOF, DBSCAN, Mahalanobis distance
- **Rule-Based Checks** - MIC vs S/I/R validation, impossible patterns
- **Composite Scoring** - Multi-method aggregation with triage
- **Automated Pipeline** - Scoring and triage for quality control

## Components

### 1. Outlier Detection (`outlier_detectors.py`)

Unsupervised anomaly detection using multiple algorithms.

**Algorithms**:
- **Isolation Forest** - Tree-based, efficient for high-dimensional data
- **Local Outlier Factor (LOF)** - Density-based, detects local outliers
- **DBSCAN** - Identifies noise points as outliers
- **Mahalanobis Distance** - Multivariate distance from distribution center

**Features**:
- Automatic data preparation and scaling
- Consensus outlier detection (agreement across methods)
- Configurable contamination rate
- Score normalization

**Usage**:
```python
from anomaly import OutlierDetector

detector = OutlierDetector(contamination=0.05)
X, features = detector.prepare_data(df)

# Fit all methods
results = detector.fit_all(X)

# Get consensus outliers (≥2 methods agree)
consensus_indices, counts = detector.get_consensus_outliers(min_methods=2)

# Get top outliers by specific method
top_indices, scores = detector.get_top_outliers(n=20, method='isolation_forest')
```

### 2. Consistency Checking (`consistency_checker.py`)

Rule-based validation for data quality and biological plausibility.

**Checks Performed**:
- MIC vs S/I/R interpretation consistency
- Impossible resistance patterns (all resistant, all susceptible)
- MAR index consistency with resistance profile

**Features**:
- Breakpoint violation detection
- Severity scoring (high/medium/low)
- Detailed inconsistency reports

**Usage**:
```python
from anomaly import ConsistencyChecker

checker = ConsistencyChecker()

# Run all checks
results = checker.check_all(df)

# Get inconsistencies
mic_sir_issues = results['mic_sir_inconsistencies']
impossible_patterns = results['impossible_patterns']
mar_issues = results['mar_inconsistencies']

# Get flagged isolates
flagged_ids = checker.get_flagged_isolates()

# Generate report
report = checker.generate_report()
print(report)
```

### 3. Anomaly Scoring (`anomaly_scorer.py`)

Combines multiple detection methods into unified scores with triage.

**Scoring System**:
- Weighted combination of all detection methods
- Normalized to [0, 1] range
- Higher scores indicate more anomalous samples

**Triage Levels**:
- **Quarantine** (≥0.8) - Automatic quarantine, immediate investigation
- **Review** (≥0.5) - Human expert review required
- **Monitor** (≥0.3) - Flag for monitoring
- **Normal** (<0.3) - No action required

**Default Weights**:
```python
{
    'isolation_forest': 0.3,
    'lof': 0.3,
    'mahalanobis': 0.2,
    'dbscan': 0.1,
    'consistency': 0.1
}
```

**Usage**:
```python
from anomaly import AnomalyScorer

scorer = AnomalyScorer()

# Compute composite scores
scores = scorer.compute_composite_scores(
    outlier_results,
    consistency_results,
    df
)

# Assign triage labels
labels = scorer.assign_triage_labels(scores)

# Get detailed report
report_df = scorer.get_anomaly_report(df, scores, labels, top_n=50)

# Get triage summary
summary = scorer.get_triage_summary()

# Save results
scorer.save_results(df, 'anomaly_results.csv')
```

## Running Phase 4

### Quick Start

```bash
# Run complete anomaly detection
python phase4_anomaly_detection.py
```

This will:
1. Load preprocessed data
2. Run 4 unsupervised outlier detection methods
3. Perform rule-based consistency checks
4. Compute composite anomaly scores
5. Assign triage labels
6. Generate comprehensive reports
7. Save all results to `anomaly_results/` directory

### Output Files

- `anomaly_report.csv` - Detailed report with scores and labels
- `outlier_detection_summary.csv` - Summary of each method
- `mic_sir_inconsistencies.csv` - MIC/SIR validation issues
- `suspicious_patterns.csv` - Impossible resistance patterns
- `triage_summary.csv` - Statistics per triage category
- `PHASE4_SUMMARY.md` - Comprehensive summary report

## Detailed Workflow

### 1. Outlier Detection Workflow

```python
import pandas as pd
from anomaly import OutlierDetector

# Load data
df = pd.read_csv('processed_data.csv')

# Initialize detector
detector = OutlierDetector(contamination=0.05, random_state=42)

# Prepare data (selects relevant features)
X, features = detector.prepare_data(df)

# Fit individual methods
if_results = detector.fit_isolation_forest(X)
lof_results = detector.fit_lof(X, n_neighbors=20)
dbscan_results = detector.fit_dbscan_outliers(X, eps=0.5, min_samples=5)
maha_results = detector.compute_mahalanobis_distance(X)

# Or fit all at once
results = detector.fit_all(X)

# Get consensus outliers
consensus_indices, counts = detector.get_consensus_outliers(min_methods=2)

print(f"Consensus outliers: {len(consensus_indices)}")

# Examine specific outliers
for idx in consensus_indices[:5]:
    print(f"Sample {idx}:")
    print(f"  Species: {df.loc[idx, 'bacterial_species']}")
    print(f"  MAR index: {df.loc[idx, 'mar_index']}")
```

### 2. Consistency Check Workflow

```python
from anomaly import ConsistencyChecker

checker = ConsistencyChecker()

# Individual checks
mic_sir = checker.check_mic_sir_consistency(df)
patterns = checker.check_impossible_patterns(df)
mar_check = checker.check_mar_consistency(df)

# All checks
results = checker.check_all(df)

# Analyze MIC/SIR issues
if len(mic_sir) > 0:
    print("\nMIC/SIR Inconsistencies:")
    for antibiotic in mic_sir['antibiotic'].unique():
        ab_issues = mic_sir[mic_sir['antibiotic'] == antibiotic]
        print(f"  {antibiotic}: {len(ab_issues)} issues")

# Analyze patterns
if len(patterns) > 0:
    print("\nSuspicious Patterns:")
    for pattern in patterns[:5]:
        print(f"  Sample {pattern['isolate_id']}: {pattern['description']}")
```

### 3. Composite Scoring Workflow

```python
from anomaly import AnomalyScorer

# Custom weights (optional)
custom_weights = {
    'isolation_forest': 0.4,  # Emphasize Isolation Forest
    'lof': 0.2,
    'mahalanobis': 0.2,
    'dbscan': 0.1,
    'consistency': 0.1
}

scorer = AnomalyScorer(weights=custom_weights)

# Compute scores
scores = scorer.compute_composite_scores(outlier_results, consistency_results, df)

# Custom triage thresholds
custom_thresholds = {
    'quarantine': 0.9,  # Stricter quarantine threshold
    'review': 0.6,
    'monitor': 0.3,
    'normal': 0.0
}

labels = scorer.assign_triage_labels(scores, thresholds=custom_thresholds)

# Get top anomalies
top_20 = scorer.get_top_anomalies(20)

# Generate report
report_df = scorer.get_anomaly_report(df, scores, labels)

# Filter by triage level
quarantine = report_df[report_df['triage_label'] == 'quarantine']
print(f"\n{len(quarantine)} isolates require quarantine")
```

## Interpretation Guidelines

### Outlier Detection Scores

**Isolation Forest**:
- Lower scores = more anomalous (score is negated for consistency)
- Good for detecting global outliers
- Fast and scalable

**LOF (Local Outlier Factor)**:
- LOF > 1: outlier (lower density than neighbors)
- LOF ≈ 1: normal point
- Good for detecting local anomalies

**DBSCAN**:
- Binary: noise points (label=-1) are outliers
- Clusters represent normal behavior
- Good for finding arbitrary-shaped clusters

**Mahalanobis Distance**:
- Measures distance from distribution center
- Accounts for correlations between features
- Higher distance = more anomalous

### Consistency Checks

**MIC vs S/I/R**:
- High MIC + Sensitive = Likely error
- Low MIC + Resistant = Possible error or mechanism
- Requires breakpoint knowledge for precise validation

**Impossible Patterns**:
- All resistant: Rare, may indicate pan-drug resistant strain or data error
- All susceptible: Unusual in surveillance, may indicate control strain
- Severity helps prioritize investigation

### Composite Scores

**Score Ranges**:
- 0.0-0.3: Normal range
- 0.3-0.5: Monitor (slight anomaly)
- 0.5-0.8: Review (moderate anomaly)
- 0.8-1.0: Quarantine (severe anomaly)

**Triage Actions**:
- **Quarantine**: Remove from analysis, investigate immediately
- **Review**: Flag for expert review before use
- **Monitor**: Include in analysis but track closely
- **Normal**: Use without restrictions

## Validation & Tuning

### Contamination Rate

The contamination parameter controls expected outlier proportion:

```python
# Conservative (expect fewer outliers)
detector = OutlierDetector(contamination=0.02)

# Moderate (default)
detector = OutlierDetector(contamination=0.05)

# Liberal (expect more outliers)
detector = OutlierDetector(contamination=0.10)
```

### Parameter Tuning

**LOF n_neighbors**:
```python
# Small neighborhood (local outliers)
detector.fit_lof(X, n_neighbors=10)

# Large neighborhood (global outliers)
detector.fit_lof(X, n_neighbors=50)
```

**DBSCAN eps and min_samples**:
```python
# Tighter clusters (more outliers)
detector.fit_dbscan_outliers(X, eps=0.3, min_samples=10)

# Looser clusters (fewer outliers)
detector.fit_dbscan_outliers(X, eps=0.7, min_samples=3)
```

### Validation

```python
# Check if outliers make biological sense
outlier_indices = results['isolation_forest']['outlier_indices']

for idx in outlier_indices[:10]:
    print(f"\nSample {idx}:")
    print(f"  Species: {df.loc[idx, 'bacterial_species']}")
    print(f"  Region: {df.loc[idx, 'administrative_region']}")
    print(f"  MAR: {df.loc[idx, 'mar_index']}")
    
    # Check resistance profile
    resistance_cols = [c for c in df.columns if '_resistant' in c]
    n_resistant = df.loc[idx, resistance_cols].sum()
    print(f"  Resistant to: {n_resistant}/{len(resistance_cols)} antibiotics")
```

## Use Cases

### 1. Quality Control Pipeline

```python
# Run anomaly detection on new batch
detector = OutlierDetector(contamination=0.05)
X, _ = detector.prepare_data(new_batch)
results = detector.fit_all(X)

checker = ConsistencyChecker()
consistency = checker.check_all(new_batch)

scorer = AnomalyScorer()
scores = scorer.compute_composite_scores(results, consistency, new_batch)
labels = scorer.assign_triage_labels(scores)

# Quarantine high-risk samples
quarantine_mask = labels == 'quarantine'
clean_data = new_batch[~quarantine_mask]
flagged_data = new_batch[quarantine_mask]

print(f"Removed {quarantine_mask.sum()} anomalous samples")
```

### 2. Surveillance System Monitoring

```python
# Weekly anomaly check
weekly_data = load_weekly_data()

scorer = AnomalyScorer()
# ... run full detection ...

# Alert if anomaly rate increases
current_rate = (labels != 'normal').sum() / len(labels)
if current_rate > historical_threshold:
    send_alert(f"Anomaly rate elevated: {current_rate:.2%}")
```

### 3. Research Data Cleaning

```python
# Identify and investigate unusual isolates
detector = OutlierDetector(contamination=0.10)  # Liberal threshold
# ... run detection ...

# Get consensus outliers for manual review
consensus, _ = detector.get_consensus_outliers(min_methods=3)  # Stricter

# Investigate each
for idx in consensus:
    isolate = df.loc[idx]
    # Perform detailed manual review
    # Document findings
    # Decide: keep, remove, or correct
```

## Troubleshooting

### Issue: Too many/few outliers detected

**Solution**: Adjust contamination parameter

```python
# Increase if too few
detector = OutlierDetector(contamination=0.10)

# Decrease if too many
detector = OutlierDetector(contamination=0.02)
```

### Issue: Outliers don't make biological sense

**Solution**: Check feature selection or use domain knowledge

```python
# Use only resistance profile features
resistance_cols = [c for c in df.columns if '_resistant' in c]
X = df[resistance_cols].values
X_scaled = StandardScaler().fit_transform(X)

detector = OutlierDetector()
results = detector.fit_isolation_forest(X_scaled)
```

### Issue: All methods disagree

**Solution**: Methods detect different types of anomalies

```python
# Examine what each method finds
for method in ['isolation_forest', 'lof', 'mahalanobis']:
    outliers = results[method]['outlier_indices']
    print(f"\n{method}: {len(outliers)} outliers")
    
    # Check characteristics
    for idx in outliers[:3]:
        print(f"  Sample {idx}: MAR={df.loc[idx, 'mar_index']:.3f}")
```

### Issue: Consistency checks find too many issues

**Solution**: Rules may be too strict or data needs cleaning

```python
# Review specific inconsistencies
issues = checker.inconsistencies
print(issues[['antibiotic', 'reason']].value_counts())

# Adjust MIC parsing if needed
# or update breakpoint thresholds in consistency_checker.py
```

## Dependencies

```
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
```

## Performance Considerations

- **Isolation Forest**: O(n log n), fast even for large datasets
- **LOF**: O(n²), can be slow for >10k samples
- **DBSCAN**: O(n log n) with spatial indexing
- **Mahalanobis**: O(n × p²), where p is number of features

For large datasets (>10k samples):
1. Use Isolation Forest as primary method
2. Sample data for LOF validation
3. Consider dimensionality reduction before Mahalanobis

## References

- Isolation Forest: [Liu et al., 2008](https://doi.org/10.1109/ICDM.2008.17)
- LOF: [Breunig et al., 2000](https://doi.org/10.1145/342009.335388)
- DBSCAN: [Ester et al., 1996](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
- Mahalanobis Distance: [Mahalanobis, 1936](https://en.wikipedia.org/wiki/Mahalanobis_distance)
