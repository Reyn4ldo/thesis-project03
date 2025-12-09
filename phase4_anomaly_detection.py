#!/usr/bin/env python3
"""
Phase 4 - Anomaly & Outlier Detection

Identifies rare, extreme, or inconsistent isolates using multiple methods:
- Unsupervised outlier detection (Isolation Forest, LOF, DBSCAN, Mahalanobis)
- Rule-based consistency checks (MIC vs S/I/R)
- Composite anomaly scoring and triage
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from anomaly import OutlierDetector, ConsistencyChecker, AnomalyScorer
from pathlib import Path


def load_processed_data():
    """Load preprocessed data."""
    print("="*80)
    print("LOADING PROCESSED DATA")
    print("="*80)
    
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    
    # Combine for anomaly detection
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"\nTotal samples: {len(full_df)}")
    print(f"Total features: {len(full_df.columns)}")
    
    return full_df


def run_outlier_detection(df):
    """Run unsupervised outlier detection."""
    print("\n" + "="*80)
    print("UNSUPERVISED OUTLIER DETECTION")
    print("="*80)
    
    # Initialize detector
    detector = OutlierDetector(contamination=0.05, random_state=42)
    
    # Prepare data
    X, feature_names = detector.prepare_data(df)
    
    # Fit all methods
    results = detector.fit_all(X)
    
    # Get consensus outliers
    consensus_indices, outlier_counts = detector.get_consensus_outliers(min_methods=2)
    
    # Get top outliers from each method
    print("\n" + "-"*80)
    print("TOP 10 OUTLIERS BY METHOD")
    print("-"*80)
    
    for method in ['isolation_forest', 'lof', 'mahalanobis']:
        top_indices, top_scores = detector.get_top_outliers(n=10, method=method)
        print(f"\n{method.upper()}:")
        for i, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
            print(f"  {i}. Sample {idx}: score={score:.3f}")
    
    return detector, results, consensus_indices, outlier_counts


def run_consistency_checks(df):
    """Run rule-based consistency checks."""
    print("\n" + "="*80)
    print("CONSISTENCY CHECKS")
    print("="*80)
    
    # Initialize checker
    checker = ConsistencyChecker()
    
    # Run all checks
    consistency_results = checker.check_all(df)
    
    # Print report
    report = checker.generate_report()
    print("\n" + report)
    
    return checker, consistency_results


def run_anomaly_scoring(df, outlier_results, consistency_results):
    """Compute composite anomaly scores and triage."""
    print("\n" + "="*80)
    print("ANOMALY SCORING & TRIAGE")
    print("="*80)
    
    # Initialize scorer
    scorer = AnomalyScorer()
    
    # Compute composite scores
    scores = scorer.compute_composite_scores(outlier_results, consistency_results, df)
    
    # Assign triage labels
    labels = scorer.assign_triage_labels(scores)
    
    # Generate detailed report
    report_df = scorer.get_anomaly_report(df, scores, labels, top_n=50)
    
    # Print top anomalies
    print("\n" + "-"*80)
    print("TOP 20 MOST ANOMALOUS ISOLATES")
    print("-"*80)
    
    top_20 = report_df.head(20)
    for idx, row in top_20.iterrows():
        print(f"Sample {idx}:")
        print(f"  Anomaly Score: {row['anomaly_score']:.3f}")
        print(f"  Triage: {row['triage_label']}")
        if 'bacterial_species' in row:
            print(f"  Species: {row['bacterial_species']}")
        if 'mar_index' in row:
            print(f"  MAR Index: {row['mar_index']:.3f}")
        print()
    
    # Generate summary
    summary_report = scorer.generate_summary_report()
    print("\n" + summary_report)
    
    return scorer, scores, labels, report_df


def save_results(detector, checker, scorer, report_df):
    """Save all results to files."""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create output directory
    output_dir = Path('anomaly_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed anomaly report
    report_path = output_dir / 'anomaly_report.csv'
    report_df.to_csv(report_path)
    print(f"  Saved anomaly report: {report_path}")
    
    # Save outlier detection summary
    outlier_summary = detector.get_summary()
    summary_df = pd.DataFrame(outlier_summary).T
    summary_path = output_dir / 'outlier_detection_summary.csv'
    summary_df.to_csv(summary_path)
    print(f"  Saved outlier summary: {summary_path}")
    
    # Save consistency check results
    if len(checker.inconsistencies) > 0:
        inconsist_path = output_dir / 'mic_sir_inconsistencies.csv'
        checker.inconsistencies.to_csv(inconsist_path, index=False)
        print(f"  Saved MIC/SIR inconsistencies: {inconsist_path}")
    
    if len(checker.suspicious_patterns) > 0:
        patterns_df = pd.DataFrame(checker.suspicious_patterns)
        patterns_path = output_dir / 'suspicious_patterns.csv'
        patterns_df.to_csv(patterns_path, index=False)
        print(f"  Saved suspicious patterns: {patterns_path}")
    
    # Save triage summary
    triage_summary = scorer.get_triage_summary()
    triage_df = pd.DataFrame(triage_summary).T
    triage_path = output_dir / 'triage_summary.csv'
    triage_df.to_csv(triage_path)
    print(f"  Saved triage summary: {triage_path}")
    
    print(f"\nAll results saved to {output_dir}/")


def generate_phase4_summary(detector, checker, scorer, consensus_indices):
    """Generate comprehensive Phase 4 summary report."""
    print("\n" + "="*80)
    print("GENERATING PHASE 4 SUMMARY")
    print("="*80)
    
    report = []
    report.append("# Phase 4 - Anomaly & Outlier Detection Summary\n")
    report.append("="*80 + "\n\n")
    
    report.append("## Analyses Completed\n\n")
    report.append("1. ✅ Unsupervised Outlier Detection\n")
    report.append("   - Isolation Forest\n")
    report.append("   - Local Outlier Factor (LOF)\n")
    report.append("   - DBSCAN outlier detection\n")
    report.append("   - Mahalanobis distance\n\n")
    
    report.append("2. ✅ Rule-Based Consistency Checks\n")
    report.append("   - MIC vs S/I/R validation\n")
    report.append("   - Impossible resistance patterns\n")
    report.append("   - MAR index consistency\n\n")
    
    report.append("3. ✅ Composite Anomaly Scoring\n")
    report.append("   - Multi-method score aggregation\n")
    report.append("   - Triage label assignment\n")
    report.append("   - Automated review pipeline\n\n")
    
    # Outlier detection results
    report.append("## Outlier Detection Results\n\n")
    outlier_summary = detector.get_summary()
    for method, stats in outlier_summary.items():
        report.append(f"### {method.upper()}\n")
        report.append(f"- Outliers detected: {stats['n_outliers']} ({stats['outlier_rate']:.2%})\n")
        report.append(f"- Score range: [{stats['score_min']:.3f}, {stats['score_max']:.3f}]\n")
        report.append(f"- Mean score: {stats['score_mean']:.3f} ± {stats['score_std']:.3f}\n\n")
    
    report.append(f"**Consensus outliers** (≥2 methods): {len(consensus_indices)}\n\n")
    
    # Consistency check results
    report.append("## Consistency Check Results\n\n")
    report.append(f"- MIC/SIR inconsistencies: {len(checker.inconsistencies)}\n")
    report.append(f"- Suspicious patterns: {len(checker.suspicious_patterns)}\n")
    report.append(f"- Total flagged isolates: {len(checker.get_flagged_isolates())}\n\n")
    
    # Triage summary
    report.append("## Triage Summary\n\n")
    if scorer.triage_labels is not None:
        triage_summary = scorer.get_triage_summary()
        for label in ['quarantine', 'review', 'monitor', 'normal']:
            if label in triage_summary:
                stats = triage_summary[label]
                report.append(f"### {label.capitalize()}\n")
                report.append(f"- Count: {stats['count']} ({stats['percentage']:.1f}%)\n")
                report.append(f"- Mean score: {stats['mean_score']:.3f}\n")
                report.append(f"- Score range: [{stats['score_range'][0]:.3f}, {stats['score_range'][1]:.3f}]\n\n")
    
    report.append("## Deliverables\n\n")
    report.append("- ✅ Anomaly score per isolate (0-1 scale)\n")
    report.append("- ✅ Triage labels (quarantine/review/monitor/normal)\n")
    report.append("- ✅ Outlier detection results from 4 methods\n")
    report.append("- ✅ Consistency check reports\n")
    report.append("- ✅ Detailed anomaly report with metadata\n\n")
    
    report.append("## Next Steps\n\n")
    report.append("- Human expert review of high-priority anomalies\n")
    report.append("- Investigate quarantine-level isolates\n")
    report.append("- Refine triage thresholds based on domain knowledge\n")
    report.append("- Integrate anomaly detection into data pipeline\n")
    
    report_text = "".join(report)
    
    with open('PHASE4_SUMMARY.md', 'w') as f:
        f.write(report_text)
    
    print("\n✅ Summary report saved to 'PHASE4_SUMMARY.md'")


def main():
    """Main execution function for Phase 4."""
    print("\n" + "="*80)
    print("PHASE 4 - ANOMALY & OUTLIER DETECTION")
    print("="*80 + "\n")
    
    # Load data
    df = load_processed_data()
    
    # Run outlier detection
    detector, outlier_results, consensus_indices, outlier_counts = run_outlier_detection(df)
    
    # Run consistency checks
    checker, consistency_results = run_consistency_checks(df)
    
    # Compute anomaly scores and triage
    scorer, scores, labels, report_df = run_anomaly_scoring(df, outlier_results, consistency_results)
    
    # Save results
    save_results(detector, checker, scorer, report_df)
    
    # Generate summary
    generate_phase4_summary(detector, checker, scorer, consensus_indices)
    
    print("\n" + "="*80)
    print("PHASE 4 COMPLETE")
    print("="*80)
    print("\nAll anomaly detection analyses completed successfully!")
    print("Review results in anomaly_results/ directory")
    print("\n")


if __name__ == "__main__":
    main()
