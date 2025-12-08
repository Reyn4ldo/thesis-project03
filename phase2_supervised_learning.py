#!/usr/bin/env python3
"""
Phase 2 - Supervised Learning Experiments

Runs supervised learning experiments across multiple prediction tasks:
1. Binary ESBL classification
2. Multi-label R/S prediction
3. Species classification from antibiogram
4. MAR index regression

Uses six algorithms with nested cross-validation and MLflow tracking.
"""

import pandas as pd
import numpy as np
import mlflow
import warnings
warnings.filterwarnings('ignore')

from experiments import (
    ESBLClassifierExperiment,
    MultilabelPredictionExperiment,
    SpeciesClassifierExperiment,
    MARRegressionExperiment
)


def load_processed_data():
    """Load preprocessed train/test data."""
    print("="*80)
    print("LOADING PROCESSED DATA")
    print("="*80)
    
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    
    print(f"\nTrain set: {len(train_df)} samples, {len(train_df.columns)} features")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df


def run_esbl_classification(train_df, test_df):
    """Run ESBL classification experiment."""
    print("\n" + "="*80)
    print("TASK 1: BINARY ESBL CLASSIFICATION")
    print("="*80)
    
    experiment = ESBLClassifierExperiment(output_dir='models')
    
    # Prepare data
    X_train, y_train, features = experiment.prepare_data(train_df)
    X_test, y_test, _ = experiment.prepare_data(test_df)
    
    # Run all algorithms
    results = experiment.run_all_algorithms(
        X_train, y_train, X_test, y_test,
        use_nested_cv=True
    )
    
    # Save results
    experiment.save_results()
    
    # Print summary
    print("\n" + "-"*80)
    print("ESBL CLASSIFICATION RESULTS SUMMARY")
    print("-"*80)
    for algo, res in results.items():
        test_metrics = res['test_metrics']
        print(f"\n{algo}:")
        print(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        print(f"  F1: {test_metrics.get('f1', 0):.4f}")
        print(f"  AUROC: {test_metrics.get('auroc', 'N/A')}")
    
    return results


def run_species_classification(train_df, test_df):
    """Run species classification experiment."""
    print("\n" + "="*80)
    print("TASK 2: SPECIES CLASSIFICATION FROM ANTIBIOGRAM")
    print("="*80)
    
    experiment = SpeciesClassifierExperiment(output_dir='models')
    
    # Prepare data
    X_train, y_train, features = experiment.prepare_data(train_df)
    X_test, y_test, _ = experiment.prepare_data(test_df)
    
    # Run all algorithms
    results = experiment.run_all_algorithms(
        X_train, y_train, X_test, y_test,
        use_nested_cv=True
    )
    
    # Save results
    experiment.save_results()
    
    # Print summary
    print("\n" + "-"*80)
    print("SPECIES CLASSIFICATION RESULTS SUMMARY")
    print("-"*80)
    for algo, res in results.items():
        test_metrics = res['test_metrics']
        print(f"\n{algo}:")
        print(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (macro): {test_metrics.get('f1_macro', 0):.4f}")
    
    return results


def run_mar_regression(train_df, test_df):
    """Run MAR index regression experiment."""
    print("\n" + "="*80)
    print("TASK 3: MAR INDEX REGRESSION")
    print("="*80)
    
    experiment = MARRegressionExperiment(output_dir='models')
    
    # Prepare data
    X_train, y_train, features = experiment.prepare_data(train_df)
    X_test, y_test, _ = experiment.prepare_data(test_df)
    
    # Run all algorithms (5 for regression)
    results = experiment.run_all_algorithms(
        X_train, y_train, X_test, y_test,
        use_nested_cv=True
    )
    
    # Save results
    experiment.save_results()
    
    # Print summary
    print("\n" + "-"*80)
    print("MAR REGRESSION RESULTS SUMMARY")
    print("-"*80)
    for algo, res in results.items():
        test_metrics = res['test_metrics']
        print(f"\n{algo}:")
        print(f"  RMSE: {test_metrics.get('rmse', 0):.4f}")
        print(f"  MAE: {test_metrics.get('mae', 0):.4f}")
        print(f"  R²: {test_metrics.get('r2', 0):.4f}")
    
    return results


def generate_phase2_report(esbl_results, species_results, mar_results):
    """Generate comprehensive Phase 2 summary report."""
    print("\n" + "="*80)
    print("GENERATING PHASE 2 SUMMARY REPORT")
    print("="*80)
    
    report = []
    report.append("# Phase 2 - Supervised Learning Experiments Summary\n")
    report.append("="*80 + "\n\n")
    
    report.append("## Experiments Completed\n\n")
    report.append("1. ✅ Binary ESBL Classification (6 algorithms)\n")
    report.append("2. ✅ Species Classification from Antibiogram (6 algorithms)\n")
    report.append("3. ✅ MAR Index Regression (5 algorithms)\n\n")
    
    report.append("## Key Findings\n\n")
    
    # ESBL results
    report.append("### Task 1: ESBL Classification\n\n")
    report.append("Best performing algorithms:\n")
    best_esbl = sorted(esbl_results.items(), 
                      key=lambda x: x[1]['test_metrics'].get('auroc', 0), 
                      reverse=True)[:3]
    for algo, res in best_esbl:
        metrics = res['test_metrics']
        report.append(f"- **{algo}**: AUROC={metrics.get('auroc', 'N/A')}, F1={metrics.get('f1', 0):.4f}\n")
    report.append("\n")
    
    # Species results
    report.append("### Task 2: Species Classification\n\n")
    report.append("Best performing algorithms:\n")
    best_species = sorted(species_results.items(), 
                         key=lambda x: x[1]['test_metrics'].get('accuracy', 0), 
                         reverse=True)[:3]
    for algo, res in best_species:
        metrics = res['test_metrics']
        report.append(f"- **{algo}**: Accuracy={metrics.get('accuracy', 0):.4f}, F1={metrics.get('f1_macro', 0):.4f}\n")
    report.append("\n")
    
    # MAR results
    report.append("### Task 3: MAR Regression\n\n")
    report.append("Best performing algorithms:\n")
    best_mar = sorted(mar_results.items(), 
                     key=lambda x: x[1]['test_metrics'].get('r2', 0), 
                     reverse=True)[:3]
    for algo, res in best_mar:
        metrics = res['test_metrics']
        report.append(f"- **{algo}**: R²={metrics.get('r2', 0):.4f}, RMSE={metrics.get('rmse', 0):.4f}\n")
    report.append("\n")
    
    report.append("## Model Artifacts\n\n")
    report.append("All trained models saved to `models/` directory:\n")
    report.append("- ESBL classifiers (6 models)\n")
    report.append("- Species classifiers (6 models)\n")
    report.append("- MAR regressors (5 models)\n\n")
    
    report.append("## MLflow Tracking\n\n")
    report.append("Experiments logged to MLflow:\n")
    report.append("- View results: `mlflow ui`\n")
    report.append("- Compare models across metrics\n")
    report.append("- Track hyperparameters and artifacts\n\n")
    
    report.append("## Next Steps\n\n")
    report.append("- Review confusion matrices and calibration plots\n")
    report.append("- Generate SHAP explanations for interpretability\n")
    report.append("- Perform model selection using Pareto front analysis\n")
    report.append("- Deploy selected models for production use\n")
    
    report_text = "".join(report)
    
    with open('PHASE2_SUMMARY.md', 'w') as f:
        f.write(report_text)
    
    print("\n✅ Summary report saved to 'PHASE2_SUMMARY.md'")
    print(report_text)


def main():
    """Main execution function for Phase 2."""
    print("\n" + "="*80)
    print("PHASE 2 - SUPERVISED LEARNING EXPERIMENTS")
    print("="*80 + "\n")
    
    # Load data
    train_df, test_df = load_processed_data()
    
    # Run experiments
    esbl_results = run_esbl_classification(train_df, test_df)
    species_results = run_species_classification(train_df, test_df)
    mar_results = run_mar_regression(train_df, test_df)
    
    # Generate report
    generate_phase2_report(esbl_results, species_results, mar_results)
    
    print("\n" + "="*80)
    print("PHASE 2 COMPLETE")
    print("="*80)
    print("\nAll experiments completed successfully!")
    print("Review MLflow UI for detailed results: mlflow ui")
    print("\n")


if __name__ == "__main__":
    main()
