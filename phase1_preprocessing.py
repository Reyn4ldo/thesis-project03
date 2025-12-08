#!/usr/bin/env python3
"""
Phase 1 - Data Cleaning & Feature Engineering Pipeline

This script demonstrates the usage of the preprocessing pipeline for
antibiotic resistance data.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add preprocessing module to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import (
    MICSIRCleaner,
    DomainAwareImputer,
    ResistanceFeatureEngineer,
    StratifiedDataSplitter,
    create_preprocessing_pipeline
)
from preprocessing.pipeline import PreprocessingPipelineWrapper


def load_raw_data(filepath='raw - data.csv'):
    """Load the raw antibiotic resistance data."""
    print("="*80)
    print("LOADING RAW DATA")
    print("="*80)
    
    df = pd.read_csv(filepath)
    print(f"\nLoaded {len(df)} isolates with {len(df.columns)} columns")
    
    return df


def run_preprocessing_pipeline(df, save_outputs=True):
    """
    Run the complete preprocessing pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    save_outputs : bool
        Whether to save processed data and pipeline
        
    Returns
    -------
    tuple
        (processed_df, pipeline_wrapper)
    """
    print("\n" + "="*80)
    print("RUNNING PREPROCESSING PIPELINE")
    print("="*80)
    
    # Create pipeline with configuration
    config = {
        'mic_strategy': 'median',
        'sir_strategy': 'not_tested',
        'encoding_strategy': 'label',  # Use label encoding for efficiency
        'add_inconsistency_flags': True,
        'add_imputation_indicators': True,
        'validate_mar_index': True,
        'verbose': True
    }
    
    print("\nPipeline configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create and fit pipeline
    pipeline_wrapper = PreprocessingPipelineWrapper(config=config)
    
    print("\n" + "-"*80)
    print("Step 1: Cleaning MIC and S/I/R values")
    print("-"*80)
    
    processed_df = pipeline_wrapper.fit_transform(df)
    
    print("\n" + "-"*80)
    print("PREPROCESSING COMPLETE")
    print("-"*80)
    print(f"Processed dataset: {len(processed_df)} isolates, {len(processed_df.columns)} columns")
    
    # Print statistics
    print("\n" + "="*80)
    print("CLEANING STATISTICS")
    print("="*80)
    cleaning_stats = pipeline_wrapper.get_cleaning_stats()
    for antibiotic, stats in list(cleaning_stats.items())[:5]:  # Show first 5
        print(f"\n{antibiotic}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    print(f"\n... and {len(cleaning_stats) - 5} more antibiotics")
    
    print("\n" + "="*80)
    print("IMPUTATION STATISTICS")
    print("="*80)
    imputation_stats = pipeline_wrapper.get_imputation_stats()
    total_mic_imputed = sum(s['mic_imputed'] for s in imputation_stats.values())
    total_sir_imputed = sum(s['sir_imputed'] for s in imputation_stats.values())
    print(f"Total MIC values imputed: {total_mic_imputed}")
    print(f"Total S/I/R values imputed: {total_sir_imputed}")
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    feature_names = pipeline_wrapper.get_feature_names()
    print(f"Created {len(feature_names)} new features")
    
    # Categorize features
    binary_features = [f for f in feature_names if '_resistant' in f and 'count' not in f]
    antibiogram_features = [f for f in feature_names if 'antibiogram_' in f]
    aggregate_features = [f for f in feature_names if any(x in f for x in ['total_', 'ratio', 'count'])]
    metadata_features = [f for f in feature_names if '_encoded' in f or any(x in f for x in [
        'bacterial_species_', 'administrative_region_', 'sample_source_'
    ])]
    
    print(f"\nFeature breakdown:")
    print(f"  Binary resistance indicators: {len(binary_features)}")
    print(f"  Antibiogram features: {len(antibiogram_features)}")
    print(f"  Aggregate features: {len(aggregate_features)}")
    print(f"  Metadata features: {len(metadata_features)}")
    
    # Save outputs
    if save_outputs:
        print("\n" + "="*80)
        print("SAVING OUTPUTS")
        print("="*80)
        
        # Save processed data
        output_path = 'processed_data.csv'
        processed_df.to_csv(output_path, index=False)
        print(f"✓ Processed data saved to '{output_path}'")
        
        # Save pipeline
        pipeline_path = 'preprocessing_pipeline'
        pipeline_wrapper.save(pipeline_path)
        print(f"✓ Pipeline saved (can be reloaded for inference)")
    
    return processed_df, pipeline_wrapper


def create_data_splits(df):
    """
    Create train/validation/test splits.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed data
        
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    print("\n" + "="*80)
    print("CREATING DATA SPLITS")
    print("="*80)
    
    # Create splitter with stratification by species
    splitter = StratifiedDataSplitter(
        stratify_by='bacterial_species',
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        verbose=True
    )
    
    train_df, val_df, test_df = splitter.split(df)
    
    # Save splits
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    print("\n✓ Data splits saved:")
    print(f"  - train_data.csv ({len(train_df)} samples)")
    print(f"  - val_data.csv ({len(val_df)} samples)")
    print(f"  - test_data.csv ({len(test_df)} samples)")
    
    return train_df, val_df, test_df


def generate_summary_report(df, processed_df, train_df, val_df, test_df):
    """Generate a summary report of the preprocessing."""
    print("\n" + "="*80)
    print("PHASE 1 SUMMARY REPORT")
    print("="*80)
    
    report = []
    report.append("# Phase 1 - Data Cleaning & Feature Engineering Summary\n")
    report.append("=" * 80 + "\n")
    
    report.append("\n## Dataset Overview\n")
    report.append(f"Raw data: {len(df)} isolates, {len(df.columns)} columns\n")
    report.append(f"Processed data: {len(processed_df)} isolates, {len(processed_df.columns)} columns\n")
    report.append(f"New features created: {len(processed_df.columns) - len(df.columns)}\n")
    
    report.append("\n## Data Splits\n")
    report.append(f"Training set: {len(train_df)} isolates ({len(train_df)/len(df)*100:.1f}%)\n")
    report.append(f"Validation set: {len(val_df)} isolates ({len(val_df)/len(df)*100:.1f}%)\n")
    report.append(f"Test set: {len(test_df)} isolates ({len(test_df)/len(df)*100:.1f}%)\n")
    
    report.append("\n## Key Features Created\n")
    report.append("- Binary resistance indicators (0/1) per antibiotic\n")
    report.append("- Antibiogram fingerprints (resistance vectors)\n")
    report.append("- Aggregate resistance metrics (counts, ratios)\n")
    report.append("- WHO priority antibiotic resistance tracking\n")
    report.append("- Encoded metadata features (species, region, source)\n")
    report.append("- MAR index validation\n")
    
    report.append("\n## Data Quality Improvements\n")
    report.append("✓ MIC values normalized (Unicode operators standardized)\n")
    report.append("✓ S/I/R interpretations cleaned and standardized\n")
    report.append("✓ MIC/SIR inconsistencies detected and flagged\n")
    report.append("✓ Missing values imputed with domain-aware strategies\n")
    report.append("✓ Imputation indicators added for transparency\n")
    
    report.append("\n## Deliverables\n")
    report.append("1. preprocessing/ - Reusable preprocessing module\n")
    report.append("   - mic_sir_cleaner.py - MIC/SIR cleaning\n")
    report.append("   - imputer.py - Missing value imputation\n")
    report.append("   - feature_engineer.py - Feature engineering\n")
    report.append("   - data_splitter.py - Data splitting\n")
    report.append("   - pipeline.py - Complete pipeline\n")
    report.append("\n2. processed_data.csv - Fully processed dataset\n")
    report.append("3. train_data.csv, val_data.csv, test_data.csv - Data splits\n")
    report.append("4. preprocessing_pipeline.pkl/.json - Saved pipeline\n")
    report.append("5. tests/ - Data validation tests\n")
    
    report.append("\n## Next Steps (Phase 2)\n")
    report.append("- Exploratory analysis of resistance patterns\n")
    report.append("- Co-resistance network analysis\n")
    report.append("- Species-specific resistance profiling\n")
    report.append("- Statistical testing across regions and sources\n")
    
    report_text = "".join(report)
    
    with open('PHASE1_SUMMARY.md', 'w') as f:
        f.write(report_text)
    
    print("\n✓ Summary report saved to 'PHASE1_SUMMARY.md'")
    print("\n" + report_text)


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("PHASE 1 - DATA CLEANING & FEATURE ENGINEERING")
    print("="*80 + "\n")
    
    # Load raw data
    df = load_raw_data('raw - data.csv')
    
    # Run preprocessing pipeline
    processed_df, pipeline_wrapper = run_preprocessing_pipeline(df, save_outputs=True)
    
    # Create data splits
    train_df, val_df, test_df = create_data_splits(processed_df)
    
    # Generate summary report
    generate_summary_report(df, processed_df, train_df, val_df, test_df)
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print("\nAll deliverables have been generated successfully.")
    print("The preprocessing pipeline is ready for use in Phase 2.\n")


if __name__ == "__main__":
    main()
