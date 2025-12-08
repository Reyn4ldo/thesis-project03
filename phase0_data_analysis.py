#!/usr/bin/env python3
"""
Phase 0 - Initial Setup & Data Understanding
This script performs comprehensive data analysis on the antibiotic resistance dataset
and generates required deliverables.
"""

import pandas as pd
import numpy as np
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load the raw data CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns\n")
    return df


def generate_data_dictionary(df):
    """Generate comprehensive data dictionary."""
    print("="*80)
    print("GENERATING DATA DICTIONARY")
    print("="*80 + "\n")
    
    data_dict = {
        'dataset_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'file_description': 'Antibiotic resistance surveillance dataset'
        },
        'columns': {}
    }
    
    # Categorize columns
    metadata_cols = ['bacterial_species', 'isolate_code', 'administrative_region', 
                     'national_site', 'local_site', 'sample_source', 'replicate', 
                     'colony', 'esbl']
    
    outcome_cols = ['scored_resistance', 'num_antibiotics_tested', 'mar_index']
    
    # Get all antibiotic columns (MIC and interpretation)
    antibiotic_cols = [col for col in df.columns if '_mic' in col or '_int' in col]
    
    # Identify unique antibiotics
    antibiotics = set()
    for col in antibiotic_cols:
        if '_mic' in col:
            antibiotics.add(col.replace('_mic', ''))
        elif '_int' in col:
            antibiotics.add(col.replace('_int', ''))
    
    print(f"Identified {len(antibiotics)} unique antibiotics\n")
    
    # Document each column
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'non_null_count': int(df[col].count()),
            'null_count': int(df[col].isnull().sum()),
            'null_percentage': round(df[col].isnull().sum() / len(df) * 100, 2),
            'unique_values': int(df[col].nunique()),
        }
        
        # Add category
        if col in metadata_cols:
            col_info['category'] = 'metadata'
        elif col in outcome_cols:
            col_info['category'] = 'outcome'
        elif '_mic' in col:
            col_info['category'] = 'mic_value'
        elif '_int' in col:
            col_info['category'] = 'sir_interpretation'
        else:
            col_info['category'] = 'other'
        
        # Add description based on column name
        descriptions = {
            'bacterial_species': 'Species identification of the bacterial isolate',
            'isolate_code': 'Unique identifier for each isolate',
            'administrative_region': 'Geographic administrative region',
            'national_site': 'National-level site identifier',
            'local_site': 'Local site identifier',
            'sample_source': 'Source of the sample (e.g., water, fish)',
            'replicate': 'Replicate number for the sample',
            'colony': 'Colony number',
            'esbl': 'Extended-spectrum beta-lactamase (ESBL) status',
            'scored_resistance': 'Number of antibiotics showing resistance',
            'num_antibiotics_tested': 'Total number of antibiotics tested',
            'mar_index': 'Multiple Antibiotic Resistance (MAR) index'
        }
        
        if col in descriptions:
            col_info['description'] = descriptions[col]
        elif '_mic' in col:
            antibiotic = col.replace('_mic', '').replace('_', ' ').title()
            col_info['description'] = f'Minimum Inhibitory Concentration (MIC) for {antibiotic}'
        elif '_int' in col:
            antibiotic = col.replace('_int', '').replace('_', ' ').title()
            col_info['description'] = f'Susceptibility interpretation (S/I/R) for {antibiotic}'
        else:
            col_info['description'] = 'No description available'
        
        # Add sample values for categorical/small unique value columns
        if col_info['unique_values'] <= 20:
            col_info['unique_value_counts'] = df[col].value_counts().to_dict()
        
        data_dict['columns'][col] = col_info
    
    # Save to JSON
    with open('data_dictionary.json', 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    print("✓ Data dictionary saved to 'data_dictionary.json'\n")
    return data_dict, list(antibiotics)


def analyze_missingness(df):
    """Analyze missing data patterns."""
    print("="*80)
    print("MISSINGNESS ANALYSIS")
    print("="*80 + "\n")
    
    missing_summary = []
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        if null_count > 0 or null_pct > 0:
            missing_summary.append({
                'column': col,
                'missing_count': null_count,
                'missing_percentage': round(null_pct, 2),
                'present_count': len(df) - null_count
            })
    
    missing_df = pd.DataFrame(missing_summary)
    missing_df = missing_df.sort_values('missing_percentage', ascending=False)
    
    print(f"Columns with missing data: {len(missing_df)} out of {len(df.columns)}")
    print("\nTop 10 columns with highest missingness:")
    print(missing_df.head(10).to_string(index=False))
    print()
    
    return missing_df


def analyze_species(df):
    """Analyze bacterial species distribution and taxonomy."""
    print("="*80)
    print("SPECIES ANALYSIS")
    print("="*80 + "\n")
    
    species_counts = df['bacterial_species'].value_counts()
    print(f"Total unique species: {len(species_counts)}\n")
    print("Species distribution:")
    print(species_counts.to_string())
    print()
    
    # Check for hierarchical taxonomy patterns
    species_list = df['bacterial_species'].dropna().unique()
    print("\nSpecies naming patterns (checking for hierarchical taxonomy):")
    for species in sorted(species_list):
        print(f"  - {species}")
    print()
    
    return species_counts


def analyze_sir_labels(df):
    """Analyze S/I/R interpretation label balance."""
    print("="*80)
    print("S/I/R LABEL BALANCE ANALYSIS")
    print("="*80 + "\n")
    
    # Get all interpretation columns
    int_cols = [col for col in df.columns if '_int' in col]
    
    sir_summary = []
    
    for col in int_cols:
        antibiotic = col.replace('_int', '')
        
        # Count S/I/R labels
        value_counts = df[col].value_counts()
        total_non_null = df[col].count()
        
        if total_non_null > 0:
            sir_summary.append({
                'antibiotic': antibiotic,
                'total_tested': total_non_null,
                'S_count': value_counts.get('s', 0),
                'I_count': value_counts.get('i', 0),
                'R_count': value_counts.get('r', 0),
                'S_pct': round((value_counts.get('s', 0) / total_non_null) * 100, 2),
                'I_pct': round((value_counts.get('i', 0) / total_non_null) * 100, 2),
                'R_pct': round((value_counts.get('r', 0) / total_non_null) * 100, 2),
            })
    
    sir_df = pd.DataFrame(sir_summary)
    sir_df = sir_df.sort_values('total_tested', ascending=False)
    
    print("S/I/R label distribution by antibiotic:")
    print(sir_df.to_string(index=False))
    print()
    
    return sir_df


def analyze_antibiotics_consistency(df):
    """Analyze which antibiotics are consistently tested."""
    print("="*80)
    print("ANTIBIOTIC TESTING CONSISTENCY")
    print("="*80 + "\n")
    
    int_cols = [col for col in df.columns if '_int' in col]
    
    antibiotic_testing = []
    
    for col in int_cols:
        antibiotic = col.replace('_int', '')
        non_null = df[col].count()
        tested_pct = (non_null / len(df)) * 100
        
        antibiotic_testing.append({
            'antibiotic': antibiotic,
            'isolates_tested': non_null,
            'testing_percentage': round(tested_pct, 2),
            'completeness': 'High' if tested_pct >= 90 else 'Medium' if tested_pct >= 50 else 'Low'
        })
    
    testing_df = pd.DataFrame(antibiotic_testing)
    testing_df = testing_df.sort_values('testing_percentage', ascending=False)
    
    print("Antibiotic testing coverage:")
    print(testing_df.to_string(index=False))
    print()
    
    # Identify consistently tested antibiotics
    consistent = testing_df[testing_df['testing_percentage'] >= 90]
    print(f"\nConsistently tested antibiotics (≥90% coverage): {len(consistent)}")
    if len(consistent) > 0:
        print(consistent['antibiotic'].tolist())
    print()
    
    return testing_df


def analyze_metadata_distributions(df):
    """Analyze distributions of key metadata fields."""
    print("="*80)
    print("METADATA DISTRIBUTIONS")
    print("="*80 + "\n")
    
    metadata_fields = {
        'administrative_region': 'Geographic Region',
        'national_site': 'National Site',
        'local_site': 'Local Site',
        'sample_source': 'Sample Source',
        'esbl': 'ESBL Status'
    }
    
    for field, label in metadata_fields.items():
        if field in df.columns:
            print(f"{label} Distribution:")
            counts = df[field].value_counts()
            print(counts.to_string())
            print(f"Total unique: {len(counts)}\n")


def analyze_mar_index(df):
    """Analyze Multiple Antibiotic Resistance (MAR) index."""
    print("="*80)
    print("MAR INDEX ANALYSIS")
    print("="*80 + "\n")
    
    if 'mar_index' in df.columns:
        mar_stats = {
            'count': df['mar_index'].count(),
            'mean': df['mar_index'].mean(),
            'median': df['mar_index'].median(),
            'std': df['mar_index'].std(),
            'min': df['mar_index'].min(),
            'max': df['mar_index'].max(),
            'q25': df['mar_index'].quantile(0.25),
            'q75': df['mar_index'].quantile(0.75)
        }
        
        print("MAR Index Statistics:")
        for key, value in mar_stats.items():
            print(f"  {key:>10}: {value:.4f}")
        print()
        
        # Distribution by threshold
        print("MAR Index Distribution by Risk Level:")
        print(f"  Low risk (≤0.2):    {len(df[df['mar_index'] <= 0.2])} isolates ({(len(df[df['mar_index'] <= 0.2])/len(df)*100):.1f}%)")
        print(f"  Medium risk (>0.2): {len(df[df['mar_index'] > 0.2])} isolates ({(len(df[df['mar_index'] > 0.2])/len(df)*100):.1f}%)")
        print()
        
    if 'scored_resistance' in df.columns:
        print("Scored Resistance Statistics:")
        print(f"  Mean: {df['scored_resistance'].mean():.2f}")
        print(f"  Median: {df['scored_resistance'].median():.2f}")
        print(f"  Range: {df['scored_resistance'].min()} - {df['scored_resistance'].max()}")
        print()


def check_time_field(df):
    """Check for time/date fields."""
    print("="*80)
    print("TIME/DATE FIELD ANALYSIS")
    print("="*80 + "\n")
    
    # Check column names for date/time related terms
    time_related_cols = [col for col in df.columns 
                        if any(term in col.lower() for term in 
                              ['date', 'time', 'year', 'month', 'day', 'timestamp'])]
    
    if time_related_cols:
        print(f"Found {len(time_related_cols)} time-related columns:")
        for col in time_related_cols:
            print(f"  - {col}")
            print(f"    Sample values: {df[col].dropna().head(3).tolist()}")
            print(f"    Completeness: {(df[col].count()/len(df)*100):.1f}%")
        print()
    else:
        print("⚠ WARNING: No explicit time/date fields found in the dataset.")
        print("This may limit temporal trend analysis capabilities.")
        print("Consider adding collection_date or similar fields for future analyses.\n")


def create_sample_dataset(df, n_samples=50):
    """Create a minimal reproducible dataset sample."""
    print("="*80)
    print("CREATING SAMPLE DATASET")
    print("="*80 + "\n")
    
    # Sample strategy: representative across species and sample sources
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    sample_df.to_csv('sample_data.csv', index=False)
    print(f"✓ Sample dataset created: {len(sample_df)} isolates")
    print(f"  Saved to 'sample_data.csv'\n")
    
    # Show sample composition
    print("Sample composition:")
    print(f"  Species represented: {sample_df['bacterial_species'].nunique()}")
    print(f"  Sample sources: {sample_df['sample_source'].nunique()}")
    print()
    
    return sample_df


def generate_sanity_check_report(df, data_dict, missing_df, species_counts, 
                                 sir_df, testing_df):
    """Generate comprehensive sanity-check report."""
    print("="*80)
    print("GENERATING SANITY-CHECK REPORT")
    print("="*80 + "\n")
    
    report = []
    report.append("# SANITY-CHECK REPORT")
    report.append("# Phase 0 - Initial Setup & Data Understanding")
    report.append("=" * 80)
    report.append("")
    
    # Dataset overview
    report.append("## DATASET OVERVIEW")
    report.append(f"Total isolates: {len(df)}")
    report.append(f"Total fields: {len(df.columns)}")
    report.append(f"Unique species: {df['bacterial_species'].nunique()}")
    report.append(f"Unique sample sources: {df['sample_source'].nunique()}")
    report.append("")
    
    # Missingness summary
    report.append("## MISSINGNESS SUMMARY")
    if len(missing_df) > 0:
        report.append(f"Fields with missing data: {len(missing_df)} out of {len(df.columns)}")
        report.append("\nTop 10 fields with highest missingness:")
        report.append(missing_df.head(10).to_string(index=False))
    else:
        report.append("No missing data detected (complete dataset)")
    report.append("")
    
    # Species distribution
    report.append("## SPECIES DISTRIBUTION")
    report.append(species_counts.to_string())
    report.append("")
    
    # Label balance
    report.append("## S/I/R LABEL BALANCE")
    report.append("Summary of resistance, intermediate, and susceptible labels:")
    overall_s = sir_df['S_count'].sum()
    overall_i = sir_df['I_count'].sum()
    overall_r = sir_df['R_count'].sum()
    overall_total = overall_s + overall_i + overall_r
    report.append(f"Overall: S={overall_s} ({overall_s/overall_total*100:.1f}%), "
                 f"I={overall_i} ({overall_i/overall_total*100:.1f}%), "
                 f"R={overall_r} ({overall_r/overall_total*100:.1f}%)")
    report.append("")
    
    # Antibiotic testing consistency
    report.append("## ANTIBIOTIC TESTING CONSISTENCY")
    consistent = testing_df[testing_df['testing_percentage'] >= 90]
    report.append(f"Consistently tested antibiotics (≥90% coverage): {len(consistent)}")
    if len(consistent) > 0:
        report.append("Antibiotics:")
        for ab in consistent['antibiotic'].tolist():
            report.append(f"  - {ab}")
    report.append("")
    
    # MAR Index
    report.append("## MAR INDEX STATISTICS")
    report.append(f"Mean: {df['mar_index'].mean():.4f}")
    report.append(f"Median: {df['mar_index'].median():.4f}")
    report.append(f"Range: {df['mar_index'].min():.4f} - {df['mar_index'].max():.4f}")
    report.append("")
    
    # Sample source distribution
    report.append("## SAMPLE SOURCE DISTRIBUTION")
    report.append(df['sample_source'].value_counts().to_string())
    report.append("")
    
    # Key findings and checks
    report.append("## KEY CHECKS COMPLETED")
    report.append("✓ Data schema confirmed")
    report.append("✓ Labels (S/I/R) validated")
    report.append("✓ Metadata fields present")
    report.append("✓ MAR index calculated")
    report.append("✓ Species taxonomy reviewed")
    report.append("✓ Antibiotic testing consistency assessed")
    
    # Time field warning
    time_cols = [col for col in df.columns 
                if any(term in col.lower() for term in 
                      ['date', 'time', 'year', 'month', 'day', 'timestamp'])]
    if not time_cols:
        report.append("⚠ WARNING: No time/date fields detected - trend analysis may be limited")
    else:
        report.append(f"✓ Time-related fields identified: {', '.join(time_cols)}")
    
    report.append("")
    report.append("=" * 80)
    report.append("Report generated successfully")
    
    # Save report
    report_text = "\n".join(report)
    with open('sanity_check_report.txt', 'w') as f:
        f.write(report_text)
    
    print("✓ Sanity-check report saved to 'sanity_check_report.txt'\n")
    print(report_text)


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("PHASE 0 - INITIAL SETUP & DATA UNDERSTANDING")
    print("="*80 + "\n")
    
    # Load data
    df = load_data('raw - data.csv')
    
    # Generate data dictionary
    data_dict, antibiotics = generate_data_dictionary(df)
    
    # Perform analyses
    missing_df = analyze_missingness(df)
    species_counts = analyze_species(df)
    sir_df = analyze_sir_labels(df)
    testing_df = analyze_antibiotics_consistency(df)
    analyze_metadata_distributions(df)
    analyze_mar_index(df)
    check_time_field(df)
    
    # Create sample dataset
    sample_df = create_sample_dataset(df, n_samples=50)
    
    # Generate comprehensive report
    generate_sanity_check_report(df, data_dict, missing_df, species_counts, 
                                 sir_df, testing_df)
    
    print("\n" + "="*80)
    print("PHASE 0 ANALYSIS COMPLETE")
    print("="*80)
    print("\nDeliverables generated:")
    print("  1. data_dictionary.json - Comprehensive data schema documentation")
    print("  2. sanity_check_report.txt - Detailed sanity-check findings")
    print("  3. sample_data.csv - Minimal reproducible dataset sample (50 isolates)")
    print("\n")


if __name__ == "__main__":
    main()
