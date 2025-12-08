#!/usr/bin/env python3
"""
Phase 5 - Spatio-temporal & Epidemiological Analysis

This script performs comprehensive spatio-temporal and epidemiological analysis:
- Spatial clustering and hotspot detection
- Time series analysis and trend detection
- Source attribution analysis
- Visualization and dashboard generation

Usage:
    python phase5_spatiotemporal.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from spatiotemporal import SpatialAnalyzer, TemporalAnalyzer, SourceAttributor, SpatioTemporalVisualizer


def load_processed_data(filepath='processed_data_train.csv'):
    """
    Load processed data from Phase 1.
    
    Parameters:
    -----------
    filepath : str
        Path to processed data file
        
    Returns:
    --------
    df : DataFrame
        Processed data
    """
    print("="*80)
    print("LOADING PROCESSED DATA")
    print("="*80 + "\n")
    
    if not Path(filepath).exists():
        print(f"Processed data not found at {filepath}")
        print("Attempting to load raw data...")
        filepath = 'raw - data.csv'
        
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} isolates with {len(df.columns)} columns\n")
    
    return df


def run_spatial_analysis(df, output_dir='results/phase5'):
    """
    Run spatial analysis and hotspot detection.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    output_dir : str
        Output directory
        
    Returns:
    --------
    spatial_analyzer : SpatialAnalyzer
        Fitted spatial analyzer
    """
    print("="*80)
    print("SPATIAL ANALYSIS & HOTSPOT DETECTION")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = SpatialAnalyzer(min_samples=5, eps_km=50)
    
    # Prepare data
    df_spatial = analyzer.prepare_data(df)
    
    # Geographic clustering
    clusters = analyzer.cluster_by_geography(df_spatial)
    
    # Identify hotspots
    hotspots = analyzer.identify_hotspots(df_spatial, threshold_percentile=75)
    
    # Compute spatial statistics
    stats = analyzer.compute_spatial_statistics(df_spatial, by_species=True, by_source=True)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    analyzer.save_results(output_prefix=f'{output_dir}/spatial_analysis')
    
    return analyzer


def run_temporal_analysis(df, output_dir='results/phase5'):
    """
    Run temporal analysis and trend detection.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    output_dir : str
        Output directory
        
    Returns:
    --------
    temporal_analyzer : TemporalAnalyzer
        Fitted temporal analyzer
    """
    print("\n" + "="*80)
    print("TEMPORAL ANALYSIS & TREND DETECTION")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = TemporalAnalyzer(window_size=30, alert_threshold=1.5)
    
    # Prepare temporal data
    df_temporal = analyzer.prepare_temporal_data(df)
    
    if df_temporal is not None:
        # Compute rolling prevalence
        time_series = analyzer.compute_rolling_prevalence(
            df_temporal, 
            date_column=next(col for col in df_temporal.columns if 'date' in col.lower()),
            by_species=True
        )
        
        # Detect change points
        change_points = analyzer.detect_change_points(min_change=10)
        
        # Analyze trends
        trends = analyzer.analyze_trends()
        
        # Generate alerts
        alerts = analyzer.generate_alerts()
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        analyzer.save_results(output_prefix=f'{output_dir}/temporal_analysis')
    else:
        print("NOTE: Temporal analysis framework is ready but requires date/time data.")
        print("The module can be used when temporal information becomes available.\n")
    
    return analyzer


def run_source_attribution(df, output_dir='results/phase5'):
    """
    Run source attribution analysis.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    output_dir : str
        Output directory
        
    Returns:
    --------
    source_attributor : SourceAttributor
        Fitted source attributor
    """
    print("\n" + "="*80)
    print("SOURCE ATTRIBUTION ANALYSIS")
    print("="*80 + "\n")
    
    # Initialize attributor
    attributor = SourceAttributor()
    
    # Prepare data
    df_source = attributor.prepare_data(df)
    
    # Analyze by source
    source_profiles = attributor.analyze_by_source(df_source)
    
    # Compare sources
    comparisons = attributor.compare_sources(df_source, test='chi2')
    
    # Identify reservoirs
    reservoirs = attributor.identify_reservoirs(df_source, threshold_percentile=75)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    attributor.save_results(output_prefix=f'{output_dir}/source_attribution')
    
    return attributor


def generate_visualizations(df, spatial_analyzer, temporal_analyzer, source_attributor, 
                           output_dir='results/phase5/visualizations'):
    """
    Generate all visualizations and dashboards.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    spatial_analyzer : SpatialAnalyzer
        Fitted spatial analyzer
    temporal_analyzer : TemporalAnalyzer
        Fitted temporal analyzer
    source_attributor : SourceAttributor
        Fitted source attributor
    output_dir : str
        Output directory
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Initialize visualizer
    visualizer = SpatioTemporalVisualizer(output_dir=output_dir)
    
    # Generate all plots
    visualizer.save_all_plots(
        df,
        spatial_analyzer=spatial_analyzer,
        temporal_analyzer=temporal_analyzer,
        source_attributor=source_attributor
    )
    
    print("\nVisualization generation complete!")
    return visualizer


def generate_summary_report(spatial_analyzer, temporal_analyzer, source_attributor,
                           output_dir='results/phase5'):
    """
    Generate comprehensive summary report for Phase 5.
    
    Parameters:
    -----------
    spatial_analyzer : SpatialAnalyzer
        Fitted spatial analyzer
    temporal_analyzer : TemporalAnalyzer
        Fitted temporal analyzer
    source_attributor : SourceAttributor
        Fitted source attributor
    output_dir : str
        Output directory
    """
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80 + "\n")
    
    report = {
        'phase': 5,
        'title': 'Spatio-temporal & Epidemiological Analysis',
        'summary': {
            'spatial_analysis_complete': spatial_analyzer.hotspots is not None,
            'temporal_analysis_complete': temporal_analyzer.time_series is not None,
            'source_attribution_complete': source_attributor.source_profiles is not None
        },
        'spatial_analysis': {},
        'temporal_analysis': {},
        'source_attribution': {}
    }
    
    # Spatial analysis summary
    if spatial_analyzer.hotspots:
        report['spatial_analysis'] = {
            'clusters_detected': {
                level: len(data) 
                for level, data in spatial_analyzer.clusters.items()
            },
            'hotspots_identified': {
                antibiotic: sum(len(level_data['hotspots']) for level_data in data.values())
                for antibiotic, data in spatial_analyzer.hotspots.items()
            },
            'total_hotspots': sum(
                sum(len(level_data['hotspots']) for level_data in data.values())
                for data in spatial_analyzer.hotspots.values()
            )
        }
        
    # Temporal analysis summary
    if temporal_analyzer.time_series is not None:
        report['temporal_analysis'] = temporal_analyzer.generate_temporal_report()
    else:
        report['temporal_analysis'] = {
            'note': 'Temporal framework ready for future use with date/time data'
        }
        
    # Source attribution summary
    if source_attributor.source_profiles:
        report['source_attribution'] = {
            'sources_analyzed': len(source_attributor.source_profiles),
            'source_summary': {
                source: {
                    'isolate_count': profile['isolate_count'],
                    'antibiotics_analyzed': len(profile['resistance']),
                    'esbl_prevalence_%': profile.get('esbl_prevalence_%', 'N/A')
                }
                for source, profile in source_attributor.source_profiles.items()
            }
        }
        
        if source_attributor.comparisons:
            significant_count = sum(
                sum(1 for comp in comps if comp['significant'])
                for comps in source_attributor.comparisons.values()
            )
            report['source_attribution']['significant_differences'] = significant_count
            
        if source_attributor.attribution:
            report['source_attribution']['reservoirs_identified'] = len(source_attributor.attribution)
            
    # Save report
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / 'phase5_summary.json'
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
        
    print(f"Summary report saved to {output_file}")
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80 + "\n")
    
    if spatial_analyzer.hotspots:
        print(f"Spatial Analysis:")
        print(f"  - Total hotspots identified: {report['spatial_analysis']['total_hotspots']}")
        print(f"  - Antibiotics with hotspots: {len(report['spatial_analysis']['hotspots_identified'])}")
        
    if source_attributor.source_profiles:
        print(f"\nSource Attribution:")
        print(f"  - Sources analyzed: {report['source_attribution']['sources_analyzed']}")
        if 'significant_differences' in report['source_attribution']:
            print(f"  - Significant differences detected: {report['source_attribution']['significant_differences']}")
        if 'reservoirs_identified' in report['source_attribution']:
            print(f"  - Potential reservoirs identified: {report['source_attribution']['reservoirs_identified']}")
            
    if temporal_analyzer.time_series is not None:
        print(f"\nTemporal Analysis:")
        if temporal_analyzer.trends:
            print(f"  - Trends analyzed: {len(temporal_analyzer.trends)}")
        if temporal_analyzer.alerts:
            print(f"  - Alerts generated: {len(temporal_analyzer.alerts)}")
    else:
        print(f"\nTemporal Analysis:")
        print(f"  - Framework ready for future temporal data")
        
    return report


def main():
    """Main execution function."""
    print("\n")
    print("="*80)
    print("PHASE 5: SPATIO-TEMPORAL & EPIDEMIOLOGICAL ANALYSIS")
    print("="*80)
    print("\n")
    
    # Load data
    df = load_processed_data()
    
    # Create output directory
    output_dir = 'results/phase5'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run spatial analysis
    spatial_analyzer = run_spatial_analysis(df, output_dir)
    
    # Run temporal analysis
    temporal_analyzer = run_temporal_analysis(df, output_dir)
    
    # Run source attribution
    source_attributor = run_source_attribution(df, output_dir)
    
    # Generate visualizations
    visualizer = generate_visualizations(
        df, spatial_analyzer, temporal_analyzer, source_attributor,
        output_dir=f'{output_dir}/visualizations'
    )
    
    # Generate summary report
    report = generate_summary_report(
        spatial_analyzer, temporal_analyzer, source_attributor,
        output_dir=output_dir
    )
    
    print("\n" + "="*80)
    print("PHASE 5 ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"Visualizations saved to: {output_dir}/visualizations/")
    print("\nAnalysis components:")
    print("  ✓ Spatial clustering and hotspot detection")
    print("  ✓ Source attribution and comparison")
    if temporal_analyzer.time_series is not None:
        print("  ✓ Temporal trends and alert generation")
    else:
        print("  ○ Temporal framework (ready for future date/time data)")
    print("  ✓ Comprehensive visualizations and dashboard")
    print("\n")


if __name__ == '__main__':
    main()
