"""
Source Attribution Analysis

This module provides source attribution methods:
- Prevalence by sample source
- Source-specific resistance profiles
- Cross-source comparison
- Environmental vs clinical source analysis
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class SourceAttributor:
    """
    Source attribution analysis for resistance patterns.
    
    Methods:
    - Prevalence by sample source
    - Source-specific profiles
    - Statistical comparisons between sources
    - Environmental reservoirs identification
    """
    
    def __init__(self):
        """Initialize source attributor."""
        self.source_profiles = None
        self.comparisons = None
        self.attribution = None
        
    def prepare_data(self, df):
        """
        Prepare data for source attribution.
        
        Parameters:
        -----------
        df : DataFrame
            Input data with sample_source column
            
        Returns:
        --------
        source_df : DataFrame
            Data prepared for analysis
        """
        print("Preparing source attribution data...")
        
        if 'sample_source' not in df.columns:
            raise ValueError("Data must contain 'sample_source' column")
            
        print(f"  Found {df['sample_source'].nunique()} unique sample sources")
        print(f"  Sources: {', '.join(df['sample_source'].unique())}")
        
        return df
        
    def analyze_by_source(self, df, antibiotics=None):
        """
        Analyze resistance prevalence by sample source.
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        antibiotics : list, optional
            Antibiotics to analyze
            
        Returns:
        --------
        source_profiles : dict
            Resistance profiles by source
        """
        print("\nAnalyzing resistance by sample source...")
        
        if antibiotics is None:
            antibiotics = [col.replace('_int', '') for col in df.columns if col.endswith('_int')]
            
        source_profiles = {}
        
        for source in df['sample_source'].unique():
            source_df = df[df['sample_source'] == source]
            
            profile = {
                'isolate_count': len(source_df),
                'species_distribution': source_df['bacterial_species'].value_counts().to_dict() if 'bacterial_species' in df.columns else {},
                'resistance': {}
            }
            
            # Calculate resistance for each antibiotic
            for antibiotic in antibiotics:
                int_col = f'{antibiotic}_int'
                if int_col not in source_df.columns:
                    continue
                    
                total = (source_df[int_col].notna() & (source_df[int_col] != 'not_tested')).sum()
                resistant = (source_df[int_col] == 'r').sum()
                intermediate = (source_df[int_col] == 'i').sum()
                susceptible = (source_df[int_col] == 's').sum()
                
                if total > 0:
                    profile['resistance'][antibiotic] = {
                        'total_tested': total,
                        'resistant_count': resistant,
                        'intermediate_count': intermediate,
                        'susceptible_count': susceptible,
                        'resistance_%': (resistant / total) * 100,
                        'intermediate_%': (intermediate / total) * 100,
                        'susceptible_%': (susceptible / total) * 100
                    }
                    
            # ESBL prevalence if available
            if 'esbl' in source_df.columns:
                esbl_total = source_df['esbl'].notna().sum()
                esbl_pos = (source_df['esbl'] == 'pos').sum()
                if esbl_total > 0:
                    profile['esbl_prevalence_%'] = (esbl_pos / esbl_total) * 100
                    
            source_profiles[source] = profile
            
        self.source_profiles = source_profiles
        
        # Print summary
        for source, profile in source_profiles.items():
            print(f"\n  {source}:")
            print(f"    Isolates: {profile['isolate_count']}")
            print(f"    Antibiotics analyzed: {len(profile['resistance'])}")
            if 'esbl_prevalence_%' in profile:
                print(f"    ESBL prevalence: {profile['esbl_prevalence_%']:.1f}%")
                
        return source_profiles
        
    def compare_sources(self, df, antibiotics=None, test='chi2'):
        """
        Statistical comparison between sample sources.
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        antibiotics : list, optional
            Antibiotics to analyze
        test : str
            Statistical test ('chi2' or 'fisher')
            
        Returns:
        --------
        comparisons : dict
            Statistical comparisons between sources
        """
        print(f"\nComparing sources (test={test})...")
        
        if antibiotics is None:
            antibiotics = [col.replace('_int', '') for col in df.columns if col.endswith('_int')]
            
        sources = df['sample_source'].unique()
        comparisons = {}
        
        for antibiotic in antibiotics:
            int_col = f'{antibiotic}_int'
            if int_col not in df.columns:
                continue
                
            antibiotic_comparisons = []
            
            # Pairwise comparisons
            for i, source1 in enumerate(sources):
                for source2 in sources[i+1:]:
                    source1_df = df[df['sample_source'] == source1]
                    source2_df = df[df['sample_source'] == source2]
                    
                    # Count resistant and non-resistant
                    s1_resistant = (source1_df[int_col] == 'r').sum()
                    s1_susceptible = ((source1_df[int_col] == 's') | (source1_df[int_col] == 'i')).sum()
                    
                    s2_resistant = (source2_df[int_col] == 'r').sum()
                    s2_susceptible = ((source2_df[int_col] == 's') | (source2_df[int_col] == 'i')).sum()
                    
                    # Skip if insufficient data
                    if s1_resistant + s1_susceptible < 5 or s2_resistant + s2_susceptible < 5:
                        continue
                        
                    # Contingency table
                    contingency = np.array([
                        [s1_resistant, s1_susceptible],
                        [s2_resistant, s2_susceptible]
                    ])
                    
                    # Chi-square test
                    if test == 'chi2':
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                        test_stat = chi2
                    else:  # Fisher's exact test
                        odds_ratio, p_value = stats.fisher_exact(contingency)
                        test_stat = odds_ratio
                        
                    comparison = {
                        'source1': source1,
                        'source2': source2,
                        'source1_resistance_%': (s1_resistant / (s1_resistant + s1_susceptible)) * 100,
                        'source2_resistance_%': (s2_resistant / (s2_resistant + s2_susceptible)) * 100,
                        'test_statistic': test_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                    
                    antibiotic_comparisons.append(comparison)
                    
            if antibiotic_comparisons:
                comparisons[antibiotic] = antibiotic_comparisons
                
        self.comparisons = comparisons
        
        # Print significant differences
        significant_count = 0
        for antibiotic, comps in comparisons.items():
            sig_comps = [c for c in comps if c['significant']]
            if sig_comps:
                significant_count += len(sig_comps)
                print(f"\n  {antibiotic}: {len(sig_comps)} significant differences")
                for comp in sig_comps[:3]:  # Show top 3
                    print(f"    {comp['source1']} ({comp['source1_resistance_%']:.1f}%) vs "
                          f"{comp['source2']} ({comp['source2_resistance_%']:.1f}%), p={comp['p_value']:.4f}")
                          
        print(f"\nTotal significant differences found: {significant_count}")
        
        return comparisons
        
    def identify_reservoirs(self, df, threshold_percentile=75):
        """
        Identify potential environmental reservoirs of resistance.
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        threshold_percentile : float
            Percentile threshold for reservoir identification
            
        Returns:
        --------
        reservoirs : dict
            Potential resistance reservoirs
        """
        print(f"\nIdentifying potential reservoirs (threshold={threshold_percentile}th percentile)...")
        
        if self.source_profiles is None:
            self.analyze_by_source(df)
            
        reservoirs = {}
        
        # For each antibiotic, identify sources with high resistance
        antibiotics = set()
        for profile in self.source_profiles.values():
            antibiotics.update(profile['resistance'].keys())
            
        for antibiotic in antibiotics:
            # Get resistance rates across sources
            resistance_rates = {}
            for source, profile in self.source_profiles.items():
                if antibiotic in profile['resistance']:
                    resistance_rates[source] = profile['resistance'][antibiotic]['resistance_%']
                    
            if not resistance_rates:
                continue
                
            # Calculate threshold
            rates = list(resistance_rates.values())
            threshold = np.percentile(rates, threshold_percentile)
            
            # Identify reservoirs
            antibiotic_reservoirs = {
                source: rate 
                for source, rate in resistance_rates.items() 
                if rate >= threshold
            }
            
            if antibiotic_reservoirs:
                reservoirs[antibiotic] = {
                    'sources': antibiotic_reservoirs,
                    'threshold_%': threshold,
                    'mean_%': np.mean(rates)
                }
                
        self.attribution = reservoirs
        
        # Print summary
        for antibiotic, data in reservoirs.items():
            print(f"\n  {antibiotic}:")
            print(f"    Threshold: {data['threshold_%']:.1f}% (mean: {data['mean_%']:.1f}%)")
            print(f"    Potential reservoirs: {', '.join(data['sources'].keys())}")
            for source, rate in data['sources'].items():
                print(f"      {source}: {rate:.1f}%")
                
        return reservoirs
        
    def generate_attribution_report(self):
        """
        Generate comprehensive source attribution report.
        
        Returns:
        --------
        report : dict
            Detailed attribution analysis
        """
        report = {
            'summary': {
                'sources_analyzed': len(self.source_profiles) if self.source_profiles else 0,
                'total_isolates': sum(p['isolate_count'] for p in self.source_profiles.values()) if self.source_profiles else 0
            }
        }
        
        if self.source_profiles:
            report['source_profiles'] = self.source_profiles
            
        if self.comparisons:
            # Summarize significant differences
            all_significant = []
            for antibiotic, comps in self.comparisons.items():
                all_significant.extend([
                    {
                        'antibiotic': antibiotic,
                        **comp
                    }
                    for comp in comps if comp['significant']
                ])
                
            report['significant_differences'] = {
                'total': len(all_significant),
                'details': all_significant
            }
            
        if self.attribution:
            report['reservoirs'] = self.attribution
            
        return report
        
    def save_results(self, output_prefix='source_attribution'):
        """
        Save source attribution results.
        
        Parameters:
        -----------
        output_prefix : str
            Prefix for output files
        """
        import json
        
        report = self.generate_attribution_report()
        
        with open(f'{output_prefix}_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved source attribution report to {output_prefix}_report.json")
