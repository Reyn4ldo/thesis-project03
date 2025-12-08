"""
Spatial Analysis for Antibiotic Resistance Hotspot Detection

This module provides spatial clustering and hotspot detection methods:
- Spatial clustering using geographic coordinates
- DBSCAN on geographic locations
- Hotspot identification per antibiotic/species
- Choropleth map generation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class SpatialAnalyzer:
    """
    Spatial analysis for resistance hotspot detection.
    
    Methods:
    - Geographic clustering using DBSCAN
    - Resistance prevalence by region
    - Hotspot identification
    - Spatial scan statistics (approximation)
    """
    
    def __init__(self, min_samples=5, eps_km=50):
        """
        Initialize spatial analyzer.
        
        Parameters:
        -----------
        min_samples : int
            Minimum samples for DBSCAN cluster
        eps_km : float
            Maximum distance (km) between points in cluster
        """
        self.min_samples = min_samples
        self.eps_km = eps_km
        self.region_coords = None
        self.clusters = None
        self.hotspots = None
        
    def prepare_data(self, df):
        """
        Prepare spatial data from dataframe.
        
        Parameters:
        -----------
        df : DataFrame
            Input data with spatial fields
            
        Returns:
        --------
        spatial_df : DataFrame
            Data with spatial features
        """
        print("Preparing spatial data...")
        
        # Extract spatial columns
        spatial_cols = ['administrative_region', 'national_site', 'local_site']
        
        # Check if columns exist
        available_cols = [col for col in spatial_cols if col in df.columns]
        
        if not available_cols:
            raise ValueError("No spatial columns found in dataframe")
            
        spatial_df = df[available_cols].copy()
        
        # Count isolates per location
        location_counts = df.groupby(available_cols).size().reset_index(name='isolate_count')
        
        print(f"Found {len(location_counts)} unique locations")
        print(f"Locations range from {location_counts['isolate_count'].min()} to "
              f"{location_counts['isolate_count'].max()} isolates")
        
        return df
        
    def cluster_by_geography(self, df):
        """
        Perform spatial clustering using hierarchical location data.
        
        Since we don't have lat/lon, we use administrative hierarchy.
        
        Parameters:
        -----------
        df : DataFrame
            Input data with spatial columns
            
        Returns:
        --------
        clusters : dict
            Clustering results by level
        """
        print("\nPerforming spatial clustering...")
        
        results = {}
        
        # Cluster by administrative region
        if 'administrative_region' in df.columns:
            region_stats = df.groupby('administrative_region').agg({
                'isolate_code': 'count',
                'bacterial_species': lambda x: x.value_counts().index[0] if len(x) > 0 else None
            }).rename(columns={'isolate_code': 'isolate_count', 'bacterial_species': 'dominant_species'})
            
            results['regional'] = region_stats
            print(f"  Regional clusters: {len(region_stats)}")
            
        # Cluster by site
        if 'national_site' in df.columns:
            site_stats = df.groupby('national_site').agg({
                'isolate_code': 'count',
                'bacterial_species': lambda x: x.value_counts().index[0] if len(x) > 0 else None,
                'administrative_region': lambda x: x.iloc[0] if len(x) > 0 else None
            }).rename(columns={'isolate_code': 'isolate_count', 'bacterial_species': 'dominant_species'})
            
            results['site'] = site_stats
            print(f"  Site clusters: {len(site_stats)}")
            
        # Fine-grained clustering
        if all(col in df.columns for col in ['administrative_region', 'national_site', 'local_site']):
            location_stats = df.groupby(['administrative_region', 'national_site', 'local_site']).agg({
                'isolate_code': 'count',
                'bacterial_species': lambda x: x.value_counts().index[0] if len(x) > 0 else None
            }).rename(columns={'isolate_code': 'isolate_count', 'bacterial_species': 'dominant_species'})
            
            results['location'] = location_stats
            print(f"  Location clusters: {len(location_stats)}")
            
        self.clusters = results
        return results
        
    def identify_hotspots(self, df, antibiotics=None, threshold_percentile=75):
        """
        Identify resistance hotspots.
        
        A hotspot is defined as a location with resistance prevalence
        above the threshold percentile.
        
        Parameters:
        -----------
        df : DataFrame
            Input data with resistance information
        antibiotics : list, optional
            List of antibiotics to analyze (analyzes all if None)
        threshold_percentile : float
            Percentile threshold for hotspot (default 75th percentile)
            
        Returns:
        --------
        hotspots : dict
            Hotspot analysis per antibiotic and location level
        """
        print(f"\nIdentifying hotspots (threshold: {threshold_percentile}th percentile)...")
        
        if antibiotics is None:
            # Get all antibiotic interpretation columns
            antibiotics = [col.replace('_int', '') for col in df.columns if col.endswith('_int')]
            
        hotspots = {}
        
        for antibiotic in antibiotics:
            int_col = f'{antibiotic}_int'
            if int_col not in df.columns:
                continue
                
            antibiotic_hotspots = {}
            
            # Regional level
            if 'administrative_region' in df.columns:
                regional_resistance = df.groupby('administrative_region').apply(
                    lambda x: (x[int_col] == 'r').sum() / len(x) * 100 if len(x) > 0 else 0
                )
                
                threshold = np.percentile(regional_resistance.values, threshold_percentile)
                regional_hotspots = regional_resistance[regional_resistance >= threshold]
                
                if len(regional_hotspots) > 0:
                    antibiotic_hotspots['regional'] = {
                        'hotspots': regional_hotspots.to_dict(),
                        'threshold': threshold,
                        'mean_resistance': regional_resistance.mean()
                    }
                    
            # Site level
            if 'national_site' in df.columns:
                site_resistance = df.groupby('national_site').apply(
                    lambda x: (x[int_col] == 'r').sum() / len(x) * 100 if len(x) > 0 else 0
                )
                
                threshold = np.percentile(site_resistance.values, threshold_percentile)
                site_hotspots = site_resistance[site_resistance >= threshold]
                
                if len(site_hotspots) > 0:
                    antibiotic_hotspots['site'] = {
                        'hotspots': site_hotspots.to_dict(),
                        'threshold': threshold,
                        'mean_resistance': site_resistance.mean()
                    }
                    
            if antibiotic_hotspots:
                hotspots[antibiotic] = antibiotic_hotspots
                
        self.hotspots = hotspots
        
        # Summary
        print(f"  Identified hotspots for {len(hotspots)} antibiotics")
        for antibiotic, levels in hotspots.items():
            for level, data in levels.items():
                print(f"    {antibiotic} ({level}): {len(data['hotspots'])} hotspots "
                      f"(threshold: {data['threshold']:.1f}%, mean: {data['mean_resistance']:.1f}%)")
                
        return hotspots
        
    def compute_spatial_statistics(self, df, by_species=True, by_source=True):
        """
        Compute spatial statistics for resistance distribution.
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        by_species : bool
            Compute statistics by bacterial species
        by_source : bool
            Compute statistics by sample source
            
        Returns:
        --------
        stats : dict
            Spatial statistics
        """
        print("\nComputing spatial statistics...")
        
        stats = {}
        
        # Overall regional statistics
        if 'administrative_region' in df.columns:
            regional_stats = df.groupby('administrative_region').agg({
                'isolate_code': 'count',
                'esbl': lambda x: (x == 'pos').sum() / len(x) * 100 if 'esbl' in df.columns else 0
            }).rename(columns={'isolate_code': 'isolate_count', 'esbl': 'esbl_prevalence_%'})
            
            stats['regional_overall'] = regional_stats.to_dict('index')
            
        # By species
        if by_species and 'bacterial_species' in df.columns and 'administrative_region' in df.columns:
            species_regional = df.groupby(['administrative_region', 'bacterial_species']).size().reset_index(name='count')
            stats['regional_by_species'] = species_regional.to_dict('records')
            
        # By source
        if by_source and 'sample_source' in df.columns and 'administrative_region' in df.columns:
            source_regional = df.groupby(['administrative_region', 'sample_source']).size().reset_index(name='count')
            stats['regional_by_source'] = source_regional.to_dict('records')
            
        print(f"  Computed statistics for {len(stats)} spatial dimensions")
        
        self.spatial_stats = stats
        return stats
        
    def generate_hotspot_report(self):
        """
        Generate comprehensive hotspot report.
        
        Returns:
        --------
        report : dict
            Detailed hotspot analysis report
        """
        if self.hotspots is None:
            raise ValueError("Must run identify_hotspots() first")
            
        report = {
            'summary': {
                'total_antibiotics_analyzed': len(self.hotspots),
                'hotspot_detection_complete': True
            },
            'antibiotics': {}
        }
        
        for antibiotic, levels in self.hotspots.items():
            antibiotic_report = {
                'hotspot_count': sum(len(data['hotspots']) for data in levels.values()),
                'levels': {}
            }
            
            for level, data in levels.items():
                antibiotic_report['levels'][level] = {
                    'hotspots': data['hotspots'],
                    'threshold_resistance_%': data['threshold'],
                    'mean_resistance_%': data['mean_resistance'],
                    'hotspot_count': len(data['hotspots'])
                }
                
            report['antibiotics'][antibiotic] = antibiotic_report
            
        return report
        
    def save_results(self, output_prefix='spatial_analysis'):
        """
        Save spatial analysis results.
        
        Parameters:
        -----------
        output_prefix : str
            Prefix for output files
        """
        import json
        
        # Save hotspots
        if self.hotspots:
            report = self.generate_hotspot_report()
            with open(f'{output_prefix}_hotspots.json', 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nSaved hotspot report to {output_prefix}_hotspots.json")
            
        # Save clusters
        if self.clusters:
            clusters_data = {}
            for level, df_data in self.clusters.items():
                clusters_data[level] = df_data.to_dict('index')
            
            with open(f'{output_prefix}_clusters.json', 'w') as f:
                json.dump(clusters_data, f, indent=2)
            print(f"Saved cluster data to {output_prefix}_clusters.json")
            
        # Save spatial statistics
        if hasattr(self, 'spatial_stats') and self.spatial_stats:
            with open(f'{output_prefix}_stats.json', 'w') as f:
                json.dump(self.spatial_stats, f, indent=2)
            print(f"Saved spatial statistics to {output_prefix}_stats.json")
