"""
Spatio-temporal Visualization

This module provides visualization tools for:
- Choropleth maps for resistance prevalence
- Heatmaps for spatial patterns
- Time series plots
- Dashboard generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SpatioTemporalVisualizer:
    """
    Visualization tools for spatio-temporal analysis.
    
    Methods:
    - Choropleth maps (simplified without geographic boundaries)
    - Heatmaps for resistance patterns
    - Time series plots
    - Dashboard generation
    """
    
    def __init__(self, output_dir='visualizations'):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        output_dir : str
            Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_regional_heatmap(self, df, antibiotics=None, figsize=(14, 8)):
        """
        Create heatmap of resistance by region and antibiotic.
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        antibiotics : list, optional
            Antibiotics to include
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        print("Creating regional resistance heatmap...")
        
        if 'administrative_region' not in df.columns:
            print("  No regional data available")
            return None
            
        if antibiotics is None:
            antibiotics = [col.replace('_int', '') for col in df.columns if col.endswith('_int')][:15]  # Limit to 15
            
        # Calculate resistance rates
        resistance_matrix = []
        regions = sorted(df['administrative_region'].unique())
        
        for region in regions:
            region_df = df[df['administrative_region'] == region]
            row = []
            
            for antibiotic in antibiotics:
                int_col = f'{antibiotic}_int'
                if int_col in region_df.columns:
                    total = (region_df[int_col].notna() & (region_df[int_col] != 'not_tested')).sum()
                    resistant = (region_df[int_col] == 'r').sum()
                    rate = (resistant / total * 100) if total > 0 else np.nan
                else:
                    rate = np.nan
                row.append(rate)
                
            resistance_matrix.append(row)
            
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(resistance_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
        
        # Set ticks
        ax.set_xticks(np.arange(len(antibiotics)))
        ax.set_yticks(np.arange(len(regions)))
        ax.set_xticklabels(antibiotics, rotation=45, ha='right')
        ax.set_yticklabels(regions)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Resistance (%)', rotation=270, labelpad=20)
        
        # Add values
        for i in range(len(regions)):
            for j in range(len(antibiotics)):
                if not np.isnan(resistance_matrix[i][j]):
                    text = ax.text(j, i, f'{resistance_matrix[i][j]:.0f}',
                                 ha="center", va="center", color="black", fontsize=8)
                    
        ax.set_title('Resistance Prevalence by Region and Antibiotic', fontsize=14, fontweight='bold')
        ax.set_xlabel('Antibiotic', fontsize=12)
        ax.set_ylabel('Administrative Region', fontsize=12)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'regional_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        
        return fig
        
    def plot_source_comparison(self, source_profiles, figsize=(12, 8)):
        """
        Create bar plot comparing resistance across sample sources.
        
        Parameters:
        -----------
        source_profiles : dict
            Source profiles from SourceAttributor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        print("Creating source comparison plot...")
        
        # Prepare data
        antibiotics = set()
        for profile in source_profiles.values():
            antibiotics.update(profile['resistance'].keys())
            
        antibiotics = sorted(list(antibiotics))[:10]  # Limit to 10
        sources = list(source_profiles.keys())
        
        # Create data matrix
        data = []
        for source in sources:
            row = []
            for antibiotic in antibiotics:
                if antibiotic in source_profiles[source]['resistance']:
                    rate = source_profiles[source]['resistance'][antibiotic]['resistance_%']
                else:
                    rate = 0
                row.append(rate)
            data.append(row)
            
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(antibiotics))
        width = 0.8 / len(sources)
        
        for i, (source, row) in enumerate(zip(sources, data)):
            offset = (i - len(sources)/2) * width + width/2
            ax.bar(x + offset, row, width, label=source)
            
        ax.set_xlabel('Antibiotic', fontsize=12)
        ax.set_ylabel('Resistance (%)', fontsize=12)
        ax.set_title('Resistance Prevalence by Sample Source', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(antibiotics, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'source_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        
        return fig
        
    def plot_temporal_trends(self, time_series, antibiotics=None, figsize=(14, 8)):
        """
        Create time series plots for resistance trends.
        
        Parameters:
        -----------
        time_series : DataFrame
            Time series data from TemporalAnalyzer
        antibiotics : list, optional
            Antibiotics to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        if time_series is None:
            print("No temporal data available for plotting")
            return None
            
        print("Creating temporal trend plots...")
        
        if antibiotics is None:
            antibiotics = time_series['antibiotic'].unique()[:6]  # Limit to 6
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, antibiotic in enumerate(antibiotics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            antibiotic_data = time_series[time_series['antibiotic'] == antibiotic]
            
            if 'species' in antibiotic_data.columns:
                # Plot by species
                for species in antibiotic_data['species'].unique()[:3]:  # Max 3 species
                    species_data = antibiotic_data[antibiotic_data['species'] == species]
                    ax.plot(species_data['date'], species_data['resistance_%'], 
                           label=species, marker='o', markersize=3)
                ax.legend(fontsize=8)
            else:
                ax.plot(antibiotic_data['date'], antibiotic_data['resistance_%'],
                       marker='o', markersize=3, color='steelblue')
                
            ax.set_title(antibiotic, fontsize=10, fontweight='bold')
            ax.set_xlabel('Date', fontsize=9)
            ax.set_ylabel('Resistance (%)', fontsize=9)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            
        # Hide unused axes
        for i in range(len(antibiotics), len(axes)):
            axes[i].axis('off')
            
        plt.suptitle('Resistance Trends Over Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'temporal_trends.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        
        return fig
        
    def plot_hotspot_map(self, hotspots, level='regional', figsize=(12, 8)):
        """
        Create visualization of resistance hotspots.
        
        Parameters:
        -----------
        hotspots : dict
            Hotspot data from SpatialAnalyzer
        level : str
            Geographic level ('regional' or 'site')
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        print(f"Creating hotspot map ({level} level)...")
        
        # Extract data for specified level
        hotspot_data = []
        for antibiotic, levels in hotspots.items():
            if level in levels:
                for location, rate in levels[level]['hotspots'].items():
                    hotspot_data.append({
                        'antibiotic': antibiotic,
                        'location': location,
                        'resistance_%': rate
                    })
                    
        if not hotspot_data:
            print(f"  No hotspot data available for {level} level")
            return None
            
        df_hotspots = pd.DataFrame(hotspot_data)
        
        # Create pivot table
        pivot = df_hotspots.pivot(index='location', columns='antibiotic', values='resistance_%')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Resistance (%)'}, ax=ax, vmin=0, vmax=100)
        
        ax.set_title(f'Resistance Hotspots ({level.capitalize()} Level)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Antibiotic', fontsize=12)
        ax.set_ylabel('Location', fontsize=12)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f'hotspot_map_{level}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        
        return fig
        
    def create_dashboard(self, df, spatial_results=None, temporal_results=None, 
                        source_results=None, figsize=(16, 12)):
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        spatial_results : dict, optional
            Results from SpatialAnalyzer
        temporal_results : DataFrame, optional
            Time series from TemporalAnalyzer
        source_results : dict, optional
            Source profiles from SourceAttributor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : Figure
            Dashboard figure
        """
        print("Creating comprehensive dashboard...")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Regional heatmap
        ax1 = fig.add_subplot(gs[0, :])
        if 'administrative_region' in df.columns:
            self._plot_regional_summary(df, ax1)
            
        # 2. Source comparison
        ax2 = fig.add_subplot(gs[1, 0])
        if source_results:
            self._plot_source_summary(source_results, ax2)
            
        # 3. Species distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if 'bacterial_species' in df.columns:
            self._plot_species_distribution(df, ax3)
            
        # 4. Temporal trends (if available)
        ax4 = fig.add_subplot(gs[2, :])
        if temporal_results is not None:
            self._plot_trend_summary(temporal_results, ax4)
        else:
            ax4.text(0.5, 0.5, 'Temporal data not available', 
                    ha='center', va='center', fontsize=12)
            ax4.axis('off')
            
        plt.suptitle('Antibiotic Resistance Surveillance Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # Save
        output_path = self.output_dir / 'dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved dashboard to {output_path}")
        
        return fig
        
    def _plot_regional_summary(self, df, ax):
        """Helper: Plot regional summary."""
        region_counts = df.groupby('administrative_region').size().sort_values(ascending=False)
        ax.barh(range(len(region_counts)), region_counts.values, color='steelblue')
        ax.set_yticks(range(len(region_counts)))
        ax.set_yticklabels(region_counts.index, fontsize=9)
        ax.set_xlabel('Number of Isolates', fontsize=10)
        ax.set_title('Isolates by Administrative Region', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
    def _plot_source_summary(self, source_results, ax):
        """Helper: Plot source summary."""
        sources = list(source_results.keys())
        counts = [source_results[s]['isolate_count'] for s in sources]
        ax.bar(range(len(sources)), counts, color='coral')
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Number of Isolates', fontsize=10)
        ax.set_title('Isolates by Sample Source', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
    def _plot_species_distribution(self, df, ax):
        """Helper: Plot species distribution."""
        species_counts = df['bacterial_species'].value_counts()[:8]
        ax.pie(species_counts.values, labels=species_counts.index, autopct='%1.1f%%',
              textprops={'fontsize': 8})
        ax.set_title('Bacterial Species Distribution', fontsize=11, fontweight='bold')
        
    def _plot_trend_summary(self, time_series, ax):
        """Helper: Plot trend summary."""
        # Show overall trend for top antibiotics
        top_antibiotics = time_series['antibiotic'].value_counts()[:5].index
        
        for antibiotic in top_antibiotics:
            ab_data = time_series[time_series['antibiotic'] == antibiotic]
            if 'species' not in ab_data.columns:
                ax.plot(ab_data['date'], ab_data['resistance_%'], 
                       label=antibiotic, marker='o', markersize=3)
                       
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Resistance (%)', fontsize=10)
        ax.set_title('Resistance Trends (Top 5 Antibiotics)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
    def save_all_plots(self, df, spatial_analyzer=None, temporal_analyzer=None, 
                      source_attributor=None):
        """
        Generate and save all available plots.
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        spatial_analyzer : SpatialAnalyzer, optional
            Fitted spatial analyzer
        temporal_analyzer : TemporalAnalyzer, optional
            Fitted temporal analyzer  
        source_attributor : SourceAttributor, optional
            Fitted source attributor
        """
        print("\nGenerating all visualizations...")
        
        # Regional heatmap
        self.plot_regional_heatmap(df)
        
        # Source comparison
        if source_attributor and source_attributor.source_profiles:
            self.plot_source_comparison(source_attributor.source_profiles)
            
        # Temporal trends
        if temporal_analyzer and temporal_analyzer.time_series is not None:
            self.plot_temporal_trends(temporal_analyzer.time_series)
            
        # Hotspot maps
        if spatial_analyzer and spatial_analyzer.hotspots:
            self.plot_hotspot_map(spatial_analyzer.hotspots, level='regional')
            # Check for site-level hotspots
            if spatial_analyzer.hotspots and len(spatial_analyzer.hotspots) > 0:
                first_antibiotic_data = next(iter(spatial_analyzer.hotspots.values()))
                if 'site' in first_antibiotic_data:
                    self.plot_hotspot_map(spatial_analyzer.hotspots, level='site')
                
        # Dashboard
        self.create_dashboard(
            df,
            spatial_results=spatial_analyzer.clusters if spatial_analyzer else None,
            temporal_results=temporal_analyzer.time_series if temporal_analyzer else None,
            source_results=source_attributor.source_profiles if source_attributor else None
        )
        
        print(f"\nAll visualizations saved to {self.output_dir}/")
