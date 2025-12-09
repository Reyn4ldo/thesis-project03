"""
Automated Antibiogram Generator

Generates standardized antibiogram reports per species, site, and time period
following CLSI guidelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime


class AntibiogramGenerator:
    """
    Automated antibiogram generator following CLSI M39 guidelines.
    
    Generates summary tables and visualizations of antimicrobial susceptibility
    patterns by species, site, and time period.
    """
    
    def __init__(self, min_isolates: int = 30, output_dir: str = 'antibiograms'):
        """
        Initialize antibiogram generator.
        
        Parameters
        ----------
        min_isolates : int
            Minimum number of isolates required to report susceptibility (CLSI: 30)
        output_dir : str
            Directory for saving antibiogram outputs
        """
        self.min_isolates = min_isolates
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_antibiogram(
        self,
        df: pd.DataFrame,
        species: Optional[str] = None,
        site: Optional[str] = None,
        year: Optional[int] = None,
        antibiotic_columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate antibiogram summary tables.
        
        Parameters
        ----------
        df : pd.DataFrame
            Processed surveillance data
        species : str, optional
            Filter by bacterial species
        site : str, optional
            Filter by collection site
        year : int, optional
            Filter by year
        antibiotic_columns : list, optional
            List of antibiotic S/I/R columns to include
            
        Returns
        -------
        dict
            Dictionary with antibiogram tables:
            - 'susceptibility': % susceptible per antibiotic
            - 'counts': isolate counts per antibiotic
            - 'sir_distribution': full S/I/R distribution
        """
        # Filter data
        filtered_df = df.copy()
        
        if species:
            if 'bacterial_species' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['bacterial_species'] == species]
            elif 'species' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['species'] == species]
                
        if site:
            site_cols = [c for c in filtered_df.columns if 'site' in c.lower() or 'location' in c.lower()]
            if site_cols:
                filtered_df = filtered_df[filtered_df[site_cols[0]] == site]
                
        if year:
            year_cols = [c for c in filtered_df.columns if 'year' in c.lower() or 'date' in c.lower()]
            if year_cols:
                filtered_df = filtered_df[filtered_df[year_cols[0]] == year]
        
        # Auto-detect antibiotic columns if not provided
        if antibiotic_columns is None:
            # Look for columns ending with _sir or containing antibiotic names
            antibiotic_columns = [col for col in filtered_df.columns 
                                if col.endswith('_sir') or 
                                any(ab in col.lower() for ab in [
                                    'cipro', 'ampi', 'genta', 'cefta', 'imi',
                                    'mero', 'aztreo', 'tobra', 'tetra'
                                ])]
        
        # Calculate susceptibility percentages
        results = {
            'susceptibility': {},
            'counts': {},
            'sir_distribution': {}
        }
        
        for antibiotic in antibiotic_columns:
            if antibiotic not in filtered_df.columns:
                continue
                
            # Count S/I/R
            values = filtered_df[antibiotic].dropna()
            
            # Skip if insufficient isolates
            n_tested = len(values)
            if n_tested < self.min_isolates:
                continue
            
            # Calculate percentages
            value_counts = values.value_counts()
            
            n_susceptible = value_counts.get('S', 0)
            n_intermediate = value_counts.get('I', 0)
            n_resistant = value_counts.get('R', 0)
            
            pct_susceptible = (n_susceptible / n_tested) * 100
            pct_intermediate = (n_intermediate / n_tested) * 100
            pct_resistant = (n_resistant / n_tested) * 100
            
            antibiotic_name = antibiotic.replace('_sir', '').replace('_', ' ').title()
            
            results['susceptibility'][antibiotic_name] = pct_susceptible
            results['counts'][antibiotic_name] = n_tested
            results['sir_distribution'][antibiotic_name] = {
                'S': n_susceptible,
                'I': n_intermediate,
                'R': n_resistant,
                '%S': pct_susceptible,
                '%I': pct_intermediate,
                '%R': pct_resistant
            }
        
        # Convert to DataFrames
        results['susceptibility'] = pd.DataFrame.from_dict(
            results['susceptibility'], 
            orient='index', 
            columns=['% Susceptible']
        ).sort_values('% Susceptible', ascending=False)
        
        results['counts'] = pd.DataFrame.from_dict(
            results['counts'],
            orient='index',
            columns=['N Tested']
        )
        
        results['sir_distribution'] = pd.DataFrame.from_dict(
            results['sir_distribution'],
            orient='index'
        )
        
        return results
    
    def plot_antibiogram(
        self,
        antibiogram: Dict[str, pd.DataFrame],
        title: str = 'Antibiogram Summary',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create visual antibiogram chart.
        
        Parameters
        ----------
        antibiogram : dict
            Antibiogram data from generate_antibiogram()
        title : str
            Chart title
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Susceptibility bar chart
        susc_data = antibiogram['susceptibility']
        if not susc_data.empty:
            susc_data.plot(kind='barh', ax=ax1, color='steelblue', legend=False)
            ax1.set_xlabel('% Susceptible', fontsize=12)
            ax1.set_ylabel('Antibiotic', fontsize=12)
            ax1.set_title(f'{title} - Susceptibility', fontsize=14, fontweight='bold')
            ax1.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
            ax1.legend()
            ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: S/I/R stacked bar chart
        sir_data = antibiogram['sir_distribution']
        if not sir_data.empty and all(col in sir_data.columns for col in ['%S', '%I', '%R']):
            sir_pct = sir_data[['%S', '%I', '%R']].sort_values('%S', ascending=True)
            sir_pct.plot(kind='barh', stacked=True, ax=ax2, 
                        color=['green', 'yellow', 'red'], alpha=0.7)
            ax2.set_xlabel('Percentage', fontsize=12)
            ax2.set_ylabel('Antibiotic', fontsize=12)
            ax2.set_title(f'{title} - S/I/R Distribution', fontsize=14, fontweight='bold')
            ax2.legend(['Susceptible', 'Intermediate', 'Resistant'])
            ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def generate_comparative_antibiogram(
        self,
        df: pd.DataFrame,
        comparison_by: str = 'bacterial_species',
        antibiotic_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate comparative antibiogram across multiple groups.
        
        Parameters
        ----------
        df : pd.DataFrame
            Processed surveillance data
        comparison_by : str
            Column to compare by (e.g., 'bacterial_species', 'site', 'year')
        antibiotic_columns : list, optional
            Antibiotics to include
            
        Returns
        -------
        pd.DataFrame
            Comparative susceptibility table
        """
        if comparison_by not in df.columns:
            raise ValueError(f"Column '{comparison_by}' not found in data")
        
        groups = df[comparison_by].unique()
        comparative_results = {}
        
        for group in groups:
            group_data = df[df[comparison_by] == group]
            
            # Generate antibiogram for this group
            antibiogram = self.generate_antibiogram(
                group_data,
                antibiotic_columns=antibiotic_columns
            )
            
            if not antibiogram['susceptibility'].empty:
                comparative_results[group] = antibiogram['susceptibility']['% Susceptible']
        
        # Combine into single DataFrame
        comparative_df = pd.DataFrame(comparative_results)
        
        return comparative_df
    
    def save_antibiogram(
        self,
        antibiogram: Dict[str, pd.DataFrame],
        name: str,
        format: str = 'both'
    ):
        """
        Save antibiogram to file.
        
        Parameters
        ----------
        antibiogram : dict
            Antibiogram data
        name : str
            Base name for output files
        format : str
            'excel', 'csv', or 'both'
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{name}_{timestamp}"
        
        if format in ['excel', 'both']:
            excel_path = self.output_dir / f"{base_name}.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for sheet_name, data in antibiogram.items():
                    data.to_excel(writer, sheet_name=sheet_name)
            print(f"Antibiogram saved to: {excel_path}")
        
        if format in ['csv', 'both']:
            for sheet_name, data in antibiogram.items():
                csv_path = self.output_dir / f"{base_name}_{sheet_name}.csv"
                data.to_csv(csv_path)
            print(f"Antibiogram CSV files saved to: {self.output_dir}")
    
    def generate_report(
        self,
        df: pd.DataFrame,
        species_list: Optional[List[str]] = None,
        sites: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Generate comprehensive antibiogram report for multiple species/sites.
        
        Parameters
        ----------
        df : pd.DataFrame
            Processed surveillance data
        species_list : list, optional
            List of species to generate antibiograms for
        sites : list, optional
            List of sites to generate antibiograms for
            
        Returns
        -------
        dict
            Report with antibiograms for all requested combinations
        """
        report = {}
        
        # Auto-detect species if not provided
        if species_list is None:
            species_col = 'bacterial_species' if 'bacterial_species' in df.columns else 'species'
            if species_col in df.columns:
                species_list = df[species_col].value_counts().head(10).index.tolist()
        
        # Generate antibiograms
        for species in (species_list or [None]):
            species_key = species or 'all_species'
            report[species_key] = {}
            
            for site in (sites or [None]):
                site_key = site or 'all_sites'
                
                antibiogram = self.generate_antibiogram(
                    df,
                    species=species,
                    site=site
                )
                
                if not antibiogram['susceptibility'].empty:
                    report[species_key][site_key] = antibiogram
                    
                    # Save individual antibiograms
                    name = f"antibiogram_{species_key}_{site_key}"
                    self.save_antibiogram(antibiogram, name, format='both')
                    
                    # Generate and save plot
                    title = f"Antibiogram: {species_key} - {site_key}"
                    plot_path = self.output_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.png"
                    self.plot_antibiogram(antibiogram, title=title, save_path=str(plot_path))
        
        return report
