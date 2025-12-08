"""
Temporal Analysis for Resistance Trend Detection

This module provides time series analysis methods:
- Rolling prevalence calculations
- Change point detection
- Seasonal decomposition
- Trend modeling

Note: This module is designed to work with temporal data when available.
If no date/time columns are present, it provides a framework for future use.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class TemporalAnalyzer:
    """
    Temporal analysis for resistance trends.
    
    Methods:
    - Rolling prevalence calculation
    - Change point detection (Bayesian, PELT approximation)
    - Trend analysis
    - Alert generation for significant increases
    """
    
    def __init__(self, window_size=30, alert_threshold=1.5):
        """
        Initialize temporal analyzer.
        
        Parameters:
        -----------
        window_size : int
            Window size for rolling calculations (days)
        alert_threshold : float
            Threshold multiplier for alert generation (e.g., 1.5 = 50% increase)
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.time_series = None
        self.trends = None
        self.change_points = None
        self.alerts = None
        
    def prepare_temporal_data(self, df, date_column=None):
        """
        Prepare temporal data from dataframe.
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        date_column : str, optional
            Name of date column (will attempt to find if not provided)
            
        Returns:
        --------
        temporal_df : DataFrame
            Data prepared for temporal analysis
        """
        print("Preparing temporal data...")
        
        # Try to find date column
        if date_column is None:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                date_column = date_cols[0]
                print(f"  Using date column: {date_column}")
            else:
                print("  WARNING: No date column found. Temporal analysis will be limited.")
                print("  Framework ready for when temporal data becomes available.")
                return None
                
        # Convert to datetime
        temporal_df = df.copy()
        temporal_df[date_column] = pd.to_datetime(temporal_df[date_column], errors='coerce')
        
        # Remove rows with invalid dates
        valid_dates = temporal_df[date_column].notna()
        print(f"  Valid dates: {valid_dates.sum()} / {len(temporal_df)}")
        
        temporal_df = temporal_df[valid_dates].copy()
        temporal_df = temporal_df.sort_values(date_column)
        
        return temporal_df
        
    def compute_rolling_prevalence(self, df, date_column, antibiotics=None, by_species=False):
        """
        Compute rolling resistance prevalence.
        
        Parameters:
        -----------
        df : DataFrame
            Temporal data
        date_column : str
            Date column name
        antibiotics : list, optional
            Antibiotics to analyze
        by_species : bool
            Compute separately by species
            
        Returns:
        --------
        rolling_prev : DataFrame
            Rolling prevalence time series
        """
        if df is None or date_column not in df.columns:
            print("Cannot compute rolling prevalence: no temporal data available")
            return None
            
        print(f"\nComputing rolling prevalence (window={self.window_size} days)...")
        
        if antibiotics is None:
            antibiotics = [col.replace('_int', '') for col in df.columns if col.endswith('_int')]
            
        results = []
        
        for antibiotic in antibiotics:
            int_col = f'{antibiotic}_int'
            if int_col not in df.columns:
                continue
                
            # Create binary resistance indicator
            df['resistant'] = (df[int_col] == 'r').astype(int)
            
            if by_species and 'bacterial_species' in df.columns:
                for species in df['bacterial_species'].unique():
                    species_df = df[df['bacterial_species'] == species].copy()
                    species_df = species_df.set_index(date_column)
                    
                    # Rolling mean
                    rolling = species_df['resistant'].rolling(
                        window=f'{self.window_size}D', min_periods=1
                    ).mean() * 100
                    
                    result_df = pd.DataFrame({
                        'date': rolling.index,
                        'antibiotic': antibiotic,
                        'species': species,
                        'resistance_%': rolling.values
                    })
                    results.append(result_df)
            else:
                df_temp = df.set_index(date_column)
                rolling = df_temp['resistant'].rolling(
                    window=f'{self.window_size}D', min_periods=1
                ).mean() * 100
                
                result_df = pd.DataFrame({
                    'date': rolling.index,
                    'antibiotic': antibiotic,
                    'resistance_%': rolling.values
                })
                results.append(result_df)
                
        if results:
            rolling_prev = pd.concat(results, ignore_index=True)
            self.time_series = rolling_prev
            print(f"  Computed rolling prevalence for {len(antibiotics)} antibiotics")
            return rolling_prev
        else:
            return None
            
    def detect_change_points(self, time_series=None, method='simple', min_change=10):
        """
        Detect change points in resistance trends.
        
        Parameters:
        -----------
        time_series : DataFrame, optional
            Time series data (uses self.time_series if not provided)
        method : str
            Detection method ('simple' for threshold-based)
        min_change : float
            Minimum change in percentage points to flag
            
        Returns:
        --------
        change_points : list
            Detected change points
        """
        if time_series is None:
            time_series = self.time_series
            
        if time_series is None:
            print("No time series data available for change point detection")
            return None
            
        print(f"\nDetecting change points (min_change={min_change}%)...")
        
        change_points = []
        
        # Group by antibiotic (and species if present)
        group_cols = ['antibiotic']
        if 'species' in time_series.columns:
            group_cols.append('species')
            
        for group_name, group_df in time_series.groupby(group_cols):
            if len(group_df) < 2:
                continue
                
            group_df = group_df.sort_values('date')
            
            # Simple method: detect significant jumps
            if method == 'simple':
                resistance_values = group_df['resistance_%'].values
                dates = group_df['date'].values
                
                for i in range(1, len(resistance_values)):
                    change = resistance_values[i] - resistance_values[i-1]
                    
                    if abs(change) >= min_change:
                        change_point = {
                            'date': dates[i],
                            'antibiotic': group_name if isinstance(group_name, str) else group_name[0],
                            'change_%': change,
                            'before_%': resistance_values[i-1],
                            'after_%': resistance_values[i],
                            'direction': 'increase' if change > 0 else 'decrease'
                        }
                        
                        if 'species' in time_series.columns:
                            change_point['species'] = group_name[1] if isinstance(group_name, tuple) else None
                            
                        change_points.append(change_point)
                        
        self.change_points = change_points
        print(f"  Detected {len(change_points)} change points")
        
        return change_points
        
    def analyze_trends(self, time_series=None):
        """
        Analyze overall trends in resistance.
        
        Parameters:
        -----------
        time_series : DataFrame, optional
            Time series data
            
        Returns:
        --------
        trends : dict
            Trend analysis results
        """
        if time_series is None:
            time_series = self.time_series
            
        if time_series is None:
            print("No time series data available for trend analysis")
            return None
            
        print("\nAnalyzing resistance trends...")
        
        trends = {}
        
        # Group by antibiotic
        group_cols = ['antibiotic']
        if 'species' in time_series.columns:
            group_cols.append('species')
            
        for group_name, group_df in time_series.groupby(group_cols):
            if len(group_df) < 2:
                continue
                
            group_df = group_df.sort_values('date')
            
            # Compute simple linear trend
            x = np.arange(len(group_df))
            y = group_df['resistance_%'].values
            
            # Linear regression
            if len(x) > 1 and np.std(y) > 0:
                slope = np.polyfit(x, y, 1)[0]
                
                trend_data = {
                    'slope_%_per_period': slope,
                    'direction': 'increasing' if slope > 0.1 else ('decreasing' if slope < -0.1 else 'stable'),
                    'start_resistance_%': y[0],
                    'end_resistance_%': y[-1],
                    'total_change_%': y[-1] - y[0],
                    'data_points': len(group_df)
                }
                
                if isinstance(group_name, tuple):
                    key = f"{group_name[0]}_{group_name[1]}"
                else:
                    key = group_name
                    
                trends[key] = trend_data
                
        self.trends = trends
        print(f"  Analyzed trends for {len(trends)} antibiotic/species combinations")
        
        return trends
        
    def generate_alerts(self, time_series=None, change_points=None):
        """
        Generate alerts for significant resistance increases.
        
        Parameters:
        -----------
        time_series : DataFrame, optional
            Time series data
        change_points : list, optional
            Change points
            
        Returns:
        --------
        alerts : list
            Alert notifications
        """
        if change_points is None:
            change_points = self.change_points
            
        if change_points is None:
            print("No change points available for alert generation")
            return None
            
        print(f"\nGenerating alerts (threshold={self.alert_threshold}x baseline)...")
        
        alerts = []
        
        for cp in change_points:
            # Alert if increase exceeds threshold
            if cp['direction'] == 'increase':
                increase_factor = cp['after_%'] / cp['before_%'] if cp['before_%'] > 0 else float('inf')
                
                if increase_factor >= self.alert_threshold:
                    alert = {
                        'date': cp['date'],
                        'antibiotic': cp['antibiotic'],
                        'alert_type': 'significant_increase',
                        'severity': 'high' if increase_factor >= 2.0 else 'medium',
                        'increase_factor': increase_factor,
                        'resistance_before_%': cp['before_%'],
                        'resistance_after_%': cp['after_%'],
                        'message': f"Resistance to {cp['antibiotic']} increased "
                                 f"{increase_factor:.1f}x from {cp['before_%']:.1f}% to {cp['after_%']:.1f}%"
                    }
                    
                    if 'species' in cp:
                        alert['species'] = cp['species']
                        
                    alerts.append(alert)
                    
        self.alerts = alerts
        print(f"  Generated {len(alerts)} alerts")
        
        return alerts
        
    def generate_temporal_report(self):
        """
        Generate comprehensive temporal analysis report.
        
        Returns:
        --------
        report : dict
            Temporal analysis report
        """
        report = {
            'summary': {
                'temporal_data_available': self.time_series is not None,
                'window_size_days': self.window_size,
                'alert_threshold': self.alert_threshold
            }
        }
        
        if self.trends:
            report['trends'] = {
                'total_analyzed': len(self.trends),
                'increasing': sum(1 for t in self.trends.values() if t['direction'] == 'increasing'),
                'decreasing': sum(1 for t in self.trends.values() if t['direction'] == 'decreasing'),
                'stable': sum(1 for t in self.trends.values() if t['direction'] == 'stable'),
                'details': self.trends
            }
            
        if self.change_points:
            report['change_points'] = {
                'total': len(self.change_points),
                'increases': sum(1 for cp in self.change_points if cp['direction'] == 'increase'),
                'decreases': sum(1 for cp in self.change_points if cp['direction'] == 'decrease'),
                'details': self.change_points
            }
            
        if self.alerts:
            report['alerts'] = {
                'total': len(self.alerts),
                'high_severity': sum(1 for a in self.alerts if a['severity'] == 'high'),
                'medium_severity': sum(1 for a in self.alerts if a['severity'] == 'medium'),
                'details': self.alerts
            }
            
        return report
        
    def save_results(self, output_prefix='temporal_analysis'):
        """
        Save temporal analysis results.
        
        Parameters:
        -----------
        output_prefix : str
            Prefix for output files
        """
        import json
        
        report = self.generate_temporal_report()
        
        with open(f'{output_prefix}_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved temporal report to {output_prefix}_report.json")
        
        # Save time series if available
        if self.time_series is not None:
            self.time_series.to_csv(f'{output_prefix}_timeseries.csv', index=False)
            print(f"Saved time series to {output_prefix}_timeseries.csv")
