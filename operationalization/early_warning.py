"""
Early Warning Alert System

Monitors resistance trends and generates alerts for significant increases
or emergence of concerning patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import joblib


class EarlyWarningSystem:
    """
    Early warning system for detecting concerning resistance trends.
    
    Monitors surveillance data and generates alerts based on:
    - Significant increases in resistance rates
    - Emergence of new resistance patterns
    - Anomalous isolates
    - Threshold breaches
    """
    
    def __init__(
        self,
        alert_threshold: float = 0.2,  # 20% increase
        significance_level: float = 0.05,
        output_dir: str = 'alerts'
    ):
        """
        Initialize early warning system.
        
        Parameters
        ----------
        alert_threshold : float
            Relative increase threshold for triggering alerts (e.g., 0.2 = 20%)
        significance_level : float
            Statistical significance level for trend detection
        output_dir : str
            Directory for saving alert logs
        """
        self.alert_threshold = alert_threshold
        self.significance_level = significance_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.alert_history = []
        
    def check_resistance_increase(
        self,
        df: pd.DataFrame,
        antibiotic: str,
        window_size: int = 30,
        species: Optional[str] = None
    ) -> Dict:
        """
        Check for significant increase in resistance rate.
        
        Parameters
        ----------
        df : pd.DataFrame
            Surveillance data with temporal ordering
        antibiotic : str
            Antibiotic column name (e.g., 'ciprofloxacin_sir')
        window_size : int
            Number of recent isolates to compare against baseline
        species : str, optional
            Filter by species
            
        Returns
        -------
        dict
            Alert information if threshold exceeded
        """
        # Filter by species if specified
        if species and 'bacterial_species' in df.columns:
            df = df[df['bacterial_species'] == species]
        
        # Get resistance column
        if antibiotic not in df.columns:
            return {}
        
        # Calculate baseline resistance (excluding recent window)
        total_isolates = len(df)
        if total_isolates < window_size * 2:
            return {}  # Insufficient data
        
        baseline_data = df.iloc[:-window_size][antibiotic]
        recent_data = df.iloc[-window_size:][antibiotic]
        
        # Calculate resistance rates
        baseline_resistant = (baseline_data == 'R').sum()
        baseline_total = baseline_data.notna().sum()
        baseline_rate = baseline_resistant / baseline_total if baseline_total > 0 else 0
        
        recent_resistant = (recent_data == 'R').sum()
        recent_total = recent_data.notna().sum()
        recent_rate = recent_resistant / recent_total if recent_total > 0 else 0
        
        # Check for significant increase
        relative_change = (recent_rate - baseline_rate) / baseline_rate if baseline_rate > 0 else 0
        
        alert = {}
        if relative_change >= self.alert_threshold:
            alert = {
                'type': 'resistance_increase',
                'antibiotic': antibiotic.replace('_sir', ''),
                'species': species or 'all',
                'baseline_rate': baseline_rate,
                'recent_rate': recent_rate,
                'relative_change': relative_change,
                'absolute_change': recent_rate - baseline_rate,
                'window_size': window_size,
                'severity': 'high' if relative_change >= 0.5 else 'medium',
                'timestamp': datetime.now().isoformat()
            }
            
            self.alert_history.append(alert)
        
        return alert
    
    def check_threshold_breach(
        self,
        df: pd.DataFrame,
        antibiotic: str,
        threshold: float = 0.3,  # 30% resistance
        species: Optional[str] = None
    ) -> Dict:
        """
        Check if resistance rate exceeds critical threshold.
        
        Parameters
        ----------
        df : pd.DataFrame
            Surveillance data
        antibiotic : str
            Antibiotic column name
        threshold : float
            Critical resistance threshold (0-1)
        species : str, optional
            Filter by species
            
        Returns
        -------
        dict
            Alert if threshold breached
        """
        # Filter by species
        if species and 'bacterial_species' in df.columns:
            df = df[df['bacterial_species'] == species]
        
        if antibiotic not in df.columns:
            return {}
        
        # Calculate current resistance rate
        data = df[antibiotic]
        resistant = (data == 'R').sum()
        total = data.notna().sum()
        resistance_rate = resistant / total if total > 0 else 0
        
        alert = {}
        if resistance_rate >= threshold:
            alert = {
                'type': 'threshold_breach',
                'antibiotic': antibiotic.replace('_sir', ''),
                'species': species or 'all',
                'resistance_rate': resistance_rate,
                'threshold': threshold,
                'n_resistant': resistant,
                'n_total': total,
                'severity': 'critical' if resistance_rate >= 0.5 else 'high',
                'timestamp': datetime.now().isoformat()
            }
            
            self.alert_history.append(alert)
        
        return alert
    
    def check_anomalous_isolate(
        self,
        anomaly_scores: pd.Series,
        threshold: float = 0.8
    ) -> List[Dict]:
        """
        Generate alerts for highly anomalous isolates.
        
        Parameters
        ----------
        anomaly_scores : pd.Series
            Anomaly scores per isolate (from Phase 4)
        threshold : float
            Anomaly score threshold for alerting
            
        Returns
        -------
        list
            List of alerts for anomalous isolates
        """
        alerts = []
        
        # Find isolates exceeding threshold
        anomalous = anomaly_scores[anomaly_scores >= threshold]
        
        for isolate_id, score in anomalous.items():
            alert = {
                'type': 'anomalous_isolate',
                'isolate_id': isolate_id,
                'anomaly_score': score,
                'severity': 'critical' if score >= 0.9 else 'high',
                'action': 'immediate_review',
                'timestamp': datetime.now().isoformat()
            }
            
            alerts.append(alert)
            self.alert_history.append(alert)
        
        return alerts
    
    def check_new_resistance_pattern(
        self,
        df: pd.DataFrame,
        antibiotic_columns: List[str],
        min_prevalence: int = 3
    ) -> List[Dict]:
        """
        Detect emergence of new multi-drug resistance patterns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Surveillance data
        antibiotic_columns : list
            Antibiotic S/I/R columns to analyze
        min_prevalence : int
            Minimum number of isolates with pattern to alert
            
        Returns
        -------
        list
            Alerts for new resistance patterns
        """
        alerts = []
        
        # Create binary resistance matrix
        resistance_matrix = pd.DataFrame()
        for col in antibiotic_columns:
            if col in df.columns:
                resistance_matrix[col] = (df[col] == 'R').astype(int)
        
        if resistance_matrix.empty:
            return alerts
        
        # Find patterns with high resistance count
        pattern_counts = resistance_matrix.value_counts()
        
        # Identify concerning patterns (resistant to many drugs)
        for pattern, count in pattern_counts.items():
            n_resistant = sum(pattern)
            
            # Alert if pattern shows resistance to ≥50% of tested antibiotics
            if n_resistant >= len(antibiotic_columns) * 0.5 and count >= min_prevalence:
                resistant_drugs = [antibiotic_columns[i].replace('_sir', '') 
                                 for i, val in enumerate(pattern) if val == 1]
                
                alert = {
                    'type': 'new_mdr_pattern',
                    'n_resistant_drugs': n_resistant,
                    'resistant_to': resistant_drugs,
                    'prevalence': count,
                    'severity': 'high' if n_resistant >= len(antibiotic_columns) * 0.75 else 'medium',
                    'timestamp': datetime.now().isoformat()
                }
                
                alerts.append(alert)
                self.alert_history.append(alert)
        
        return alerts
    
    def run_surveillance_check(
        self,
        df: pd.DataFrame,
        anomaly_scores: Optional[pd.Series] = None,
        antibiotic_columns: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Run complete surveillance check across all alert types.
        
        Parameters
        ----------
        df : pd.DataFrame
            Current surveillance data
        anomaly_scores : pd.Series, optional
            Anomaly scores from Phase 4
        antibiotic_columns : list, optional
            Antibiotics to monitor
            
        Returns
        -------
        dict
            All alerts generated, grouped by type
        """
        all_alerts = {
            'resistance_increase': [],
            'threshold_breach': [],
            'anomalous_isolates': [],
            'new_patterns': []
        }
        
        # Auto-detect antibiotic columns
        if antibiotic_columns is None:
            antibiotic_columns = [col for col in df.columns if col.endswith('_sir')]
        
        # Check for resistance increases
        for antibiotic in antibiotic_columns:
            alert = self.check_resistance_increase(df, antibiotic)
            if alert:
                all_alerts['resistance_increase'].append(alert)
            
            # Check threshold breaches
            alert = self.check_threshold_breach(df, antibiotic)
            if alert:
                all_alerts['threshold_breach'].append(alert)
        
        # Check anomalous isolates
        if anomaly_scores is not None:
            alerts = self.check_anomalous_isolate(anomaly_scores)
            all_alerts['anomalous_isolates'].extend(alerts)
        
        # Check for new patterns
        alerts = self.check_new_resistance_pattern(df, antibiotic_columns)
        all_alerts['new_patterns'].extend(alerts)
        
        # Save alerts
        self.save_alerts(all_alerts)
        
        return all_alerts
    
    def save_alerts(self, alerts: Dict[str, List[Dict]]):
        """
        Save alerts to file.
        
        Parameters
        ----------
        alerts : dict
            Alert dictionary from run_surveillance_check()
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        alert_file = self.output_dir / f"alerts_{timestamp}.json"
        
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        print(f"Alerts saved to: {alert_file}")
    
    def get_alert_summary(self) -> pd.DataFrame:
        """
        Get summary of recent alerts.
        
        Returns
        -------
        pd.DataFrame
            Summary table of alerts
        """
        if not self.alert_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.alert_history)
    
    def format_alert_message(self, alert: Dict) -> str:
        """
        Format alert as human-readable message.
        
        Parameters
        ----------
        alert : dict
            Alert dictionary
            
        Returns
        -------
        str
            Formatted alert message
        """
        alert_type = alert.get('type', 'unknown')
        severity = alert.get('severity', 'medium').upper()
        
        if alert_type == 'resistance_increase':
            msg = f"[{severity}] Resistance increase detected\n"
            msg += f"Antibiotic: {alert['antibiotic']}\n"
            msg += f"Species: {alert['species']}\n"
            msg += f"Baseline rate: {alert['baseline_rate']:.1%}\n"
            msg += f"Recent rate: {alert['recent_rate']:.1%}\n"
            msg += f"Change: +{alert['relative_change']:.1%}\n"
            
        elif alert_type == 'threshold_breach':
            msg = f"[{severity}] Resistance threshold breached\n"
            msg += f"Antibiotic: {alert['antibiotic']}\n"
            msg += f"Species: {alert['species']}\n"
            msg += f"Current rate: {alert['resistance_rate']:.1%}\n"
            msg += f"Threshold: {alert['threshold']:.1%}\n"
            
        elif alert_type == 'anomalous_isolate':
            msg = f"[{severity}] Anomalous isolate detected\n"
            msg += f"Isolate ID: {alert['isolate_id']}\n"
            msg += f"Anomaly score: {alert['anomaly_score']:.2f}\n"
            msg += f"Action required: {alert['action']}\n"
            
        elif alert_type == 'new_mdr_pattern':
            msg = f"[{severity}] New MDR pattern detected\n"
            msg += f"Resistant to {alert['n_resistant_drugs']} drugs\n"
            msg += f"Drugs: {', '.join(alert['resistant_to'])}\n"
            msg += f"Prevalence: {alert['prevalence']} isolates\n"
            
        else:
            msg = f"[{severity}] Alert: {alert_type}\n"
            msg += str(alert)
        
        return msg
