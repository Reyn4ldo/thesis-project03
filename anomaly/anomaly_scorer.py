"""
Anomaly scoring and triage system.

Combines multiple detection methods into unified anomaly scores
and provides triage rules for automated review/quarantine.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class AnomalyScorer:
    """
    Unified anomaly scoring system combining multiple detection methods.
    
    Produces composite anomaly scores and triage recommendations.
    """
    
    def __init__(self, weights=None):
        """
        Initialize anomaly scorer.
        
        Parameters
        ----------
        weights : dict, optional
            Weights for each detection method
            Default: {'isolation_forest': 0.3, 'lof': 0.3, 'mahalanobis': 0.2, 
                     'dbscan': 0.1, 'consistency': 0.1}
        """
        if weights is None:
            self.weights = {
                'isolation_forest': 0.3,
                'lof': 0.3,
                'mahalanobis': 0.2,
                'dbscan': 0.1,
                'consistency': 0.1
            }
        else:
            self.weights = weights
        
        self.anomaly_scores = None
        self.triage_labels = None
    
    def compute_composite_scores(self, outlier_results, consistency_results, df):
        """
        Compute composite anomaly scores from multiple sources.
        
        Parameters
        ----------
        outlier_results : dict
            Results from OutlierDetector
        consistency_results : dict
            Results from ConsistencyChecker
        df : pd.DataFrame
            Original dataframe for reference
            
        Returns
        -------
        pd.Series
            Composite anomaly scores for each isolate
        """
        print("\nComputing composite anomaly scores...")
        
        n_samples = len(df)
        scores = pd.Series(0.0, index=df.index, name='anomaly_score')
        
        # Normalize and combine outlier detection scores
        for method, weight in self.weights.items():
            if method == 'consistency':
                # Handle consistency separately
                continue
            
            if method not in outlier_results:
                print(f"  Warning: {method} not found in results, skipping...")
                continue
            
            method_scores = outlier_results[method]['scores']
            
            # Normalize to [0, 1] range
            min_score = method_scores.min()
            max_score = method_scores.max()
            
            if max_score > min_score:
                normalized = (method_scores - min_score) / (max_score - min_score)
            else:
                # All scores are identical - treat as no anomalies
                normalized = np.zeros_like(method_scores)
            
            # Add weighted contribution
            scores += weight * normalized
        
        # Add consistency check scores
        if consistency_results:
            consistency_score = np.zeros(n_samples)
            
            # Score MIC/SIR inconsistencies
            if 'mic_sir_inconsistencies' in consistency_results:
                inconsist_df = consistency_results['mic_sir_inconsistencies']
                if len(inconsist_df) > 0:
                    flagged_ids = inconsist_df['isolate_id'].unique()
                    consistency_score[flagged_ids] += 0.5
            
            # Score suspicious patterns
            if 'impossible_patterns' in consistency_results:
                patterns = consistency_results['impossible_patterns']
                for pattern in patterns:
                    isolate_id = pattern['isolate_id']
                    severity = pattern.get('severity', 'medium')
                    
                    if severity == 'high':
                        consistency_score[isolate_id] += 0.8
                    elif severity == 'medium':
                        consistency_score[isolate_id] += 0.5
                    else:
                        consistency_score[isolate_id] += 0.3
            
            # Add weighted consistency score
            scores += self.weights['consistency'] * np.minimum(consistency_score, 1.0)
        
        # Normalize final scores to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()
        
        self.anomaly_scores = scores
        
        print(f"  Composite scores computed")
        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"  Mean score: {scores.mean():.3f}")
        print(f"  Median score: {scores.median():.3f}")
        
        return scores
    
    def assign_triage_labels(self, scores, thresholds=None):
        """
        Assign triage labels based on anomaly scores.
        
        Parameters
        ----------
        scores : pd.Series
            Anomaly scores
        thresholds : dict, optional
            Thresholds for triage levels
            Default: {'quarantine': 0.8, 'review': 0.5, 'monitor': 0.3}
            
        Returns
        -------
        pd.Series
            Triage labels for each isolate
        """
        if thresholds is None:
            thresholds = {
                'quarantine': 0.8,   # Automatic quarantine
                'review': 0.5,        # Human review required
                'monitor': 0.3,       # Flag for monitoring
                'normal': 0.0         # No action
            }
        
        print("\nAssigning triage labels...")
        
        labels = pd.Series('normal', index=scores.index, name='triage_label')
        
        # Assign labels based on thresholds
        labels[scores >= thresholds['quarantine']] = 'quarantine'
        labels[(scores >= thresholds['review']) & (scores < thresholds['quarantine'])] = 'review'
        labels[(scores >= thresholds['monitor']) & (scores < thresholds['review'])] = 'monitor'
        
        self.triage_labels = labels
        
        # Print distribution
        label_counts = labels.value_counts()
        print(f"  Triage distribution:")
        for label in ['quarantine', 'review', 'monitor', 'normal']:
            count = label_counts.get(label, 0)
            pct = count / len(labels) * 100
            print(f"    {label}: {count} ({pct:.1f}%)")
        
        return labels
    
    def get_anomaly_report(self, df, scores, labels, top_n=20):
        """
        Generate detailed anomaly report.
        
        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe
        scores : pd.Series
            Anomaly scores
        labels : pd.Series
            Triage labels
        top_n : int, default=20
            Number of top anomalies to include
            
        Returns
        -------
        pd.DataFrame
            Detailed report of anomalies
        """
        # Create report dataframe
        report_df = pd.DataFrame({
            'anomaly_score': scores,
            'triage_label': labels
        })
        
        # Add key metadata columns if available
        metadata_cols = ['bacterial_species', 'administrative_region', 
                        'sample_source', 'esbl', 'mar_index']
        
        for col in metadata_cols:
            if col in df.columns:
                report_df[col] = df[col]
        
        # Sort by anomaly score
        report_df = report_df.sort_values('anomaly_score', ascending=False)
        
        return report_df
    
    def get_top_anomalies(self, n=20):
        """
        Get top N most anomalous samples.
        
        Parameters
        ----------
        n : int, default=20
            Number of top anomalies to return
            
        Returns
        -------
        pd.Series
            Top anomaly scores
        """
        if self.anomaly_scores is None:
            raise ValueError("Must compute scores first")
        
        return self.anomaly_scores.nlargest(n)
    
    def get_triage_summary(self):
        """
        Get summary statistics for triage categories.
        
        Returns
        -------
        dict
            Summary statistics per category
        """
        if self.triage_labels is None or self.anomaly_scores is None:
            raise ValueError("Must assign triage labels first")
        
        summary = {}
        
        for label in self.triage_labels.unique():
            mask = self.triage_labels == label
            summary[label] = {
                'count': mask.sum(),
                'percentage': mask.sum() / len(self.triage_labels) * 100,
                'mean_score': self.anomaly_scores[mask].mean(),
                'median_score': self.anomaly_scores[mask].median(),
                'score_range': [
                    self.anomaly_scores[mask].min(),
                    self.anomaly_scores[mask].max()
                ]
            }
        
        return summary
    
    def save_results(self, df, filepath='anomaly_results.csv'):
        """
        Save anomaly detection results to file.
        
        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe
        filepath : str
            Output file path
        """
        if self.anomaly_scores is None or self.triage_labels is None:
            raise ValueError("Must compute scores and labels first")
        
        # Create output dataframe
        output_df = df.copy()
        output_df['anomaly_score'] = self.anomaly_scores
        output_df['triage_label'] = self.triage_labels
        
        # Save to file
        output_df.to_csv(filepath, index=False)
        
        print(f"\nResults saved to {filepath}")
    
    def generate_summary_report(self):
        """
        Generate text summary report.
        
        Returns
        -------
        str
            Formatted summary report
        """
        if self.anomaly_scores is None or self.triage_labels is None:
            return "No results to report. Run compute_composite_scores and assign_triage_labels first."
        
        report = []
        report.append("="*80)
        report.append("ANOMALY DETECTION SUMMARY REPORT")
        report.append("="*80)
        report.append("")
        
        # Score statistics
        report.append("Anomaly Score Statistics:")
        report.append(f"  Total samples: {len(self.anomaly_scores)}")
        report.append(f"  Mean score: {self.anomaly_scores.mean():.3f}")
        report.append(f"  Median score: {self.anomaly_scores.median():.3f}")
        report.append(f"  Std dev: {self.anomaly_scores.std():.3f}")
        report.append(f"  Min score: {self.anomaly_scores.min():.3f}")
        report.append(f"  Max score: {self.anomaly_scores.max():.3f}")
        report.append("")
        
        # Triage distribution
        report.append("Triage Distribution:")
        label_counts = self.triage_labels.value_counts()
        for label in ['quarantine', 'review', 'monitor', 'normal']:
            count = label_counts.get(label, 0)
            pct = count / len(self.triage_labels) * 100
            report.append(f"  {label.capitalize()}: {count} ({pct:.1f}%)")
        report.append("")
        
        # Top anomalies
        top_5 = self.get_top_anomalies(5)
        report.append("Top 5 Most Anomalous Samples:")
        for idx, score in top_5.items():
            label = self.triage_labels[idx]
            report.append(f"  Sample {idx}: score={score:.3f}, triage={label}")
        report.append("")
        
        # Recommendations
        n_quarantine = (self.triage_labels == 'quarantine').sum()
        n_review = (self.triage_labels == 'review').sum()
        
        report.append("Recommendations:")
        report.append(f"  - {n_quarantine} samples require immediate quarantine/investigation")
        report.append(f"  - {n_review} samples require human expert review")
        report.append(f"  - Investigate top anomalies for data quality issues")
        report.append(f"  - Review consistency check results for specific issues")
        report.append("")
        
        return "\n".join(report)
