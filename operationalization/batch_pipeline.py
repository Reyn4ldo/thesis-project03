"""
Batch Scoring Pipeline

Automated batch scoring pipeline for weekly surveillance reports.
Can be scheduled with cron or orchestrated with Airflow.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import json
from typing import Dict, List, Optional
import argparse

from operationalization import (
    AntibiogramGenerator,
    EarlyWarningSystem,
    EmpiricTherapyRecommender
)


class BatchScoringPipeline:
    """
    Automated batch scoring and reporting pipeline.
    
    Runs weekly surveillance analysis including:
    - Model predictions on new isolates
    - Antibiogram generation
    - Early warning alerts
    - Therapy recommendations
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        model_dir: str = 'models',
        output_dir: str = 'batch_reports'
    ):
        """
        Initialize batch pipeline.
        
        Parameters
        ----------
        data_dir : str
            Directory containing surveillance data
        model_dir : str
            Directory containing trained models
        output_dir : str
            Directory for output reports
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.preprocessing_pipeline = None
        
    def load_models(self):
        """Load trained models."""
        print("Loading models...")
        
        # Load preprocessing pipeline
        pipeline_path = self.model_dir / 'preprocessing_pipeline.pkl'
        if pipeline_path.exists():
            self.preprocessing_pipeline = joblib.load(pipeline_path)
            print(f"Loaded preprocessing pipeline")
        
        # Load prediction models
        for model_file in self.model_dir.glob('*.pkl'):
            if model_file.name != 'preprocessing_pipeline.pkl':
                model_name = model_file.stem
                self.models[model_name] = joblib.load(model_file)
                print(f"Loaded model: {model_name}")
    
    def load_new_data(self, date_from: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load new surveillance data.
        
        Parameters
        ----------
        date_from : datetime, optional
            Load data from this date onwards (default: last 7 days)
            
        Returns
        -------
        pd.DataFrame
            New surveillance data
        """
        # Default to last 7 days
        if date_from is None:
            date_from = datetime.now() - timedelta(days=7)
        
        # In production, load from database
        # For now, load from CSV
        data_file = self.data_dir / 'sample_data.csv'
        
        if data_file.exists():
            df = pd.read_csv(data_file)
            print(f"Loaded {len(df)} isolates from {data_file}")
            return df
        else:
            print(f"No data file found at {data_file}")
            return pd.DataFrame()
    
    def run_predictions(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Run all model predictions on new data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
            
        Returns
        -------
        dict
            Predictions from all models
        """
        predictions = {}
        
        # Preprocess data if pipeline available
        if self.preprocessing_pipeline:
            try:
                df_processed = self.preprocessing_pipeline.transform(df)
            except Exception as e:
                print(f"Warning: Preprocessing failed: {e}. Using raw data.")
                df_processed = df
        else:
            print("Warning: No preprocessing pipeline available. Using raw data.")
            df_processed = df
        
        # Run predictions for each model
        for model_name, model in self.models.items():
            try:
                pred = model.predict(df_processed)
                predictions[model_name] = pd.Series(pred, index=df.index)
                print(f"Generated predictions for {model_name}")
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
        
        return predictions
    
    def generate_antibiograms(self, df: pd.DataFrame) -> Dict:
        """
        Generate antibiograms for all species/sites.
        
        Parameters
        ----------
        df : pd.DataFrame
            Surveillance data
            
        Returns
        -------
        dict
            Antibiogram reports
        """
        print("Generating antibiograms...")
        
        generator = AntibiogramGenerator(output_dir=str(self.output_dir / 'antibiograms'))
        report = generator.generate_report(df)
        
        print(f"Generated {len(report)} antibiogram reports")
        return report
    
    def run_early_warning(
        self,
        df: pd.DataFrame,
        anomaly_scores: Optional[pd.Series] = None
    ) -> Dict:
        """
        Run early warning surveillance checks.
        
        Parameters
        ----------
        df : pd.DataFrame
            Surveillance data
        anomaly_scores : pd.Series, optional
            Anomaly scores from anomaly detection
            
        Returns
        -------
        dict
            Alerts generated
        """
        print("Running early warning checks...")
        
        ews = EarlyWarningSystem(output_dir=str(self.output_dir / 'alerts'))
        alerts = ews.run_surveillance_check(df, anomaly_scores=anomaly_scores)
        
        total_alerts = sum(len(v) for v in alerts.values())
        print(f"Generated {total_alerts} alerts")
        
        return alerts
    
    def generate_therapy_recommendations(self, df: pd.DataFrame) -> Dict:
        """
        Generate therapy recommendations for common scenarios.
        
        Parameters
        ----------
        df : pd.DataFrame
            Surveillance data
            
        Returns
        -------
        dict
            Therapy recommendations
        """
        print("Generating therapy recommendations...")
        
        recommender = EmpiricTherapyRecommender()
        
        # Generate recommendations for common scenarios
        scenarios = [
            {'species': None, 'source': 'urine'},
            {'species': None, 'source': 'blood'},
            {'species': 'Escherichia coli', 'source': None},
            {'species': 'Klebsiella pneumoniae', 'source': None}
        ]
        
        recommendations = {}
        for scenario in scenarios:
            key = f"{scenario.get('species', 'all')}_{scenario.get('source', 'all')}"
            report = recommender.generate_therapy_report(df, scenario)
            recommendations[key] = report
        
        print(f"Generated {len(recommendations)} therapy recommendation reports")
        return recommendations
    
    def generate_summary_report(
        self,
        df: pd.DataFrame,
        predictions: Dict,
        antibiograms: Dict,
        alerts: Dict,
        recommendations: Dict
    ) -> Dict:
        """
        Generate comprehensive summary report.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
        predictions : dict
            Model predictions
        antibiograms : dict
            Antibiogram reports
        alerts : dict
            Early warning alerts
        recommendations : dict
            Therapy recommendations
            
        Returns
        -------
        dict
            Summary report
        """
        summary = {
            'report_date': datetime.now().isoformat(),
            'data_summary': {
                'n_isolates': len(df),
                'date_range': 'last_7_days',
                'species_distribution': df.get('bacterial_species', pd.Series()).value_counts().to_dict() if 'bacterial_species' in df.columns else {}
            },
            'predictions': {
                model: pred.value_counts().to_dict()
                for model, pred in predictions.items()
            },
            'antibiograms_generated': len(antibiograms),
            'alerts': {
                'total': sum(len(v) for v in alerts.values()),
                'by_type': {k: len(v) for k, v in alerts.items()},
                'critical_count': sum(1 for alert_list in alerts.values() 
                                    for alert in alert_list 
                                    if alert.get('severity') == 'critical')
            },
            'recommendations_generated': len(recommendations)
        }
        
        return summary
    
    def run_pipeline(self, date_from: Optional[datetime] = None):
        """
        Run complete batch scoring pipeline.
        
        Parameters
        ----------
        date_from : datetime, optional
            Process data from this date onwards
        """
        print(f"\n{'='*60}")
        print(f"Batch Scoring Pipeline - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Load models
        self.load_models()
        
        # Load new data
        df = self.load_new_data(date_from)
        
        if df.empty:
            print("No data to process. Exiting.")
            return
        
        # Run predictions
        predictions = self.run_predictions(df)
        
        # Generate antibiograms
        antibiograms = self.generate_antibiograms(df)
        
        # Run early warning
        alerts = self.run_early_warning(df)
        
        # Generate therapy recommendations
        recommendations = self.generate_therapy_recommendations(df)
        
        # Generate summary report
        summary = self.generate_summary_report(
            df, predictions, antibiograms, alerts, recommendations
        )
        
        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.output_dir / f'weekly_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Pipeline completed successfully")
        print(f"Summary report: {summary_file}")
        print(f"{'='*60}\n")
        
        return summary


def main():
    """Main entry point for batch pipeline."""
    parser = argparse.ArgumentParser(description='Run batch scoring pipeline')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Model directory')
    parser.add_argument('--output-dir', type=str, default='batch_reports',
                       help='Output directory')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days of data to process')
    
    args = parser.parse_args()
    
    # Calculate date_from
    date_from = datetime.now() - timedelta(days=args.days)
    
    # Run pipeline
    pipeline = BatchScoringPipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    pipeline.run_pipeline(date_from=date_from)


if __name__ == '__main__':
    main()
