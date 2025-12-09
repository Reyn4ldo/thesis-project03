"""
Phase 6: Operationalization & Outputs

Main orchestration script for operational AMR surveillance system.
Demonstrates all components: antibiograms, early warning, therapy recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

from operationalization import (
    AntibiogramGenerator,
    EarlyWarningSystem,
    EmpiricTherapyRecommender
)


def load_data(data_file: str = 'sample_data.csv') -> pd.DataFrame:
    """Load surveillance data."""
    print(f"\nLoading data from {data_file}...")
    
    if not Path(data_file).exists():
        print(f"Warning: {data_file} not found. Using sample data.")
        # Return empty dataframe - in production would load from database
        return pd.DataFrame()
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} isolates")
    
    return df


def run_antibiogram_generation(df: pd.DataFrame, output_dir: str = 'antibiograms'):
    """Generate antibiograms for all species and sites."""
    print("\n" + "="*60)
    print("ANTIBIOGRAM GENERATION")
    print("="*60)
    
    generator = AntibiogramGenerator(output_dir=output_dir)
    
    # Generate comprehensive report
    report = generator.generate_report(df)
    
    print(f"\nGenerated {len(report)} antibiogram reports")
    print(f"Output directory: {output_dir}")
    
    return report


def run_early_warning_system(df: pd.DataFrame, output_dir: str = 'alerts'):
    """Run early warning surveillance checks."""
    print("\n" + "="*60)
    print("EARLY WARNING SYSTEM")
    print("="*60)
    
    ews = EarlyWarningSystem(output_dir=output_dir)
    
    # Run surveillance checks
    alerts = ews.run_surveillance_check(df)
    
    # Print summary
    total_alerts = sum(len(v) for v in alerts.values())
    print(f"\nTotal alerts generated: {total_alerts}")
    
    for alert_type, alert_list in alerts.items():
        if alert_list:
            print(f"\n{alert_type.upper()}: {len(alert_list)} alerts")
            for alert in alert_list[:3]:  # Show first 3
                print(f"  - {ews.format_alert_message(alert)[:200]}...")
    
    print(f"\nAlert logs saved to: {output_dir}")
    
    return alerts


def run_therapy_recommendations(df: pd.DataFrame, output_dir: str = 'recommendations'):
    """Generate empiric therapy recommendations."""
    print("\n" + "="*60)
    print("THERAPY RECOMMENDATIONS")
    print("="*60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    recommender = EmpiricTherapyRecommender()
    
    # Generate recommendations for common scenarios
    scenarios = [
        {
            'name': 'UTI_ecoli',
            'info': {'species': 'Escherichia coli', 'source': 'urine'}
        },
        {
            'name': 'Bacteremia_kpneumoniae',
            'info': {'species': 'Klebsiella pneumoniae', 'source': 'blood'}
        },
        {
            'name': 'General_UTI',
            'info': {'source': 'urine'}
        }
    ]
    
    recommendations = {}
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        
        report = recommender.generate_therapy_report(df, scenario['info'])
        recommendations[scenario['name']] = report
        
        # Print top recommendations
        if report['primary_recommendations']:
            print("  Top recommendations:")
            for rec in report['primary_recommendations'][:3]:
                print(f"    {rec['rank']}. {rec['antibiotic']}: "
                      f"{rec['susceptibility_probability']:.1%} susceptible "
                      f"({rec['confidence']} confidence)")
        
        # Save report
        report_file = Path(output_dir) / f"{scenario['name']}_recommendations.json"
        recommender.export_report(report, str(report_file))
    
    print(f"\nRecommendation reports saved to: {output_dir}")
    
    return recommendations


def generate_operational_dashboard(
    antibiograms: dict,
    alerts: dict,
    recommendations: dict,
    output_file: str = 'operational_dashboard.json'
):
    """Generate operational dashboard summary."""
    print("\n" + "="*60)
    print("OPERATIONAL DASHBOARD")
    print("="*60)
    
    dashboard = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'antibiograms_generated': len(antibiograms),
            'total_alerts': sum(len(v) for v in alerts.values()),
            'critical_alerts': sum(
                1 for alert_list in alerts.values()
                for alert in alert_list
                if alert.get('severity') == 'critical'
            ),
            'recommendations_generated': len(recommendations)
        },
        'alert_breakdown': {
            alert_type: len(alert_list)
            for alert_type, alert_list in alerts.items()
        },
        'status': 'operational'
    }
    
    # Save dashboard
    with open(output_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"\nDashboard Summary:")
    print(f"  - Antibiograms: {dashboard['summary']['antibiograms_generated']}")
    print(f"  - Total alerts: {dashboard['summary']['total_alerts']}")
    print(f"  - Critical alerts: {dashboard['summary']['critical_alerts']}")
    print(f"  - Recommendations: {dashboard['summary']['recommendations_generated']}")
    print(f"\nDashboard saved to: {output_file}")
    
    return dashboard


def main():
    """Main entry point for Phase 6 demonstration."""
    parser = argparse.ArgumentParser(
        description='Phase 6: Operationalization & Outputs'
    )
    parser.add_argument('--data', type=str, default='sample_data.csv',
                       help='Input data file')
    parser.add_argument('--output-dir', type=str, default='phase6_outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("PHASE 6: OPERATIONALIZATION & OUTPUTS")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = load_data(args.data)
    
    if df.empty:
        print("\nWarning: No data available. Demonstrating with empty dataset.")
        print("In production, this would connect to live surveillance database.")
    
    # Run all components
    antibiograms = run_antibiogram_generation(
        df, 
        output_dir=str(output_dir / 'antibiograms')
    )
    
    alerts = run_early_warning_system(
        df,
        output_dir=str(output_dir / 'alerts')
    )
    
    recommendations = run_therapy_recommendations(
        df,
        output_dir=str(output_dir / 'recommendations')
    )
    
    # Generate dashboard
    dashboard = generate_operational_dashboard(
        antibiograms,
        alerts,
        recommendations,
        output_file=str(output_dir / 'operational_dashboard.json')
    )
    
    print("\n" + "="*60)
    print("PHASE 6 COMPLETE")
    print("="*60)
    print("\nOperational components demonstrated:")
    print("  ✓ Automated antibiogram generation")
    print("  ✓ Early warning alert system")
    print("  ✓ Empiric therapy recommendations")
    print("  ✓ Operational dashboard")
    print("\nNext steps:")
    print("  1. Start REST API: uvicorn operationalization.api:app")
    print("  2. Run batch pipeline: python operationalization/batch_pipeline.py")
    print("  3. Deploy with Docker: docker-compose up")
    print("\nSee operationalization/README.md for detailed documentation")


if __name__ == '__main__':
    main()
