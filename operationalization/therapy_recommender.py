"""
Empiric Therapy Recommender

Provides evidence-based antibiotic recommendations based on local resistance
patterns and patient-specific factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json


class EmpiricTherapyRecommender:
    """
    Empiric therapy recommendation system.
    
    Recommends antibiotics for empiric therapy based on:
    - Local resistance patterns (antibiogram data)
    - Predicted species (if available)
    - Infection site/source
    - Patient-specific contraindications
    """
    
    def __init__(
        self,
        min_susceptibility: float = 0.80,  # 80% susceptibility threshold
        confidence_threshold: float = 0.70
    ):
        """
        Initialize therapy recommender.
        
        Parameters
        ----------
        min_susceptibility : float
            Minimum susceptibility rate to recommend antibiotic
        confidence_threshold : float
            Confidence threshold for recommendations
        """
        self.min_susceptibility = min_susceptibility
        self.confidence_threshold = confidence_threshold
        
    def calculate_susceptibility_probability(
        self,
        df: pd.DataFrame,
        antibiotic: str,
        species: Optional[str] = None,
        site: Optional[str] = None
    ) -> Tuple[float, int]:
        """
        Calculate probability of susceptibility for an antibiotic.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical surveillance data
        antibiotic : str
            Antibiotic column name
        species : str, optional
            Filter by species
        site : str, optional
            Filter by site/location
            
        Returns
        -------
        tuple
            (probability of susceptibility, number of isolates)
        """
        # Filter data
        filtered_df = df.copy()
        
        if species and 'bacterial_species' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['bacterial_species'] == species]
        
        if site:
            site_cols = [c for c in filtered_df.columns if 'site' in c.lower()]
            if site_cols:
                filtered_df = filtered_df[filtered_df[site_cols[0]] == site]
        
        # Calculate susceptibility
        if antibiotic not in filtered_df.columns:
            return 0.0, 0
        
        data = filtered_df[antibiotic]
        n_total = data.notna().sum()
        
        if n_total == 0:
            return 0.0, 0
        
        n_susceptible = (data == 'S').sum()
        probability = n_susceptible / n_total
        
        return probability, n_total
    
    def recommend_antibiotics(
        self,
        df: pd.DataFrame,
        species: Optional[str] = None,
        site: Optional[str] = None,
        source: Optional[str] = None,
        contraindications: Optional[List[str]] = None,
        antibiotic_columns: Optional[List[str]] = None,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Recommend antibiotics for empiric therapy.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical surveillance data
        species : str, optional
            Suspected bacterial species
        site : str, optional
            Collection site
        source : str, optional
            Sample source (blood, urine, etc.)
        contraindications : list, optional
            Antibiotics to exclude (patient allergies/restrictions)
        antibiotic_columns : list, optional
            Available antibiotics to consider
        top_n : int
            Number of top recommendations to return
            
        Returns
        -------
        list
            Ranked list of antibiotic recommendations with probabilities
        """
        # Auto-detect antibiotic columns
        if antibiotic_columns is None:
            antibiotic_columns = [col for col in df.columns if col.endswith('_sir')]
        
        # Calculate susceptibility for each antibiotic
        recommendations = []
        
        for antibiotic in antibiotic_columns:
            antibiotic_name = antibiotic.replace('_sir', '')
            
            # Skip if contraindicated
            if contraindications and antibiotic_name in contraindications:
                continue
            
            # Calculate susceptibility probability
            prob, n_isolates = self.calculate_susceptibility_probability(
                df, antibiotic, species=species, site=site
            )
            
            # Determine confidence level
            if n_isolates >= 100:
                confidence = 'high'
            elif n_isolates >= 30:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            # Only recommend if meets minimum threshold
            if prob >= self.min_susceptibility:
                recommendations.append({
                    'antibiotic': antibiotic_name,
                    'susceptibility_probability': prob,
                    'n_isolates': n_isolates,
                    'confidence': confidence,
                    'rank': None  # Will be assigned after sorting
                })
        
        # Sort by susceptibility probability
        recommendations.sort(key=lambda x: x['susceptibility_probability'], reverse=True)
        
        # Assign ranks
        for i, rec in enumerate(recommendations[:top_n]):
            rec['rank'] = i + 1
        
        return recommendations[:top_n]
    
    def get_alternative_therapies(
        self,
        df: pd.DataFrame,
        primary_antibiotic: str,
        species: Optional[str] = None,
        min_susceptibility_diff: float = 0.05
    ) -> List[Dict]:
        """
        Get alternative therapies if primary choice fails.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical surveillance data
        primary_antibiotic : str
            Primary antibiotic choice
        species : str, optional
            Bacterial species
        min_susceptibility_diff : float
            Minimum acceptable difference from primary
            
        Returns
        -------
        list
            Alternative antibiotic options
        """
        # Get all recommendations
        all_recommendations = self.recommend_antibiotics(
            df,
            species=species,
            top_n=10
        )
        
        # Find primary antibiotic susceptibility
        primary_susc = None
        for rec in all_recommendations:
            if rec['antibiotic'] == primary_antibiotic:
                primary_susc = rec['susceptibility_probability']
                break
        
        if primary_susc is None:
            return all_recommendations[:5]
        
        # Filter alternatives within acceptable range
        alternatives = []
        for rec in all_recommendations:
            if rec['antibiotic'] != primary_antibiotic:
                if rec['susceptibility_probability'] >= primary_susc - min_susceptibility_diff:
                    alternatives.append(rec)
        
        return alternatives[:5]
    
    def generate_therapy_report(
        self,
        df: pd.DataFrame,
        patient_info: Dict,
        antibiotic_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate comprehensive therapy recommendation report.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical surveillance data
        patient_info : dict
            Patient information:
            - species (optional): suspected species
            - site (optional): collection site
            - source (optional): sample source
            - contraindications (optional): list of excluded antibiotics
        antibiotic_columns : list, optional
            Available antibiotics
            
        Returns
        -------
        dict
            Comprehensive recommendation report
        """
        # Get primary recommendations
        recommendations = self.recommend_antibiotics(
            df,
            species=patient_info.get('species'),
            site=patient_info.get('site'),
            source=patient_info.get('source'),
            contraindications=patient_info.get('contraindications'),
            antibiotic_columns=antibiotic_columns,
            top_n=5
        )
        
        # Get alternatives for top recommendation
        alternatives = []
        if recommendations:
            primary = recommendations[0]['antibiotic']
            alternatives = self.get_alternative_therapies(
                df,
                primary,
                species=patient_info.get('species')
            )
        
        # Generate report
        report = {
            'patient_info': patient_info,
            'primary_recommendations': recommendations,
            'alternative_options': alternatives,
            'recommendation_basis': {
                'min_susceptibility_threshold': self.min_susceptibility,
                'data_source': 'local_antibiogram',
                'species_specific': patient_info.get('species') is not None,
                'site_specific': patient_info.get('site') is not None
            },
            'caveats': [
                'Recommendations based on local surveillance data',
                'Consider patient-specific factors (allergies, organ function)',
                'Adjust therapy based on culture results when available',
                'Follow institutional antibiotic stewardship guidelines'
            ]
        }
        
        return report
    
    def format_recommendation(self, recommendation: Dict) -> str:
        """
        Format recommendation as human-readable text.
        
        Parameters
        ----------
        recommendation : dict
            Recommendation dictionary
            
        Returns
        -------
        str
            Formatted recommendation
        """
        antibiotic = recommendation['antibiotic']
        prob = recommendation['susceptibility_probability']
        confidence = recommendation['confidence']
        n = recommendation['n_isolates']
        
        msg = f"{antibiotic.upper()}\n"
        msg += f"  Susceptibility probability: {prob:.1%}\n"
        msg += f"  Confidence: {confidence} (n={n} isolates)\n"
        
        return msg
    
    def export_report(self, report: Dict, output_path: str):
        """
        Export recommendation report to JSON.
        
        Parameters
        ----------
        report : dict
            Recommendation report
        output_path : str
            Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Recommendation report saved to: {output_path}")
