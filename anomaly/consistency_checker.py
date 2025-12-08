"""
Rule-based consistency checker for MIC vs interpretive call validation.

Detects inconsistencies between MIC values and S/I/R interpretations
using CLSI/EUCAST breakpoint rules.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class ConsistencyChecker:
    """
    Rule-based consistency checking for MIC vs S/I/R interpretations.
    
    Identifies isolates with inconsistent or suspicious resistance patterns.
    """
    
    def __init__(self):
        """Initialize consistency checker."""
        self.inconsistencies = []
        self.suspicious_patterns = []
    
    def check_mic_sir_consistency(self, df):
        """
        Check for inconsistencies between MIC values and S/I/R calls.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with MIC and S/I/R columns
            
        Returns
        -------
        pd.DataFrame
            Inconsistency report with flagged samples
        """
        print("\nChecking MIC vs S/I/R consistency...")
        
        # Get MIC and SIR columns
        mic_cols = [col for col in df.columns if '_mic' in col.lower() 
                   and not col.endswith('_imputed')]
        sir_cols = [col for col in df.columns if '_sir' in col.lower() 
                   and not col.endswith('_imputed')]
        
        inconsistencies = []
        
        for mic_col in mic_cols:
            # Find corresponding SIR column
            antibiotic = mic_col.replace('_mic', '').replace('_MIC', '')
            
            # Try different naming conventions
            possible_sir_cols = [
                f"{antibiotic}_sir",
                f"{antibiotic}_SIR",
                f"{antibiotic}_interpretation"
            ]
            
            sir_col = None
            for col in possible_sir_cols:
                if col in df.columns:
                    sir_col = col
                    break
            
            if sir_col is None:
                continue
            
            # Check each row for inconsistency
            for idx, row in df.iterrows():
                mic_val = row[mic_col]
                sir_val = row[sir_col]
                
                # Skip if either is missing
                if pd.isna(mic_val) or pd.isna(sir_val):
                    continue
                
                # Parse MIC value
                mic_numeric = self._parse_mic_value(mic_val)
                
                if mic_numeric is None:
                    continue
                
                # Check for obvious inconsistencies
                # These are heuristics - actual breakpoints vary by antibiotic
                inconsistent = False
                reason = ""
                
                # Very high MIC should not be S
                if mic_numeric >= 32 and sir_val in ['S', 's', 'sensitive']:
                    inconsistent = True
                    reason = f"High MIC ({mic_val}) marked as Sensitive"
                
                # Very low MIC should not be R
                elif mic_numeric <= 1 and sir_val in ['R', 'r', 'resistant']:
                    inconsistent = True
                    reason = f"Low MIC ({mic_val}) marked as Resistant"
                
                if inconsistent:
                    inconsistencies.append({
                        'isolate_id': idx,
                        'antibiotic': antibiotic,
                        'mic_value': mic_val,
                        'sir_value': sir_val,
                        'reason': reason
                    })
        
        inconsistency_df = pd.DataFrame(inconsistencies)
        
        print(f"  Found {len(inconsistencies)} potential MIC/SIR inconsistencies")
        
        self.inconsistencies = inconsistency_df
        
        return inconsistency_df
    
    def check_impossible_patterns(self, df):
        """
        Check for biologically impossible resistance patterns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with resistance data
            
        Returns
        -------
        list
            List of suspicious patterns found
        """
        print("\nChecking for impossible resistance patterns...")
        
        suspicious = []
        
        # Get resistance columns
        resistance_cols = [col for col in df.columns if '_resistant' in col]
        
        # Check for all resistant (very rare)
        for idx, row in df.iterrows():
            resistance_values = row[resistance_cols]
            n_resistant = resistance_values.sum()
            n_tested = (resistance_values != -1).sum()  # Exclude not tested
            
            if n_tested > 10 and n_resistant == n_tested:
                suspicious.append({
                    'isolate_id': idx,
                    'pattern': 'all_resistant',
                    'description': f'Resistant to all {n_tested} antibiotics tested',
                    'severity': 'high'
                })
            
            # Check for no resistance (unusual in surveillance data)
            elif n_tested > 10 and n_resistant == 0:
                suspicious.append({
                    'isolate_id': idx,
                    'pattern': 'all_susceptible',
                    'description': f'Susceptible to all {n_tested} antibiotics tested',
                    'severity': 'low'
                })
        
        print(f"  Found {len(suspicious)} suspicious patterns")
        
        self.suspicious_patterns = suspicious
        
        return suspicious
    
    def check_mar_consistency(self, df):
        """
        Check MAR index consistency with resistance profile.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with MAR index and resistance columns
            
        Returns
        -------
        list
            List of inconsistencies
        """
        print("\nChecking MAR index consistency...")
        
        if 'mar_index' not in df.columns:
            print("  MAR index not found, skipping...")
            return []
        
        inconsistencies = []
        
        resistance_cols = [col for col in df.columns if '_resistant' in col 
                          and 'total' not in col]
        
        for idx, row in df.iterrows():
            mar_index = row['mar_index']
            
            if pd.isna(mar_index):
                continue
            
            # Calculate actual resistance rate
            resistance_values = row[resistance_cols]
            n_tested = (resistance_values != -1).sum()
            n_resistant = (resistance_values == 1).sum()
            
            if n_tested > 0:
                actual_rate = n_resistant / n_tested
                
                # Check if MAR index is very different from actual rate
                if abs(mar_index - actual_rate) > 0.3:
                    inconsistencies.append({
                        'isolate_id': idx,
                        'mar_index': mar_index,
                        'actual_rate': actual_rate,
                        'difference': abs(mar_index - actual_rate),
                        'reason': 'MAR index does not match resistance profile'
                    })
        
        print(f"  Found {len(inconsistencies)} MAR inconsistencies")
        
        return inconsistencies
    
    def check_all(self, df):
        """
        Run all consistency checks.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to check
            
        Returns
        -------
        dict
            All consistency check results
        """
        results = {
            'mic_sir_inconsistencies': self.check_mic_sir_consistency(df),
            'impossible_patterns': self.check_impossible_patterns(df),
            'mar_inconsistencies': self.check_mar_consistency(df)
        }
        
        return results
    
    def get_flagged_isolates(self):
        """
        Get all isolates flagged by consistency checks.
        
        Returns
        -------
        set
            Set of isolate IDs with issues
        """
        flagged = set()
        
        if len(self.inconsistencies) > 0:
            flagged.update(self.inconsistencies['isolate_id'].unique())
        
        if len(self.suspicious_patterns) > 0:
            flagged.update([p['isolate_id'] for p in self.suspicious_patterns])
        
        return flagged
    
    def _parse_mic_value(self, mic_str):
        """
        Parse MIC value string to numeric.
        
        Parameters
        ----------
        mic_str : str or float
            MIC value (e.g., "<=0.5", ">32", "4")
            
        Returns
        -------
        float or None
            Numeric MIC value
        """
        if pd.isna(mic_str):
            return None
        
        # If already numeric
        if isinstance(mic_str, (int, float)):
            return float(mic_str)
        
        # Convert to string
        mic_str = str(mic_str).strip()
        
        # Remove operators
        mic_str = mic_str.replace('<=', '').replace('>=', '').replace('<', '').replace('>', '')
        mic_str = mic_str.strip()
        
        try:
            return float(mic_str)
        except ValueError:
            return None
    
    def generate_report(self):
        """
        Generate summary report of all consistency checks.
        
        Returns
        -------
        str
            Formatted report
        """
        report = []
        report.append("="*80)
        report.append("CONSISTENCY CHECK REPORT")
        report.append("="*80)
        report.append("")
        
        # MIC/SIR inconsistencies
        report.append(f"MIC vs S/I/R Inconsistencies: {len(self.inconsistencies)}")
        if len(self.inconsistencies) > 0:
            report.append(f"  Top antibiotics with issues:")
            antibiotic_counts = self.inconsistencies['antibiotic'].value_counts().head(5)
            for ab, count in antibiotic_counts.items():
                report.append(f"    - {ab}: {count} inconsistencies")
        report.append("")
        
        # Suspicious patterns
        report.append(f"Suspicious Patterns: {len(self.suspicious_patterns)}")
        if len(self.suspicious_patterns) > 0:
            pattern_counts = pd.Series([p['pattern'] for p in self.suspicious_patterns]).value_counts()
            for pattern, count in pattern_counts.items():
                report.append(f"  - {pattern}: {count} isolates")
        report.append("")
        
        # Total flagged
        flagged = self.get_flagged_isolates()
        report.append(f"Total Unique Isolates Flagged: {len(flagged)}")
        report.append("")
        
        return "\n".join(report)
