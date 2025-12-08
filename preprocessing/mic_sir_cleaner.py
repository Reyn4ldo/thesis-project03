"""
MIC and SIR data cleaning module.

Handles normalization of MIC values, standardization of S/I/R interpretations,
and detection of inconsistencies between MIC values and interpretations.
"""

import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
import warnings


class MICSIRCleaner(BaseEstimator, TransformerMixin):
    """
    Cleaner for MIC values and S/I/R interpretations.
    
    Features:
    - Normalizes MIC values (handles <=, >=, Unicode variants)
    - Standardizes S/I/R interpretations
    - Detects and flags MIC/SIR inconsistencies
    - Handles special values (e.g., 'trm', 'c')
    """
    
    def __init__(self, add_inconsistency_flags=True, verbose=False):
        """
        Parameters
        ----------
        add_inconsistency_flags : bool, default=True
            Whether to add columns flagging MIC/SIR inconsistencies
        verbose : bool, default=False
            Whether to print cleaning statistics
        """
        self.add_inconsistency_flags = add_inconsistency_flags
        self.verbose = verbose
        self.antibiotics_ = []
        self.cleaning_stats_ = {}
        
    def _identify_antibiotics(self, df):
        """Identify all antibiotics that have both MIC and interpretation columns."""
        antibiotics = set()
        mic_cols = [col for col in df.columns if '_mic' in col.lower()]
        int_cols = [col for col in df.columns if '_int' in col.lower()]
        
        for mic_col in mic_cols:
            antibiotic = mic_col.replace('_mic', '').replace('_MIC', '')
            int_col_lower = antibiotic + '_int'
            int_col_upper = antibiotic + '_INT'
            
            if int_col_lower in df.columns or int_col_upper in df.columns:
                antibiotics.add(antibiotic)
        
        return sorted(list(antibiotics))
    
    def _normalize_mic_value(self, value):
        """
        Normalize MIC value to standard format.
        
        Handles:
        - Unicode comparison operators (≤, ≥) -> standard (<=, >=)
        - Asterisks and special characters
        - Invalid values (trm, c) -> NaN
        - Numeric extraction
        """
        if pd.isna(value):
            return np.nan, np.nan, np.nan
        
        value_str = str(value).strip()
        
        # Handle special invalid values
        if value_str.lower() in ['trm', 'c', 'nan', '']:
            return np.nan, np.nan, np.nan
        
        # Remove asterisks
        value_str = value_str.replace('*', '')
        
        # Normalize Unicode operators
        value_str = value_str.replace('≤', '<=').replace('≥', '>=')
        value_str = value_str.replace('⩽', '<=').replace('⩾', '>=')
        
        # Extract operator and numeric value
        operator = None
        numeric_value = None
        
        if value_str.startswith('<='):
            operator = '<='
            numeric_str = value_str[2:].strip()
        elif value_str.startswith('>='):
            operator = '>='
            numeric_str = value_str[2:].strip()
        elif value_str.startswith('<'):
            operator = '<'
            numeric_str = value_str[1:].strip()
        elif value_str.startswith('>'):
            operator = '>'
            numeric_str = value_str[1:].strip()
        else:
            operator = '='
            numeric_str = value_str
        
        # Extract numeric value
        try:
            # Handle fractions (e.g., "2/38")
            if '/' in numeric_str:
                parts = numeric_str.split('/')
                numeric_value = float(parts[0]) / float(parts[1])
            else:
                # Replace comma with dot for European notation
                numeric_str = numeric_str.replace(',', '.')
                numeric_value = float(numeric_str)
        except (ValueError, ZeroDivisionError):
            return np.nan, np.nan, np.nan
        
        return value_str, operator, numeric_value
    
    def _standardize_sir_value(self, value):
        """
        Standardize S/I/R interpretation value.
        
        Converts to lowercase, removes asterisks, handles special characters.
        """
        if pd.isna(value):
            return np.nan
        
        value_str = str(value).strip().lower()
        
        # Remove asterisks
        value_str = value_str.replace('*', '')
        
        # Only keep valid values
        if value_str in ['s', 'i', 'r']:
            return value_str
        
        return np.nan
    
    def _check_mic_sir_consistency(self, mic_numeric, operator, sir_value):
        """
        Check if MIC value is consistent with S/I/R interpretation.
        
        This is a simplified check. In practice, each antibiotic has specific
        breakpoints that define S/I/R categories. This implementation flags
        potential inconsistencies for further review.
        
        Returns True if potentially inconsistent, False otherwise.
        """
        if pd.isna(mic_numeric) or pd.isna(sir_value):
            return False
        
        # Simplified heuristic: very low MIC values (<=2) should typically be S,
        # very high MIC values (>=32) should typically be R
        # This is a rough approximation and should be refined with actual breakpoints
        
        if operator == '<=' and mic_numeric <= 2 and sir_value == 'r':
            return True  # Potentially inconsistent
        
        if operator == '>=' and mic_numeric >= 32 and sir_value == 's':
            return True  # Potentially inconsistent
        
        return False
    
    def fit(self, X, y=None):
        """
        Fit the cleaner to the data (identifies antibiotics).
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with MIC and interpretation columns
        y : None
            Ignored, present for API consistency
            
        Returns
        -------
        self
        """
        self.antibiotics_ = self._identify_antibiotics(X)
        
        if self.verbose:
            print(f"Identified {len(self.antibiotics_)} antibiotics for cleaning")
        
        return self
    
    def transform(self, X):
        """
        Transform the data by cleaning MIC and S/I/R values.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with normalized values
        """
        X_clean = X.copy()
        self.cleaning_stats_ = {}
        
        for antibiotic in self.antibiotics_:
            mic_col = f"{antibiotic}_mic"
            int_col = f"{antibiotic}_int"
            
            if mic_col not in X_clean.columns or int_col not in X_clean.columns:
                continue
            
            stats = {
                'mic_cleaned': 0,
                'sir_cleaned': 0,
                'inconsistencies': 0
            }
            
            # Create new columns for cleaned values
            mic_cleaned_col = f"{antibiotic}_mic_clean"
            mic_operator_col = f"{antibiotic}_mic_operator"
            mic_numeric_col = f"{antibiotic}_mic_numeric"
            int_cleaned_col = f"{antibiotic}_int_clean"
            
            # Clean MIC values
            mic_data = X_clean[mic_col].apply(self._normalize_mic_value)
            X_clean[mic_cleaned_col] = [x[0] for x in mic_data]
            X_clean[mic_operator_col] = [x[1] for x in mic_data]
            X_clean[mic_numeric_col] = [x[2] for x in mic_data]
            
            stats['mic_cleaned'] = X_clean[mic_numeric_col].notna().sum()
            
            # Clean S/I/R values
            X_clean[int_cleaned_col] = X_clean[int_col].apply(self._standardize_sir_value)
            stats['sir_cleaned'] = X_clean[int_cleaned_col].notna().sum()
            
            # Check for inconsistencies
            if self.add_inconsistency_flags:
                inconsistency_col = f"{antibiotic}_mic_sir_inconsistent"
                X_clean[inconsistency_col] = [
                    self._check_mic_sir_consistency(mic_num, op, sir)
                    for mic_num, op, sir in zip(
                        X_clean[mic_numeric_col],
                        X_clean[mic_operator_col],
                        X_clean[int_cleaned_col]
                    )
                ]
                stats['inconsistencies'] = X_clean[inconsistency_col].sum()
            
            self.cleaning_stats_[antibiotic] = stats
        
        if self.verbose:
            self._print_stats()
        
        return X_clean
    
    def _print_stats(self):
        """Print cleaning statistics."""
        print("\n" + "="*80)
        print("MIC/SIR CLEANING STATISTICS")
        print("="*80)
        
        for antibiotic, stats in self.cleaning_stats_.items():
            print(f"\n{antibiotic}:")
            print(f"  MIC values cleaned: {stats['mic_cleaned']}")
            print(f"  SIR values cleaned: {stats['sir_cleaned']}")
            if 'inconsistencies' in stats:
                print(f"  Potential inconsistencies: {stats['inconsistencies']}")
    
    def get_cleaning_stats(self):
        """
        Get cleaning statistics.
        
        Returns
        -------
        dict
            Dictionary with cleaning statistics per antibiotic
        """
        return self.cleaning_stats_
