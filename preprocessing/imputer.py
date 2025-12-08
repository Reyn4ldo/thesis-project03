"""
Domain-aware imputation module for missing values.

Implements multiple imputation strategies:
- Domain-aware categorical imputation (e.g., "not_tested" category)
- KNN imputation for numeric MIC values
- Median imputation for numeric values
- Indicator flags for imputed values
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
import warnings


class DomainAwareImputer(BaseEstimator, TransformerMixin):
    """
    Domain-aware imputer for antibiotic resistance data.
    
    Features:
    - Handles missing MIC values with KNN or median imputation
    - Handles missing S/I/R interpretations with 'not_tested' category
    - Adds indicator flags for imputed values
    - Respects domain-specific constraints
    """
    
    def __init__(self, 
                 mic_strategy='median',
                 sir_strategy='not_tested',
                 add_indicators=True,
                 knn_neighbors=5,
                 verbose=False):
        """
        Parameters
        ----------
        mic_strategy : str, default='median'
            Strategy for imputing MIC values: 'median', 'knn', or 'drop'
        sir_strategy : str, default='not_tested'
            Strategy for imputing S/I/R values: 'not_tested', 'mode', or 'drop'
        add_indicators : bool, default=True
            Whether to add indicator columns for imputed values
        knn_neighbors : int, default=5
            Number of neighbors for KNN imputation
        verbose : bool, default=False
            Whether to print imputation statistics
        """
        self.mic_strategy = mic_strategy
        self.sir_strategy = sir_strategy
        self.add_indicators = add_indicators
        self.knn_neighbors = knn_neighbors
        self.verbose = verbose
        
        self.mic_imputer_ = None
        self.sir_imputer_ = None
        self.antibiotics_ = []
        self.imputation_stats_ = {}
    
    def _identify_antibiotics(self, df):
        """Identify all antibiotics with MIC or interpretation columns."""
        antibiotics = set()
        
        for col in df.columns:
            if '_mic_numeric' in col:
                antibiotic = col.replace('_mic_numeric', '')
                antibiotics.add(antibiotic)
            elif '_int_clean' in col:
                antibiotic = col.replace('_int_clean', '')
                antibiotics.add(antibiotic)
        
        return sorted(list(antibiotics))
    
    def fit(self, X, y=None):
        """
        Fit the imputer to the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe
        y : None
            Ignored, present for API consistency
            
        Returns
        -------
        self
        """
        self.antibiotics_ = self._identify_antibiotics(X)
        
        # Fit MIC imputer if using KNN
        if self.mic_strategy == 'knn':
            mic_cols = [f"{ab}_mic_numeric" for ab in self.antibiotics_ 
                       if f"{ab}_mic_numeric" in X.columns]
            
            if mic_cols:
                self.mic_imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
                self.mic_imputer_.fit(X[mic_cols])
        
        if self.verbose:
            print(f"Imputer fitted for {len(self.antibiotics_)} antibiotics")
        
        return self
    
    def transform(self, X):
        """
        Transform the data by imputing missing values.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        pd.DataFrame
            Dataframe with imputed values
        """
        X_imputed = X.copy()
        self.imputation_stats_ = {}
        
        for antibiotic in self.antibiotics_:
            stats = {
                'mic_missing': 0,
                'mic_imputed': 0,
                'sir_missing': 0,
                'sir_imputed': 0
            }
            
            # Handle MIC imputation
            mic_col = f"{antibiotic}_mic_numeric"
            if mic_col in X_imputed.columns:
                missing_mask = X_imputed[mic_col].isna()
                stats['mic_missing'] = missing_mask.sum()
                
                if stats['mic_missing'] > 0:
                    if self.mic_strategy == 'median':
                        median_value = X_imputed[mic_col].median()
                        if not pd.isna(median_value):
                            X_imputed.loc[missing_mask, mic_col] = median_value
                            stats['mic_imputed'] = missing_mask.sum()
                    
                    elif self.mic_strategy == 'knn' and self.mic_imputer_ is not None:
                        # KNN imputation is done collectively for better results
                        pass  # Handled separately below
                    
                    # Add indicator flag
                    if self.add_indicators:
                        indicator_col = f"{antibiotic}_mic_imputed"
                        X_imputed[indicator_col] = missing_mask.astype(int)
            
            # Handle S/I/R imputation
            sir_col = f"{antibiotic}_int_clean"
            if sir_col in X_imputed.columns:
                missing_mask = X_imputed[sir_col].isna()
                stats['sir_missing'] = missing_mask.sum()
                
                if stats['sir_missing'] > 0:
                    if self.sir_strategy == 'not_tested':
                        X_imputed.loc[missing_mask, sir_col] = 'not_tested'
                        stats['sir_imputed'] = missing_mask.sum()
                    
                    elif self.sir_strategy == 'mode':
                        mode_value = X_imputed[sir_col].mode()
                        if len(mode_value) > 0:
                            X_imputed.loc[missing_mask, sir_col] = mode_value[0]
                            stats['sir_imputed'] = missing_mask.sum()
                    
                    # Add indicator flag
                    if self.add_indicators:
                        indicator_col = f"{antibiotic}_sir_imputed"
                        X_imputed[indicator_col] = missing_mask.astype(int)
            
            self.imputation_stats_[antibiotic] = stats
        
        # Perform KNN imputation collectively if selected
        if self.mic_strategy == 'knn' and self.mic_imputer_ is not None:
            mic_cols = [f"{ab}_mic_numeric" for ab in self.antibiotics_ 
                       if f"{ab}_mic_numeric" in X_imputed.columns]
            
            if mic_cols:
                imputed_values = self.mic_imputer_.transform(X_imputed[mic_cols])
                X_imputed[mic_cols] = imputed_values
                
                # Update stats
                for i, col in enumerate(mic_cols):
                    antibiotic = col.replace('_mic_numeric', '')
                    if antibiotic in self.imputation_stats_:
                        self.imputation_stats_[antibiotic]['mic_imputed'] = \
                            self.imputation_stats_[antibiotic]['mic_missing']
        
        if self.verbose:
            self._print_stats()
        
        return X_imputed
    
    def _print_stats(self):
        """Print imputation statistics."""
        print("\n" + "="*80)
        print("IMPUTATION STATISTICS")
        print("="*80)
        
        total_mic_missing = 0
        total_mic_imputed = 0
        total_sir_missing = 0
        total_sir_imputed = 0
        
        for antibiotic, stats in self.imputation_stats_.items():
            if stats['mic_missing'] > 0 or stats['sir_missing'] > 0:
                print(f"\n{antibiotic}:")
                if stats['mic_missing'] > 0:
                    print(f"  MIC: {stats['mic_imputed']}/{stats['mic_missing']} imputed")
                    total_mic_missing += stats['mic_missing']
                    total_mic_imputed += stats['mic_imputed']
                if stats['sir_missing'] > 0:
                    print(f"  S/I/R: {stats['sir_imputed']}/{stats['sir_missing']} imputed")
                    total_sir_missing += stats['sir_missing']
                    total_sir_imputed += stats['sir_imputed']
        
        print(f"\nTotal MIC values imputed: {total_mic_imputed}/{total_mic_missing}")
        print(f"Total S/I/R values imputed: {total_sir_imputed}/{total_sir_missing}")
    
    def get_imputation_stats(self):
        """
        Get imputation statistics.
        
        Returns
        -------
        dict
            Dictionary with imputation statistics per antibiotic
        """
        return self.imputation_stats_
