"""
Species Classification Experiment.

Predicts bacterial species from antibiogram profile.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .base_experiment import BaseExperiment


class SpeciesClassifierExperiment(BaseExperiment):
    """
    Species classification from antibiogram.
    """
    
    def __init__(self, output_dir='models'):
        super().__init__(
            experiment_name="Species_Classification",
            task_name="species_classifier",
            output_dir=output_dir
        )
    
    def prepare_data(self, df):
        """Prepare data for species classification."""
        # Check if bacterial_species_encoded exists
        if 'bacterial_species_encoded' not in df.columns:
            raise ValueError("Species encoding not found")
        
        # Filter out samples where original bacterial_species was missing
        # to avoid having 'missing' as a class
        if 'bacterial_species' in df.columns:
            valid_species_idx = df['bacterial_species'].notna()
            df_filtered = df[valid_species_idx].copy()
        else:
            df_filtered = df.copy()
        
        # Features: antibiogram vectors
        feature_cols = [col for col in df_filtered.columns if 'antibiogram_' in col or '_resistant' in col]
        
        X = df_filtered[feature_cols].values
        y = df_filtered['bacterial_species_encoded'].values.astype(int)
        
        print(f"Species classification data:")
        print(f"  Samples: {len(y)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Classes: {len(np.unique(y))}")
        
        return X, y, feature_cols
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate species classifier."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
