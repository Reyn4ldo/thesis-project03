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
        # Features: antibiogram vectors
        feature_cols = [col for col in df.columns if 'antibiogram_' in col or '_resistant' in col]
        
        X = df[feature_cols].values
        y = df['bacterial_species_encoded'].values if 'bacterial_species_encoded' in df.columns else None
        
        if y is None:
            raise ValueError("Species encoding not found")
        
        valid_idx = ~pd.isna(y)
        X = X[valid_idx]
        y = y[valid_idx].astype(int)
        
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
