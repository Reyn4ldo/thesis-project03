"""
Multi-label R/S Prediction Experiment.

Predicts resistance/susceptibility for multiple antibiotics simultaneously.
"""

import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from .base_experiment import BaseExperiment


class MultilabelPredictionExperiment(BaseExperiment):
    """
    Multi-label R/S prediction experiment.
    
    Predicts R/S status for multiple antibiotics from metadata + other antibiotics.
    """
    
    def __init__(self, output_dir='models'):
        super().__init__(
            experiment_name="Multilabel_RS_Prediction",
            task_name="multilabel_prediction",
            output_dir=output_dir
        )
    
    def prepare_data(self, df):
        """Prepare data for multi-label prediction."""
        # Use metadata as features, binary resistance as targets
        feature_cols = [col for col in df.columns if 
                       any(x in col for x in ['_encoded', 'total_', 'ratio', 'mar_index'])]
        target_cols = [col for col in df.columns if '_resistant' in col and 'total' not in col]
        
        X = df[feature_cols].values
        y = df[target_cols].values
        
        print(f"Multi-label prediction data:")
        print(f"  Samples: {len(y)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Targets: {len(target_cols)}")
        
        return X, y, feature_cols
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate multi-label model."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'hamming_loss': hamming_loss(y_test, y_pred),
            'subset_accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0)
        }
        
        return metrics
