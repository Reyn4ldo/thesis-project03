"""
MAR Index Regression Experiment.

Predicts Multiple Antibiotic Resistance index from resistance profile + metadata.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .base_experiment import BaseExperiment


class MARRegressionExperiment(BaseExperiment):
    """
    MAR index regression experiment.
    """
    
    # Override algorithms for regression
    ALGORITHMS = {
        'random_forest': {
            'model': RandomForestRegressor,
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'random_state': [42]
            },
            'requires_scaling': False
        },
        'ridge': {
            'model': Ridge,
            'params': {
                'alpha': [0.1, 1.0, 10.0]
            },
            'requires_scaling': True
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor,
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'random_state': [42]
            },
            'requires_scaling': False
        },
        'svr': {
            'model': SVR,
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear']
            },
            'requires_scaling': True
        },
        'knn': {
            'model': KNeighborsRegressor,
            'params': {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance']
            },
            'requires_scaling': True
        }
    }
    
    def __init__(self, output_dir='models'):
        super().__init__(
            experiment_name="MAR_Regression",
            task_name="mar_regression",
            output_dir=output_dir
        )
    
    def is_classification(self):
        """This is a regression task."""
        return False
    
    def prepare_data(self, df):
        """Prepare data for MAR regression."""
        # Features: resistance profiles + metadata (excluding MAR)
        feature_cols = [col for col in df.columns if col not in [
            'mar_index', 'isolate_code', 'bacterial_species',
            'administrative_region', 'sample_source'
        ] and any(x in col for x in [
            '_resistant', 'antibiogram_', 'total_', '_encoded'
        ])]
        
        X = df[feature_cols].values
        y = df['mar_index'].values
        
        # Remove NaN targets
        valid_idx = ~pd.isna(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"MAR regression data:")
        print(f"  Samples: {len(y)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  MAR range: [{y.min():.3f}, {y.max():.3f}]")
        
        return X, y, feature_cols
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate MAR regression model."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
