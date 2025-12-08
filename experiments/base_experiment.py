"""
Base experiment class for supervised learning tasks.

Provides common functionality for:
- Model training and evaluation
- Nested cross-validation
- Hyperparameter tuning
- MLflow tracking
- Metrics computation
- Model persistence
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, log_loss, brier_score_loss
)
import json
import joblib
from pathlib import Path
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class BaseExperiment(ABC):
    """
    Base class for supervised learning experiments.
    
    Provides:
    - Nested CV framework
    - Model training and evaluation
    - MLflow experiment tracking
    - Hyperparameter tuning
    - Model persistence
    """
    
    # Define the six algorithms to test
    ALGORITHMS = {
        'random_forest': {
            'model': RandomForestClassifier,
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'random_state': [42]
            },
            'requires_scaling': False
        },
        'logistic_regression': {
            'model': LogisticRegression,
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000],
                'random_state': [42]
            },
            'requires_scaling': True
        },
        'svm': {
            'model': SVC,
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
                'probability': [True],
                'random_state': [42]
            },
            'requires_scaling': True
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier,
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'random_state': [42]
            },
            'requires_scaling': False
        },
        'knn': {
            'model': KNeighborsClassifier,
            'params': {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'requires_scaling': True
        },
        'naive_bayes': {
            'model': GaussianNB,
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            },
            'requires_scaling': False
        }
    }
    
    def __init__(self, experiment_name, task_name, output_dir='models'):
        """
        Initialize experiment.
        
        Parameters
        ----------
        experiment_name : str
            Name of the MLflow experiment
        task_name : str
            Name of the specific task
        output_dir : str
            Directory to save models and artifacts
        """
        self.experiment_name = experiment_name
        self.task_name = task_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        self.results = {}
        self.best_models = {}
    
    @abstractmethod
    def prepare_data(self, df):
        """
        Prepare data for the specific task.
        Must be implemented by subclasses.
        
        Parameters
        ----------
        df : pd.DataFrame
            Processed dataframe with features and target
            
        Returns
        -------
        tuple
            (X, y, feature_names) where X is feature matrix,
            y is target, and feature_names is list of feature column names
        """
        pass
    
    @abstractmethod
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance.
        Must be implemented by subclasses.
        """
        pass
    
    def train_with_nested_cv(self, X, y, algorithm_name, n_outer_folds=5, n_inner_folds=3):
        """
        Train model using nested cross-validation.
        
        Outer CV: Performance estimation
        Inner CV: Hyperparameter tuning
        
        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Target
        algorithm_name : str
            Name of algorithm from ALGORITHMS dict
        n_outer_folds : int
            Number of outer CV folds
        n_inner_folds : int
            Number of inner CV folds
            
        Returns
        -------
        dict
            Results dictionary with metrics
        """
        if algorithm_name not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        algo_config = self.ALGORITHMS[algorithm_name]
        model_class = algo_config['model']
        param_grid = algo_config['params']
        requires_scaling = algo_config['requires_scaling']
        
        # Determine CV strategy
        if self.is_classification():
            outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
            inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
        else:
            outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
            inner_cv = KFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
        
        outer_results = []
        best_params_list = []
        
        # Outer CV loop
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale if needed
            if requires_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                model_class(),
                param_grid,
                cv=inner_cv,
                scoring=self.get_scoring_metric(),
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate on test fold
            fold_metrics = self.evaluate_model(grid_search.best_estimator_, X_test, y_test)
            fold_metrics['fold'] = fold_idx
            fold_metrics['best_params'] = grid_search.best_params_
            
            outer_results.append(fold_metrics)
            best_params_list.append(grid_search.best_params_)
        
        # Aggregate results
        aggregated_metrics = self._aggregate_cv_results(outer_results)
        aggregated_metrics['algorithm'] = algorithm_name
        aggregated_metrics['best_params_per_fold'] = best_params_list
        
        return aggregated_metrics
    
    def train_final_model(self, X_train, y_train, X_test, y_test, algorithm_name, params=None):
        """
        Train final model on full training set with best parameters.
        
        Parameters
        ----------
        X_train, y_train : array-like
            Training data
        X_test, y_test : array-like
            Test data
        algorithm_name : str
            Algorithm name
        params : dict, optional
            Hyperparameters (if None, uses defaults)
            
        Returns
        -------
        tuple
            (trained_model, metrics, scaler)
        """
        if algorithm_name not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        algo_config = self.ALGORITHMS[algorithm_name]
        model_class = algo_config['model']
        requires_scaling = algo_config['requires_scaling']
        
        # Scale if needed
        scaler = None
        if requires_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Initialize model with parameters
        if params is None:
            model = model_class()
        else:
            model = model_class(**params)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test)
        metrics['algorithm'] = algorithm_name
        
        return model, metrics, scaler
    
    def run_all_algorithms(self, X_train, y_train, X_test, y_test, use_nested_cv=True):
        """
        Run all six algorithms on the task.
        
        Parameters
        ----------
        X_train, y_train : array-like
            Training data
        X_test, y_test : array-like
            Test data
        use_nested_cv : bool
            Whether to use nested CV for hyperparameter tuning
            
        Returns
        -------
        dict
            Results for all algorithms
        """
        all_results = {}
        
        for algo_name in self.ALGORITHMS.keys():
            print(f"\n{'='*80}")
            print(f"Training {algo_name}...")
            print(f"{'='*80}")
            
            with mlflow.start_run(run_name=f"{self.task_name}_{algo_name}"):
                # Log parameters
                mlflow.log_param("algorithm", algo_name)
                mlflow.log_param("task", self.task_name)
                
                if use_nested_cv:
                    # Nested CV on training set
                    cv_results = self.train_with_nested_cv(X_train, y_train, algo_name)
                    
                    # Train final model with mean of best params
                    best_params = self._get_consensus_params(cv_results['best_params_per_fold'])
                    model, test_metrics, scaler = self.train_final_model(
                        X_train, y_train, X_test, y_test, algo_name, best_params
                    )
                    
                    # Combine results
                    results = {
                        'cv_metrics': cv_results,
                        'test_metrics': test_metrics,
                        'best_params': best_params
                    }
                else:
                    # Simple train-test split
                    model, test_metrics, scaler = self.train_final_model(
                        X_train, y_train, X_test, y_test, algo_name
                    )
                    results = {
                        'test_metrics': test_metrics,
                        'best_params': None
                    }
                
                # Log metrics to MLflow
                self._log_metrics_to_mlflow(test_metrics)
                
                # Save model
                model_path = self.output_dir / f"{self.task_name}_{algo_name}_model.pkl"
                joblib.dump({'model': model, 'scaler': scaler}, model_path)
                mlflow.log_artifact(str(model_path))
                
                all_results[algo_name] = results
                self.best_models[algo_name] = {'model': model, 'scaler': scaler}
                
                print(f"\n{algo_name} completed!")
                print(f"Test metrics: {test_metrics}")
        
        self.results = all_results
        return all_results
    
    def _aggregate_cv_results(self, results_list):
        """Aggregate metrics across CV folds."""
        aggregated = {}
        
        # Get all metric keys (excluding non-numeric fields)
        metric_keys = [k for k in results_list[0].keys() 
                      if k not in ['fold', 'best_params', 'confusion_matrix']]
        
        for key in metric_keys:
            values = [r[key] for r in results_list if key in r]
            if values and isinstance(values[0], (int, float)):
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
        
        return aggregated
    
    def _get_consensus_params(self, params_list):
        """Get consensus parameters from CV folds."""
        # Use mode for categorical, mean for numeric
        consensus = {}
        
        if not params_list:
            return {}
        
        for key in params_list[0].keys():
            values = [p[key] for p in params_list]
            
            # Use most common value
            from collections import Counter
            consensus[key] = Counter(values).most_common(1)[0][0]
        
        return consensus
    
    def _log_metrics_to_mlflow(self, metrics):
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
    
    def is_classification(self):
        """Override in subclass if task is regression."""
        return True
    
    def get_scoring_metric(self):
        """Get scoring metric for GridSearchCV."""
        if self.is_classification():
            return 'roc_auc' if hasattr(self, 'binary') and self.binary else 'f1_macro'
        else:
            return 'neg_mean_squared_error'
    
    def save_results(self, filename=None):
        """Save experiment results to JSON."""
        if filename is None:
            filename = self.output_dir / f"{self.task_name}_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = self._make_json_serializable(self.results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
