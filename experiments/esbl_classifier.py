"""
ESBL Classification Experiment.

Binary classification task to predict ESBL status from resistance profile + metadata.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, brier_score_loss,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from .base_experiment import BaseExperiment


class ESBLClassifierExperiment(BaseExperiment):
    """
    Binary ESBL classification experiment.
    
    Predicts ESBL status (positive/negative) from:
    - Resistance profile (binary resistance indicators)
    - Antibiogram features
    - Aggregate resistance metrics
    - Metadata features
    """
    
    def __init__(self, output_dir='models'):
        super().__init__(
            experiment_name="ESBL_Classification",
            task_name="esbl_classifier",
            output_dir=output_dir
        )
        self.binary = True
    
    def prepare_data(self, df):
        """
        Prepare data for ESBL classification.
        
        Parameters
        ----------
        df : pd.DataFrame
            Processed dataframe with all features
            
        Returns
        -------
        tuple
            (X, y) where X is features and y is ESBL label
        """
        # Check if ESBL column exists
        if 'esbl_encoded' not in df.columns:
            raise ValueError("ESBL status not found in dataframe. Check preprocessing.")
        
        # Features: resistance profiles + metadata
        # Exclude target and original ESBL column
        feature_cols = [col for col in df.columns if col not in [
            'esbl', 'esbl_encoded', 'isolate_code', 'bacterial_species',
            'administrative_region', 'national_site', 'local_site',
            'sample_source'
        ]]
        
        # Filter to keep only numeric and useful features
        feature_cols = [col for col in feature_cols if 
                       any(x in col for x in [
                           '_resistant', '_mic_numeric', 'antibiogram_',
                           'total_', 'ratio', '_count', 'who_', 'mar_index',
                           '_encoded', 'replicate', 'colony'
                       ])]
        
        X = df[feature_cols].values
        y = df['esbl_encoded'].values
        
        # Handle missing values in target
        valid_idx = ~pd.isna(y)
        X = X[valid_idx]
        y = y[valid_idx].astype(int)
        
        print(f"Prepared ESBL classification data:")
        print(f"  Samples: {len(y)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Class distribution: {np.bincount(y)}")
        print(f"  Positive rate: {y.mean():.2%}")
        
        return X, y, feature_cols
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate ESBL classifier.
        
        Metrics:
        - AUROC
        - AUPRC
        - Accuracy, Precision, Recall, F1
        - Brier score (calibration)
        - Sensitivity at 95% specificity
        
        Parameters
        ----------
        model : sklearn estimator
            Trained model
        X_test, y_test : array-like
            Test data
            
        Returns
        -------
        dict
            Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Add probability-based metrics if available
        if hasattr(model, 'predict_proba'):
            metrics['auroc'] = roc_auc_score(y_test, y_proba)
            metrics['auprc'] = average_precision_score(y_test, y_proba)
            metrics['brier_score'] = brier_score_loss(y_test, y_proba)
            
            # Sensitivity at 95% specificity
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            specificity = 1 - fpr
            idx_95_spec = np.argmin(np.abs(specificity - 0.95))
            metrics['sensitivity_at_95spec'] = tpr[idx_95_spec]
        
        return metrics
    
    def plot_roc_curve(self, model, X_test, y_test, save_path=None):
        """Plot ROC curve."""
        if not hasattr(model, 'predict_proba'):
            print("Model does not support predict_proba. Skipping ROC curve.")
            return
        
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - ESBL Classification')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, model, X_test, y_test, save_path=None):
        """Plot Precision-Recall curve."""
        if not hasattr(model, 'predict_proba'):
            print("Model does not support predict_proba. Skipping PR curve.")
            return
        
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUPRC = {auprc:.3f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - ESBL Classification')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - ESBL Classification')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
