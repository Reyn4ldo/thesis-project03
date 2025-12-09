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
        
        # Filter out samples where original esbl was missing
        # The preprocessing fills missing values with 'missing' before encoding,
        # resulting in 3+ classes. We only want binary classification.
        if 'esbl' in df.columns:
            # Filter based on original esbl column
            valid_esbl_idx = df['esbl'].notna()
            df_filtered = df[valid_esbl_idx].copy()
        else:
            # If original esbl column is not present, try to identify missing values
            # by checking if esbl_encoded has more than 2 unique values
            df_filtered = df.copy()
        
        # Features: resistance profiles + metadata
        # Exclude target and original ESBL column
        feature_cols = [col for col in df_filtered.columns if col not in [
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
        
        X = df_filtered[feature_cols].values
        y = df_filtered['esbl_encoded'].values.astype(int)
        
        # Verify we have binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) > 2:
            print(f"WARNING: Found {len(unique_classes)} classes in ESBL target: {unique_classes}")
            print("Filtering to keep only the two most common classes for binary classification")
            # Keep only the two most common classes
            class_counts = np.bincount(y)
            # Get indices of two most common classes
            top_2_classes = np.argsort(class_counts)[-2:]
            valid_idx = np.isin(y, top_2_classes)
            X = X[valid_idx]
            y = y[valid_idx]
            # Remap to 0 and 1
            y = (y == top_2_classes[1]).astype(int)
            unique_classes = np.unique(y)
        
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
        
        # Check if we have binary classification
        unique_test = np.unique(y_test)
        unique_pred = np.unique(y_pred)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # For binary metrics, check if we have at least 2 classes in test set
        if len(unique_test) >= 2:
            metrics['precision'] = precision_score(y_test, y_pred, average='binary', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='binary', zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, average='binary', zero_division=0)
        else:
            # Single class in test set - use macro averaging as fallback
            print(f"WARNING: Only {len(unique_test)} class(es) in test set. Using macro averaging.")
            metrics['precision'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Add probability-based metrics if available and we have binary classes
        if hasattr(model, 'predict_proba') and len(unique_test) >= 2:
            try:
                metrics['auroc'] = roc_auc_score(y_test, y_proba)
                metrics['auprc'] = average_precision_score(y_test, y_proba)
                metrics['brier_score'] = brier_score_loss(y_test, y_proba)
                
                # Sensitivity at 95% specificity
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                specificity = 1 - fpr
                idx_95_spec = np.argmin(np.abs(specificity - 0.95))
                metrics['sensitivity_at_95spec'] = tpr[idx_95_spec]
            except ValueError as e:
                print(f"WARNING: Could not compute probability-based metrics: {e}")
        
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
