"""
Main preprocessing pipeline combining all components.

Creates a reusable scikit-learn Pipeline for:
1. Data cleaning (MIC/SIR normalization)
2. Missing value imputation
3. Feature engineering
4. Data validation
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path

from .mic_sir_cleaner import MICSIRCleaner
from .imputer import DomainAwareImputer
from .feature_engineer import ResistanceFeatureEngineer


def create_preprocessing_pipeline(
    mic_strategy='median',
    sir_strategy='not_tested',
    encoding_strategy='onehot',
    add_inconsistency_flags=True,
    add_imputation_indicators=True,
    validate_mar_index=True,
    verbose=False
):
    """
    Create a complete preprocessing pipeline.
    
    Parameters
    ----------
    mic_strategy : str, default='median'
        Strategy for imputing MIC values: 'median', 'knn', or 'drop'
    sir_strategy : str, default='not_tested'
        Strategy for imputing S/I/R values: 'not_tested', 'mode', or 'drop'
    encoding_strategy : str, default='onehot'
        Encoding strategy for metadata: 'onehot' or 'label'
    add_inconsistency_flags : bool, default=True
        Whether to add MIC/SIR inconsistency flags
    add_imputation_indicators : bool, default=True
        Whether to add imputation indicator columns
    validate_mar_index : bool, default=True
        Whether to validate/calculate MAR index
    verbose : bool, default=False
        Whether to print processing statistics
        
    Returns
    -------
    Pipeline
        scikit-learn Pipeline object
    """
    pipeline = Pipeline([
        ('cleaner', MICSIRCleaner(
            add_inconsistency_flags=add_inconsistency_flags,
            verbose=verbose
        )),
        ('imputer', DomainAwareImputer(
            mic_strategy=mic_strategy,
            sir_strategy=sir_strategy,
            add_indicators=add_imputation_indicators,
            verbose=verbose
        )),
        ('feature_engineer', ResistanceFeatureEngineer(
            create_binary_resistance=True,
            create_antibiogram=True,
            create_aggregates=True,
            create_who_features=True,
            encode_metadata=True,
            encoding_strategy=encoding_strategy,
            validate_mar_index=validate_mar_index,
            verbose=verbose
        ))
    ])
    
    return pipeline


class PreprocessingPipelineWrapper:
    """
    Wrapper for preprocessing pipeline with save/load functionality.
    """
    
    def __init__(self, pipeline=None, config=None):
        """
        Parameters
        ----------
        pipeline : Pipeline, optional
            Existing pipeline to wrap
        config : dict, optional
            Configuration dictionary for creating a new pipeline
        """
        if pipeline is not None:
            self.pipeline = pipeline
        elif config is not None:
            self.pipeline = create_preprocessing_pipeline(**config)
        else:
            self.pipeline = create_preprocessing_pipeline()
        
        self.config = config or {}
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        """
        Fit the pipeline to data.
        
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
        self.pipeline.fit(X, y)
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform data using the fitted pipeline.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        pd.DataFrame
            Transformed dataframe
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")
        
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe
        y : None
            Ignored, present for API consistency
            
        Returns
        -------
        pd.DataFrame
            Transformed dataframe
        """
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, filepath):
        """
        Save the pipeline to disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the pipeline (without extension)
        """
        filepath = Path(filepath)
        
        # Save the pipeline
        pipeline_path = filepath.with_suffix('.pkl')
        joblib.dump(self.pipeline, pipeline_path)
        
        # Save configuration
        config_path = filepath.with_suffix('.json')
        config_data = {
            'config': self.config,
            'is_fitted': self.is_fitted_
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Pipeline saved to {pipeline_path}")
        print(f"Configuration saved to {config_path}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a pipeline from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to load the pipeline from (without extension)
            
        Returns
        -------
        PreprocessingPipelineWrapper
            Loaded pipeline wrapper
        """
        filepath = Path(filepath)
        
        # Load the pipeline
        pipeline_path = filepath.with_suffix('.pkl')
        pipeline = joblib.load(pipeline_path)
        
        # Load configuration
        config_path = filepath.with_suffix('.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        wrapper = cls(pipeline=pipeline, config=config_data.get('config', {}))
        wrapper.is_fitted_ = config_data.get('is_fitted', False)
        
        print(f"Pipeline loaded from {pipeline_path}")
        
        return wrapper
    
    def get_feature_names(self):
        """
        Get names of features created by the pipeline.
        
        Returns
        -------
        list
            List of feature names
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted first")
        
        # Get feature names from feature engineer step
        if 'feature_engineer' in self.pipeline.named_steps:
            return self.pipeline.named_steps['feature_engineer'].get_feature_names()
        
        return []
    
    def get_cleaning_stats(self):
        """
        Get cleaning statistics from the cleaner step.
        
        Returns
        -------
        dict
            Cleaning statistics
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted first")
        
        if 'cleaner' in self.pipeline.named_steps:
            return self.pipeline.named_steps['cleaner'].get_cleaning_stats()
        
        return {}
    
    def get_imputation_stats(self):
        """
        Get imputation statistics from the imputer step.
        
        Returns
        -------
        dict
            Imputation statistics
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted first")
        
        if 'imputer' in self.pipeline.named_steps:
            return self.pipeline.named_steps['imputer'].get_imputation_stats()
        
        return {}
