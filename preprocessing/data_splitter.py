"""
Data splitting module for creating train/test/validation splits.

Implements:
- Stratified splits by species, site, or resistance profile
- Time-aware splits (when temporal data is available)
- Cross-validation fold generation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings


class StratifiedDataSplitter:
    """
    Data splitter with stratification support for antibiotic resistance data.
    
    Features:
    - Stratified splits by species, site, or custom column
    - Time-aware splits preserving temporal order
    - Train/validation/test split generation
    - Cross-validation fold creation
    """
    
    def __init__(self,
                 stratify_by='bacterial_species',
                 test_size=0.2,
                 val_size=0.1,
                 random_state=42,
                 temporal_col=None,
                 verbose=False):
        """
        Parameters
        ----------
        stratify_by : str or list, default='bacterial_species'
            Column(s) to stratify by. Common options:
            - 'bacterial_species' : stratify by species
            - 'administrative_region' : stratify by region
            - 'sample_source' : stratify by sample source
            - ['bacterial_species', 'administrative_region'] : multi-column stratification
        test_size : float, default=0.2
            Proportion of data for test set
        val_size : float, default=0.1
            Proportion of training data for validation set
        random_state : int, default=42
            Random seed for reproducibility
        temporal_col : str, optional
            Column name for temporal ordering (if available)
        verbose : bool, default=False
            Whether to print split statistics
        """
        self.stratify_by = stratify_by
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.temporal_col = temporal_col
        self.verbose = verbose
        
        self.train_indices_ = None
        self.val_indices_ = None
        self.test_indices_ = None
    
    def _create_stratification_key(self, df, columns):
        """
        Create a stratification key from one or more columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : str or list
            Column(s) to use for stratification
            
        Returns
        -------
        pd.Series
            Stratification key
        """
        if isinstance(columns, str):
            columns = [columns]
        
        # Combine columns into a single stratification key
        strat_key = df[columns[0]].astype(str)
        for col in columns[1:]:
            strat_key = strat_key + '_' + df[col].astype(str)
        
        return strat_key
    
    def _temporal_split(self, df):
        """
        Create time-aware splits preserving temporal order.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with temporal column
            
        Returns
        -------
        tuple
            (train_indices, val_indices, test_indices)
        """
        if self.temporal_col not in df.columns:
            raise ValueError(f"Temporal column '{self.temporal_col}' not found in dataframe")
        
        # Sort by temporal column
        df_sorted = df.sort_values(self.temporal_col).reset_index(drop=True)
        
        n_total = len(df_sorted)
        n_test = int(n_total * self.test_size)
        n_val = int((n_total - n_test) * self.val_size)
        n_train = n_total - n_test - n_val
        
        # Split by time: train (oldest) -> val -> test (newest)
        train_indices = df_sorted.index[:n_train].tolist()
        val_indices = df_sorted.index[n_train:n_train+n_val].tolist()
        test_indices = df_sorted.index[n_train+n_val:].tolist()
        
        return train_indices, val_indices, test_indices
    
    def _stratified_split(self, df):
        """
        Create stratified random splits.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        tuple
            (train_indices, val_indices, test_indices)
        """
        # Create stratification key
        strat_key = self._create_stratification_key(df, self.stratify_by)
        
        # Handle rare classes by grouping them
        class_counts = strat_key.value_counts()
        min_samples = 2  # Minimum samples needed for stratification
        rare_classes = class_counts[class_counts < min_samples].index
        
        if len(rare_classes) > 0:
            warnings.warn(
                f"Found {len(rare_classes)} rare classes with <{min_samples} samples. "
                f"These will be grouped as 'rare' for stratification."
            )
            strat_key = strat_key.replace(rare_classes, 'rare')
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            df.index,
            test_size=self.test_size,
            stratify=strat_key,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        if self.val_size > 0:
            # Adjust validation size relative to train+val set
            adjusted_val_size = self.val_size / (1 - self.test_size)
            
            # Get stratification key for train+val set
            strat_key_train_val = strat_key.loc[train_val_idx]
            
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=adjusted_val_size,
                stratify=strat_key_train_val,
                random_state=self.random_state
            )
        else:
            train_idx = train_val_idx
            val_idx = []
        
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
    
    def split(self, df):
        """
        Split data into train, validation, and test sets.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        tuple
            (train_df, val_df, test_df)
        """
        if self.temporal_col is not None and self.temporal_col in df.columns:
            # Use temporal split
            train_idx, val_idx, test_idx = self._temporal_split(df)
        else:
            # Use stratified split
            train_idx, val_idx, test_idx = self._stratified_split(df)
        
        # Store indices
        self.train_indices_ = train_idx
        self.val_indices_ = val_idx
        self.test_indices_ = test_idx
        
        # Create dataframes
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True) if val_idx else pd.DataFrame()
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        if self.verbose:
            self._print_split_stats(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def get_indices(self):
        """
        Get indices for train, validation, and test sets.
        
        Returns
        -------
        tuple
            (train_indices, val_indices, test_indices)
        """
        return self.train_indices_, self.val_indices_, self.test_indices_
    
    def create_cv_folds(self, df, n_folds=5):
        """
        Create cross-validation folds with stratification.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        n_folds : int, default=5
            Number of cross-validation folds
            
        Returns
        -------
        list
            List of (train_indices, val_indices) tuples for each fold
        """
        # Create stratification key
        strat_key = self._create_stratification_key(df, self.stratify_by)
        
        # Handle rare classes
        class_counts = strat_key.value_counts()
        min_samples = n_folds  # Need at least n_folds samples per class
        rare_classes = class_counts[class_counts < min_samples].index
        
        if len(rare_classes) > 0:
            warnings.warn(
                f"Found {len(rare_classes)} rare classes with <{min_samples} samples. "
                f"These will be grouped as 'rare' for stratification."
            )
            strat_key = strat_key.replace(rare_classes, 'rare')
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        folds = []
        for train_idx, val_idx in skf.split(df, strat_key):
            folds.append((train_idx.tolist(), val_idx.tolist()))
        
        if self.verbose:
            print(f"\nCreated {n_folds} cross-validation folds")
            for i, (train_idx, val_idx) in enumerate(folds):
                print(f"  Fold {i+1}: {len(train_idx)} train, {len(val_idx)} validation")
        
        return folds
    
    def _print_split_stats(self, train_df, val_df, test_df):
        """Print statistics about the data splits."""
        print("\n" + "="*80)
        print("DATA SPLIT STATISTICS")
        print("="*80)
        
        print(f"\nTotal samples: {len(train_df) + len(val_df) + len(test_df)}")
        print(f"  Train: {len(train_df)} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        if len(val_df) > 0:
            print(f"  Validation: {len(val_df)} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        print(f"  Test: {len(test_df)} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        
        # Print stratification statistics
        if isinstance(self.stratify_by, str):
            stratify_cols = [self.stratify_by]
        else:
            stratify_cols = self.stratify_by
        
        for col in stratify_cols:
            if col in train_df.columns:
                print(f"\n{col} distribution:")
                print("  Train:")
                print("    " + train_df[col].value_counts().to_string().replace('\n', '\n    '))
                if len(val_df) > 0:
                    print("  Validation:")
                    print("    " + val_df[col].value_counts().to_string().replace('\n', '\n    '))
                print("  Test:")
                print("    " + test_df[col].value_counts().to_string().replace('\n', '\n    '))
