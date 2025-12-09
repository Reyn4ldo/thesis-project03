"""
Feature engineering module for resistance data.

Creates derived features including:
- Binary R/S per antibiotic (0/1 encoding)
- MAR index calculation/validation
- Antibiogram fingerprints
- Aggregate resistance features
- WHO priority antibiotic tracking
- Metadata encoding
- Temporal and spatial features
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import warnings


class ResistanceFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer for antibiotic resistance data.
    
    Creates comprehensive features for modeling:
    - Binary resistance indicators per antibiotic
    - Antibiogram vectors
    - Aggregate resistance metrics
    - WHO priority antibiotic resistance
    - Encoded categorical features
    - Temporal and spatial features
    """
    
    # WHO priority antibiotics for resistance surveillance
    WHO_CRITICAL_ANTIBIOTICS = [
        'ceftaroline', 'cefotaxime', 'ceftazidime/avibactam', 'imepenem',
        'amikacin', 'gentamicin', 'enrofloxacin', 'marbofloxacin'
    ]
    
    # Antibiotic class mappings
    ANTIBIOTIC_CLASSES = {
        'beta_lactams': ['ampicillin', 'amoxicillin/clavulanic_acid', 'ceftaroline',
                         'cefalexin', 'cefalotin', 'cefpodoxime', 'cefotaxime',
                         'cefovecin', 'ceftiofur', 'ceftazidime/avibactam', 'imepenem'],
        'aminoglycosides': ['amikacin', 'gentamicin', 'neomycin'],
        'fluoroquinolones': ['nalidixic_acid', 'enrofloxacin', 'marbofloxacin', 'pradofloxacin'],
        'tetracyclines': ['doxycycline', 'tetracycline'],
        'others': ['nitrofurantoin', 'chloramphenicol', 'trimethoprim/sulfamethazole']
    }
    
    def __init__(self,
                 create_binary_resistance=True,
                 create_antibiogram=True,
                 create_aggregates=True,
                 create_who_features=True,
                 encode_metadata=True,
                 encoding_strategy='onehot',
                 validate_mar_index=True,
                 verbose=False):
        """
        Parameters
        ----------
        create_binary_resistance : bool, default=True
            Create binary R/S indicators per antibiotic
        create_antibiogram : bool, default=True
            Create antibiogram fingerprint vectors
        create_aggregates : bool, default=True
            Create aggregate resistance features
        create_who_features : bool, default=True
            Create WHO priority antibiotic features
        encode_metadata : bool, default=True
            Encode categorical metadata features
        encoding_strategy : str, default='onehot'
            Encoding strategy: 'onehot' or 'label'
        validate_mar_index : bool, default=True
            Validate existing MAR index or calculate if missing
        verbose : bool, default=False
            Whether to print feature engineering statistics
        """
        self.create_binary_resistance = create_binary_resistance
        self.create_antibiogram = create_antibiogram
        self.create_aggregates = create_aggregates
        self.create_who_features = create_who_features
        self.encode_metadata = encode_metadata
        self.encoding_strategy = encoding_strategy
        self.validate_mar_index = validate_mar_index
        self.verbose = verbose
        
        self.antibiotics_ = []
        self.metadata_encoders_ = {}
        self.feature_names_ = []
        self.engineering_stats_ = {}
    
    def _identify_antibiotics(self, df):
        """Identify all antibiotics with clean interpretation columns."""
        antibiotics = []
        
        for col in df.columns:
            if '_int_clean' in col:
                antibiotic = col.replace('_int_clean', '')
                antibiotics.append(antibiotic)
        
        return sorted(antibiotics)
    
    def _create_binary_resistance_features(self, X):
        """
        Create binary resistance indicators (0=S/I, 1=R) per antibiotic.
        """
        X_features = X.copy()
        
        for antibiotic in self.antibiotics_:
            sir_col = f"{antibiotic}_int_clean"
            binary_col = f"{antibiotic}_resistant"
            
            if sir_col in X.columns:
                # R=1, S/I/not_tested=0
                X_features[binary_col] = (X[sir_col] == 'r').astype(int)
                self.feature_names_.append(binary_col)
        
        return X_features
    
    def _create_antibiogram_features(self, X):
        """
        Create antibiogram fingerprint as a vector of R/S for all antibiotics.
        This creates a consolidated resistance profile.
        """
        X_features = X.copy()
        
        # Create resistance vector columns
        resistance_values = []
        
        for antibiotic in self.antibiotics_:
            sir_col = f"{antibiotic}_int_clean"
            if sir_col in X.columns:
                resistance_values.append(X[sir_col].map({
                    's': 0,
                    'i': 0.5,
                    'r': 1,
                    'not_tested': -1
                }).fillna(-1))
        
        if resistance_values:
            # Create consolidated antibiogram vector
            antibiogram_df = pd.DataFrame(resistance_values).T
            antibiogram_df.columns = [f"antibiogram_{ab}" for ab in self.antibiotics_]
            
            for col in antibiogram_df.columns:
                X_features[col] = antibiogram_df[col]
                self.feature_names_.append(col)
        
        return X_features
    
    def _create_aggregate_features(self, X):
        """
        Create aggregate resistance features:
        - Total number of resistant antibiotics
        - Number of resistant antibiotic classes
        - Resistance counts per class
        - Resistance ratios
        """
        X_features = X.copy()
        
        # Count total resistances
        resistance_cols = [f"{ab}_int_clean" for ab in self.antibiotics_ 
                          if f"{ab}_int_clean" in X.columns]
        
        if resistance_cols:
            # Total number of resistant antibiotics
            X_features['total_resistant'] = sum(
                (X[col] == 'r').astype(int) for col in resistance_cols
            )
            self.feature_names_.append('total_resistant')
            
            # Total number of susceptible antibiotics
            X_features['total_susceptible'] = sum(
                (X[col] == 's').astype(int) for col in resistance_cols
            )
            self.feature_names_.append('total_susceptible')
            
            # Total number tested (excluding not_tested)
            X_features['total_tested'] = sum(
                (X[col].isin(['s', 'i', 'r'])).astype(int) for col in resistance_cols
            )
            self.feature_names_.append('total_tested')
            
            # Resistance ratio
            X_features['resistance_ratio'] = (
                X_features['total_resistant'] / 
                X_features['total_tested'].replace(0, np.nan)
            ).fillna(0)
            self.feature_names_.append('resistance_ratio')
        
        # Count resistance by antibiotic class
        for class_name, class_antibiotics in self.ANTIBIOTIC_CLASSES.items():
            class_resistance_count = pd.Series(0, index=X.index)
            class_tested_count = pd.Series(0, index=X.index)
            
            for antibiotic in class_antibiotics:
                sir_col = f"{antibiotic}_int_clean"
                if sir_col in X.columns:
                    class_resistance_count += (X[sir_col] == 'r').astype(int)
                    class_tested_count += (X[sir_col].isin(['s', 'i', 'r'])).astype(int)
            
            # Number resistant in class
            col_name = f"{class_name}_resistant_count"
            X_features[col_name] = class_resistance_count
            self.feature_names_.append(col_name)
            
            # Ratio resistant in class
            col_name_ratio = f"{class_name}_resistance_ratio"
            X_features[col_name_ratio] = (
                class_resistance_count / class_tested_count.replace(0, np.nan)
            ).fillna(0)
            self.feature_names_.append(col_name_ratio)
        
        return X_features
    
    def _create_who_priority_features(self, X):
        """
        Create features tracking WHO critical/priority antibiotics.
        """
        X_features = X.copy()
        
        who_resistance_count = pd.Series(0, index=X.index)
        who_tested_count = pd.Series(0, index=X.index)
        
        for antibiotic in self.WHO_CRITICAL_ANTIBIOTICS:
            sir_col = f"{antibiotic}_int_clean"
            if sir_col in X.columns:
                who_resistance_count += (X[sir_col] == 'r').astype(int)
                who_tested_count += (X[sir_col].isin(['s', 'i', 'r'])).astype(int)
        
        X_features['who_priority_resistant_count'] = who_resistance_count
        self.feature_names_.append('who_priority_resistant_count')
        
        X_features['who_priority_resistance_ratio'] = (
            who_resistance_count / who_tested_count.replace(0, np.nan)
        ).fillna(0)
        self.feature_names_.append('who_priority_resistance_ratio')
        
        return X_features
    
    def _validate_or_calculate_mar_index(self, X):
        """
        Validate existing MAR index or calculate if missing.
        
        MAR index = (Number of resistant antibiotics) / (Number of antibiotics tested)
        """
        X_features = X.copy()
        
        if 'mar_index' not in X_features.columns:
            # Calculate MAR index
            if 'total_resistant' in X_features.columns and 'total_tested' in X_features.columns:
                X_features['mar_index'] = (
                    X_features['total_resistant'] / 
                    X_features['total_tested'].replace(0, np.nan)
                ).fillna(0)
        else:
            # Validate existing MAR index
            if 'total_resistant' in X_features.columns and 'total_tested' in X_features.columns:
                calculated_mar = (
                    X_features['total_resistant'] / 
                    X_features['total_tested'].replace(0, np.nan)
                ).fillna(0)
                
                # Check for discrepancies
                discrepancies = np.abs(X_features['mar_index'].fillna(0) - calculated_mar) > 0.01
                if discrepancies.any():
                    if self.verbose:
                        print(f"Warning: {discrepancies.sum()} MAR index discrepancies detected")
                
                # Add validated flag
                X_features['mar_index_validated'] = (~discrepancies).astype(int)
                self.feature_names_.append('mar_index_validated')
        
        return X_features
    
    def _encode_metadata(self, X):
        """
        Encode categorical metadata features.
        """
        X_features = X.copy()
        
        # Metadata columns to encode
        metadata_cols = ['bacterial_species', 'administrative_region', 'national_site',
                        'local_site', 'sample_source', 'esbl']
        
        for col in metadata_cols:
            if col not in X.columns:
                continue
            
            if self.encoding_strategy == 'onehot':
                # One-hot encoding
                if col not in self.metadata_encoders_:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoder.fit(X[[col]].fillna('missing'))
                    self.metadata_encoders_[col] = encoder
                else:
                    encoder = self.metadata_encoders_[col]
                
                encoded = encoder.transform(X[[col]].fillna('missing'))
                encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                
                for i, encoded_col in enumerate(encoded_cols):
                    X_features[encoded_col] = encoded[:, i]
                    self.feature_names_.append(encoded_col)
            
            elif self.encoding_strategy == 'label':
                # Label encoding
                if col not in self.metadata_encoders_:
                    encoder = LabelEncoder()
                    encoder.fit(X[col].fillna('missing'))
                    self.metadata_encoders_[col] = encoder
                else:
                    encoder = self.metadata_encoders_[col]
                
                encoded_col = f"{col}_encoded"
                X_features[encoded_col] = encoder.transform(X[col].fillna('missing'))
                self.feature_names_.append(encoded_col)
        
        # Encode numeric replicate and colony as is
        if 'replicate' in X.columns:
            X_features['replicate_num'] = X['replicate']
            self.feature_names_.append('replicate_num')
        
        if 'colony' in X.columns:
            X_features['colony_num'] = X['colony']
            self.feature_names_.append('colony_num')
        
        return X_features
    
    def fit(self, X, y=None):
        """
        Fit the feature engineer to the data.
        
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
        self.feature_names_ = []
        
        # Fit encoders for metadata
        if self.encode_metadata:
            metadata_cols = ['bacterial_species', 'administrative_region', 'national_site',
                           'local_site', 'sample_source', 'esbl']
            
            for col in metadata_cols:
                if col not in X.columns:
                    continue
                
                if self.encoding_strategy == 'onehot':
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoder.fit(X[[col]].fillna('missing'))
                    self.metadata_encoders_[col] = encoder
                
                elif self.encoding_strategy == 'label':
                    encoder = LabelEncoder()
                    encoder.fit(X[col].fillna('missing'))
                    self.metadata_encoders_[col] = encoder
        
        if self.verbose:
            print(f"Feature engineer fitted for {len(self.antibiotics_)} antibiotics")
        
        return self
    
    def transform(self, X):
        """
        Transform the data by creating derived features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        pd.DataFrame
            Dataframe with engineered features
        """
        X_features = X.copy()
        self.feature_names_ = []
        
        # Create binary resistance features
        if self.create_binary_resistance:
            X_features = self._create_binary_resistance_features(X_features)
        
        # Create aggregate features first (needed for MAR validation)
        if self.create_aggregates:
            X_features = self._create_aggregate_features(X_features)
        
        # Validate or calculate MAR index
        if self.validate_mar_index:
            X_features = self._validate_or_calculate_mar_index(X_features)
        
        # Create antibiogram features
        if self.create_antibiogram:
            X_features = self._create_antibiogram_features(X_features)
        
        # Create WHO priority features
        if self.create_who_features:
            X_features = self._create_who_priority_features(X_features)
        
        # Encode metadata
        if self.encode_metadata:
            X_features = self._encode_metadata(X_features)
        
        if self.verbose:
            print(f"\nCreated {len(self.feature_names_)} new features")
        
        return X_features
    
    def get_feature_names(self):
        """
        Get names of engineered features.
        
        Returns
        -------
        list
            List of feature names
        """
        return self.feature_names_
