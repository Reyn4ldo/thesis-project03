# Preprocessing Module

This module provides comprehensive data cleaning and feature engineering for antibiotic resistance surveillance data.

## Components

### 1. MICSIRCleaner
Cleans and normalizes MIC (Minimum Inhibitory Concentration) values and S/I/R interpretations.

**Features:**
- Normalizes MIC values (handles Unicode operators: ≤, ≥)
- Standardizes S/I/R interpretations to lowercase
- Detects MIC/SIR inconsistencies
- Handles special values (trm, c, asterisks)

**Usage:**
```python
from preprocessing import MICSIRCleaner

cleaner = MICSIRCleaner(add_inconsistency_flags=True, verbose=True)
cleaned_df = cleaner.fit_transform(raw_df)

# Access cleaning statistics
stats = cleaner.get_cleaning_stats()
```

### 2. DomainAwareImputer
Implements domain-aware imputation strategies for missing values.

**Features:**
- MIC imputation: median or KNN strategies
- S/I/R imputation: 'not_tested' category or mode
- Adds indicator flags for imputed values

**Usage:**
```python
from preprocessing import DomainAwareImputer

imputer = DomainAwareImputer(
    mic_strategy='median',
    sir_strategy='not_tested',
    add_indicators=True,
    verbose=True
)
imputed_df = imputer.fit_transform(cleaned_df)

# Access imputation statistics
stats = imputer.get_imputation_stats()
```

### 3. ResistanceFeatureEngineer
Creates derived features for modeling.

**Features:**
- Binary resistance indicators (0/1) per antibiotic
- Antibiogram fingerprints (resistance vectors)
- Aggregate resistance metrics (counts, ratios)
- WHO priority antibiotic tracking
- Metadata encoding (one-hot or label encoding)
- MAR index validation

**Usage:**
```python
from preprocessing import ResistanceFeatureEngineer

engineer = ResistanceFeatureEngineer(
    create_binary_resistance=True,
    create_antibiogram=True,
    create_aggregates=True,
    create_who_features=True,
    encode_metadata=True,
    encoding_strategy='label',
    validate_mar_index=True,
    verbose=True
)
featured_df = engineer.fit_transform(imputed_df)

# Get feature names
feature_names = engineer.get_feature_names()
```

### 4. StratifiedDataSplitter
Creates train/validation/test splits with stratification.

**Features:**
- Stratified splits by species, region, or custom column
- Time-aware splits (if temporal data available)
- Cross-validation fold generation
- Handles rare classes

**Usage:**
```python
from preprocessing import StratifiedDataSplitter

splitter = StratifiedDataSplitter(
    stratify_by='bacterial_species',
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    verbose=True
)

train_df, val_df, test_df = splitter.split(featured_df)

# Get indices
train_idx, val_idx, test_idx = splitter.get_indices()

# Create cross-validation folds
folds = splitter.create_cv_folds(featured_df, n_folds=5)
```

### 5. Complete Pipeline
Combines all components into a reusable scikit-learn Pipeline.

**Usage:**
```python
from preprocessing import create_preprocessing_pipeline
from preprocessing.pipeline import PreprocessingPipelineWrapper

# Create pipeline with custom configuration
pipeline_wrapper = PreprocessingPipelineWrapper(config={
    'mic_strategy': 'median',
    'sir_strategy': 'not_tested',
    'encoding_strategy': 'label',
    'add_inconsistency_flags': True,
    'add_imputation_indicators': True,
    'validate_mar_index': True,
    'verbose': True
})

# Fit and transform
processed_df = pipeline_wrapper.fit_transform(raw_df)

# Save pipeline for later use
pipeline_wrapper.save('my_pipeline')

# Load saved pipeline
loaded_pipeline = PreprocessingPipelineWrapper.load('my_pipeline')
new_processed_df = loaded_pipeline.transform(new_raw_df)
```

## Quick Start

```python
import pandas as pd
from preprocessing.pipeline import PreprocessingPipelineWrapper

# Load raw data
df = pd.read_csv('raw - data.csv')

# Create and fit pipeline
pipeline = PreprocessingPipelineWrapper(config={
    'verbose': True
})
processed_df = pipeline.fit_transform(df)

# Save processed data
processed_df.to_csv('processed_data.csv', index=False)

# Save pipeline for inference
pipeline.save('preprocessing_pipeline')
```

## Running the Complete Phase 1 Pipeline

```bash
python phase1_preprocessing.py
```

This will:
1. Load the raw data
2. Clean MIC and S/I/R values
3. Impute missing values
4. Engineer features
5. Create train/validation/test splits
6. Save all outputs and the fitted pipeline

## Testing

Run the test suite to validate the preprocessing pipeline:

```bash
python tests/test_preprocessing.py
```

## Output Files

After running `phase1_preprocessing.py`, the following files are generated:

- `processed_data.csv` - Fully processed dataset with all engineered features
- `train_data.csv` - Training set (stratified)
- `val_data.csv` - Validation set (stratified)
- `test_data.csv` - Test set (stratified)
- `preprocessing_pipeline.pkl` - Fitted pipeline (can be reloaded)
- `preprocessing_pipeline.json` - Pipeline configuration
- `PHASE1_SUMMARY.md` - Summary report

## Features Created

The pipeline creates 232 new features including:

1. **Cleaned Values** (per antibiotic):
   - `{antibiotic}_mic_clean` - Normalized MIC value
   - `{antibiotic}_mic_operator` - Comparison operator (<=, >=, =)
   - `{antibiotic}_mic_numeric` - Numeric MIC value
   - `{antibiotic}_int_clean` - Standardized S/I/R interpretation
   - `{antibiotic}_mic_sir_inconsistent` - Inconsistency flag

2. **Imputation Indicators** (per antibiotic):
   - `{antibiotic}_mic_imputed` - Binary flag (1 if imputed)
   - `{antibiotic}_sir_imputed` - Binary flag (1 if imputed)

3. **Binary Resistance** (per antibiotic):
   - `{antibiotic}_resistant` - Binary (1=R, 0=S/I)

4. **Antibiogram Fingerprints** (per antibiotic):
   - `antibiogram_{antibiotic}` - Resistance score (-1=not tested, 0=S, 0.5=I, 1=R)

5. **Aggregate Features**:
   - `total_resistant` - Total resistant antibiotics
   - `total_susceptible` - Total susceptible antibiotics
   - `total_tested` - Total antibiotics tested
   - `resistance_ratio` - Proportion resistant
   - `{class}_resistant_count` - Per antibiotic class counts
   - `{class}_resistance_ratio` - Per antibiotic class ratios

6. **WHO Priority Features**:
   - `who_priority_resistant_count` - Resistance to WHO critical antibiotics
   - `who_priority_resistance_ratio` - Ratio of WHO critical resistance

7. **Encoded Metadata**:
   - `bacterial_species_encoded` - Encoded species
   - `administrative_region_encoded` - Encoded region
   - `sample_source_encoded` - Encoded sample source
   - `local_site_encoded` - Encoded site
   - `esbl_encoded` - Encoded ESBL status

8. **Validated Metrics**:
   - `mar_index_validated` - Binary flag indicating MAR index validity

## Antibiotic Classes

The pipeline tracks resistance by antibiotic class:

- **Beta-lactams**: ampicillin, amoxicillin/clavulanic acid, ceftaroline, cefalexin, cefalotin, cefpodoxime, cefotaxime, cefovecin, ceftiofur, ceftazidime/avibactam, imepenem
- **Aminoglycosides**: amikacin, gentamicin, neomycin
- **Fluoroquinolones**: nalidixic acid, enrofloxacin, marbofloxacin, pradofloxacin
- **Tetracyclines**: doxycycline, tetracycline
- **Others**: nitrofurantoin, chloramphenicol, trimethoprim/sulfamethazole

## WHO Critical Antibiotics

The pipeline tracks resistance to WHO priority antibiotics:
- ceftaroline, cefotaxime, ceftazidime/avibactam, imepenem, amikacin, gentamicin, enrofloxacin, marbofloxacin

## Data Validation

The test suite validates:
- MIC value ranges and formats
- S/I/R label consistency
- Resistance ratio ranges [0, 1]
- MAR index ranges [0, 1.5]
- Data leak detection in splits
- Feature consistency
- Pipeline reproducibility
