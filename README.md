# Antibiotic Resistance Surveillance Thesis Project

## Overview
This repository contains the implementation of a comprehensive antibiotic resistance surveillance analysis pipeline. The project processes bacterial isolate data from environmental and aquaculture sources in the Philippines to analyze resistance patterns and develop predictive models.

## Project Structure

```
thesis-project03/
├── README.md                      # This file
├── data_dictionary.json           # Data schema documentation
├── raw - data.csv                 # Original dataset (583 isolates)
├── sample_data.csv                # Sample subset (50 isolates)
├── preprocessing/                 # Phase 1: Preprocessing module
│   ├── __init__.py
│   ├── mic_sir_cleaner.py        # MIC/SIR cleaning
│   ├── imputer.py                # Missing value imputation
│   ├── feature_engineer.py       # Feature engineering
│   ├── data_splitter.py          # Data splitting
│   ├── pipeline.py               # Complete pipeline
│   └── README.md                 # Module documentation
├── tests/                         # Unit tests
│   └── test_preprocessing.py     # Preprocessing tests
├── phase0_data_analysis.py        # Phase 0: Data understanding
├── phase1_preprocessing.py        # Phase 1: Preprocessing pipeline
├── PHASE0_SUMMARY.md             # Phase 0 summary
├── PHASE1_SUMMARY.md             # Phase 1 summary
└── .gitignore
```

## Dataset Summary
- **Total isolates**: 583 bacterial samples
- **Original fields**: 58 columns
- **Processed fields**: 290 columns (after feature engineering)
- **Unique species**: 13 bacterial species
- **Sample sources**: 9 different environmental and biological sources
- **Geographic coverage**: 3 administrative regions in the Philippines
- **Antibiotics tested**: 23 antibiotics across 5 classes

---

## Phase 0 — Initial Setup & Data Understanding

### Objectives ✅ Complete
- Gain access to representative data sample
- Confirm data schema and labels
- Document missingness patterns
- Validate metadata fields
- Assess data quality

### Key Deliverables

### 1. Data Dictionary (`data_dictionary.json`)
A comprehensive JSON file documenting:
- Dataset metadata (row count, column count, description)
- Detailed information for each column including:
  - Data type
  - Completeness (null vs non-null counts)
  - Category (metadata, MIC value, S/I/R interpretation, outcome)
  - Description and meaning
  - Unique value distributions (for categorical fields)

**Key Insights from Data Dictionary:**
- **23 antibiotics** with paired MIC (Minimum Inhibitory Concentration) and interpretation (S/I/R) columns
- Antibiotic classes include: β-lactams, aminoglycosides, fluoroquinolones, tetracyclines, and others
- Three outcome metrics: `scored_resistance`, `num_antibiotics_tested`, and `mar_index`

### 2. Sanity-Check Report (`sanity_check_report.txt`)
A detailed text report covering:

#### Missingness Analysis
- 51 out of 58 fields have some missing data
- Most critical: No time/date fields for temporal trend analysis
- Some antibiotic MIC values are 100% missing (ceftaroline, cefotaxime, ceftazidime/avibactam, nalidixic acid)
- ESBL status missing for 31% of isolates

#### Species Distribution
Top 5 bacterial species:
1. *Escherichia coli* (235 isolates, 40.3%)
2. *Klebsiella pneumoniae* ssp. *pneumoniae* (158 isolates, 27.1%)
3. *Enterobacter cloacae* complex (70 isolates, 12.0%)
4. *Pseudomonas aeruginosa* (38 isolates, 6.5%)
5. *Enterobacter aerogenes* (24 isolates, 4.1%)

**Note**: Species names follow hierarchical taxonomy with subspecies designations where applicable.

#### Label Balance (S/I/R)
Overall resistance profile across all antibiotics:
- **Susceptible (S)**: 9,487 observations (87.8%)
- **Intermediate (I)**: 333 observations (3.1%)
- **Resistant (R)**: 986 observations (9.1%)

This indicates a generally susceptible population with significant resistance present.

#### Antibiotic Testing Consistency
**Consistently tested antibiotics (≥90% coverage)**:
- Imipenem (90.22%)
- Gentamicin (90.22%)
- Marbofloxacin (90.05%)

**Moderately tested antibiotics (50-90% coverage)**: 19 antibiotics
**Rarely tested antibiotics (<50% coverage)**: Nalidixic acid (0%)

#### MAR Index Statistics
The Multiple Antibiotic Resistance (MAR) index distribution:
- **Mean**: 0.1165
- **Median**: 0.0909
- **Range**: 0.0000 - 1.5000
- **Low risk (≤0.2)**: 458 isolates (78.6%)
- **Medium/High risk (>0.2)**: 80 isolates (13.7%)

*Note*: MAR index > 0.2 suggests high-risk contamination sources or frequent antibiotic exposure.

#### Sample Source Distribution
- Fish (tilapia, gusaw, kaolang, banak): 287 isolates (49.2%)
- Water (drinking, river, lake, effluent): 296 isolates (50.8%)
  - Drinking water: 87 isolates
  - River water: 70 isolates
  - Effluent water (untreated): 71 isolates
  - Effluent water (treated): 17 isolates
  - Lake water: 51 isolates

### 3. Sample Dataset (`sample_data.csv`)
A representative sample of 50 isolates selected using stratified random sampling to ensure:
- Coverage of multiple bacterial species
- Representation from different sample sources
- Diverse geographic locations
- Range of resistance profiles

This sample can be used for:
- Method development and testing
- Quick exploratory analyses
- Algorithm prototyping
- Sharing with collaborators without full dataset disclosure

## Key Checks Completed ✓

### Schema and Labels
✓ **Data schema confirmed**: All expected fields present and documented
✓ **S/I/R labels validated**: Labels follow standard conventions (s/i/r format)
✓ **MIC values documented**: Numeric and comparative operators (≤, ≥) present
✓ **Metadata fields present**: Species, region, site, sample source, replicate information

### Data Quality
✓ **Missingness patterns analyzed**: Documented in data dictionary and report
✓ **Species taxonomy reviewed**: Hierarchical naming with subspecies designations
✓ **MAR index calculated**: Valid range (0-1.5), mathematically consistent
✓ **Antibiotic coverage assessed**: Identified core vs. supplementary antibiotics

### Important Findings

#### ⚠️ Critical Observations

1. **No Time/Date Fields**
   - **Impact**: Limits ability to perform temporal trend analyses
   - **Recommendation**: If collection dates are available in source records, add them to enable:
     - Time-series analysis of resistance trends
     - Seasonal pattern detection
     - Before/after intervention studies

2. **Incomplete Antibiotic Testing**
   - Only 3 antibiotics tested consistently across >90% of isolates
   - Many antibiotics tested on <85% of samples
   - **Recommendation**: Document testing protocols and reasons for selective testing

3. **Missing ESBL Status**
   - ESBL phenotype missing for 31% of isolates
   - **Impact**: Important predictor of β-lactam resistance
   - **Recommendation**: Prioritize ESBL testing completion where possible

#### ✓ Positive Findings

1. **Good Overall Data Quality**
   - Core metadata fields nearly complete
   - Consistent naming conventions
   - Standardized S/I/R interpretations

2. **Diverse Sample Representation**
   - Multiple species, sources, and geographic regions
   - Balanced between aquatic and water sources
   - Sufficient sample sizes for major categories

3. **Calculated Metrics Available**
   - MAR index pre-calculated and validated
   - Scored resistance counts available
   - Number of antibiotics tested tracked

#### 📝 Data Quality Notes

1. **Antibiotic Naming**
   - The source data contains "imepenem" which is likely a misspelling of "imipenem" (carbapenem antibiotic)
   - All references maintained as-is to match source data schema
   - Future data cleaning phases should consider standardizing antibiotic nomenclature

## Technical Details

### Antibiotics in Dataset (23 total)

**β-lactams (8)**:
- Ampicillin
- Amoxicillin/clavulanic acid
- Ceftaroline
- Cefalexin
- Cefalotin
- Cefpodoxime
- Cefotaxime
- Cefovecin
- Ceftiofur
- Ceftazidime/avibactam
- Imipenem (Note: spelled "imepenem" in source data)

**Aminoglycosides (3)**:
- Amikacin
- Gentamicin
- Neomycin

**Fluoroquinolones (4)**:
- Nalidixic acid
- Enrofloxacin
- Marbofloxacin
- Pradofloxacin

**Tetracyclines (2)**:
- Doxycycline
- Tetracycline

**Others (3)**:
- Nitrofurantoin
- Chloramphenicol
- Trimethoprim/sulfamethazole

### Geographic Coverage

**Regions**:
1. BARMM (309 isolates, 53.0%)
2. Region III - Central Luzon (153 isolates, 26.2%)
3. Region VIII - Eastern Visayas (121 isolates, 20.8%)

**Sites**:
- Marawi City (BARMM): 309 isolates
- Pampanga (Central Luzon): 153 isolates
- Ormoc (Eastern Visayas): 121 isolates

## Usage Instructions

### Running the Analysis

```bash
# Install required packages
pip install pandas numpy

# Run the analysis script
python3 phase0_data_analysis.py
```

This will regenerate all three deliverables:
1. `data_dictionary.json`
2. `sanity_check_report.txt`
3. `sample_data.csv`

### Viewing Results

```bash
# View sanity check report
cat sanity_check_report.txt

# Explore data dictionary
python3 -m json.tool data_dictionary.json | less

# Load sample data
python3
>>> import pandas as pd
>>> df = pd.read_csv('sample_data.csv')
>>> df.info()
```

## Recommendations for Next Phases

### Phase 1 - Data Cleaning and Preparation
1. **Handle missing data**:
   - Implement appropriate strategies for missing MIC/interpretations
   - Consider multiple imputation for critical antibiotics
   - Document all data cleaning decisions

2. **Add temporal context**:
   - Attempt to recover collection dates from source records
   - Create temporal proxy variables if dates unavailable

3. **Feature engineering**:
   - Create antibiotic class resistance indicators
   - Derive additional resistance metrics
   - Generate geographic/source interaction features

### Phase 2 - Exploratory Analysis
1. **Resistance pattern analysis**:
   - Co-resistance networks
   - Species-specific resistance profiles
   - Source contamination patterns

2. **Statistical testing**:
   - Compare resistance across regions
   - Test associations with sample sources
   - Analyze MAR index distributions

### Phase 3 - Predictive Modeling
1. **Potential targets**:
   - Predict resistance patterns from metadata
   - Classify high-risk isolates (MAR index > 0.2)
   - Forecast resistance likelihood by antibiotic class

2. **Consider missing data implications**:
   - Use only consistently tested antibiotics for core models
   - Develop separate models for subsets with complete data
   - Implement multiple imputation strategies

---

## Phase 1 — Data Cleaning & Feature Engineering

### Objectives ✅ Complete
- Create robust, reproducible preprocessing pipelines
- Normalize and clean MIC/SIR values
- Impute missing values with domain-aware strategies
- Engineer comprehensive feature set for modeling
- Create stratified train/validation/test splits
- Implement data validation tests

### Key Deliverables

1. **Preprocessing Module** (`preprocessing/`)
   - Reusable, modular components
   - scikit-learn compatible transformers
   - Save/load functionality for reproducibility

2. **Processed Datasets**
   - `processed_data.csv` - Full processed dataset (290 columns)
   - `train_data.csv` - Training set (407 isolates, 69.8%)
   - `val_data.csv` - Validation set (59 isolates, 10.1%)
   - `test_data.csv` - Test set (117 isolates, 20.1%)

3. **Saved Pipeline**
   - `preprocessing_pipeline.pkl` - Fitted pipeline
   - `preprocessing_pipeline.json` - Configuration

4. **Data Validation Tests**
   - 19 unit tests covering all components
   - Range validation, consistency checks
   - Data leak detection

### Features Implemented

#### 1. Data Cleaning
- **MIC Normalization**: Unicode operators (≤, ≥) standardized to ASCII (<=, >=)
- **S/I/R Standardization**: Cleaned to lowercase, removed asterisks
- **Inconsistency Detection**: Flags potential MIC/SIR mismatches
- **Special Value Handling**: Invalid values (trm, c) handled appropriately

#### 2. Missing Value Imputation
- **MIC Values**: Median or KNN imputation strategies
- **S/I/R Values**: Domain-aware 'not_tested' category
- **Indicator Flags**: Binary flags track which values were imputed
- **Statistics**: 1,480 MIC values and 2,433 S/I/R values imputed

#### 3. Feature Engineering
Created **232 new features** including:

- **Binary Resistance Indicators** (24 features)
  - `{antibiotic}_resistant`: 0/1 encoding per antibiotic

- **Antibiogram Fingerprints** (23 features)
  - Resistance vectors: -1 (not tested), 0 (S), 0.5 (I), 1 (R)

- **Aggregate Metrics** (16 features)
  - Total resistant/susceptible counts
  - Resistance ratios overall and per antibiotic class
  - Class-specific resistance counts (β-lactams, aminoglycosides, etc.)

- **WHO Priority Tracking** (2 features)
  - Resistance to WHO critical antibiotics
  - Ratio of WHO priority resistance

- **Metadata Encoding** (6 features)
  - Species, region, site, source encoded
  - Label encoding for efficiency

- **MAR Index Validation**
  - Validation flag for existing MAR calculations
  - Detection of 23 discrepancies

#### 4. Data Splitting
- **Stratification**: By bacterial species
- **No Data Leakage**: Verified through unit tests
- **Balanced Splits**: Maintains species distribution across sets
- **Reproducible**: Fixed random seed (42)

### Usage

#### Quick Start
```bash
# Install dependencies
pip install pandas numpy scikit-learn joblib

# Run complete pipeline
python phase1_preprocessing.py
```

#### Programmatic Usage
```python
from preprocessing.pipeline import PreprocessingPipelineWrapper

# Create and fit pipeline
pipeline = PreprocessingPipelineWrapper(config={'verbose': True})
processed_df = pipeline.fit_transform(raw_df)

# Save for later use
pipeline.save('my_pipeline')

# Load and reuse
loaded_pipeline = PreprocessingPipelineWrapper.load('my_pipeline')
new_processed_df = loaded_pipeline.transform(new_data)
```

#### Running Tests
```bash
python tests/test_preprocessing.py
```

### Data Quality Improvements

✅ **MIC Values**: 10,524 values cleaned and normalized  
✅ **S/I/R Labels**: 11,806 values standardized  
✅ **Inconsistencies**: 225 potential MIC/SIR inconsistencies flagged  
✅ **Missing Data**: Comprehensive imputation with transparency flags  
✅ **Features**: 232 engineered features ready for modeling  
✅ **Validation**: 19 unit tests ensure data quality  

### Key Statistics

- **Imputation Coverage**: 38.8% of MIC values, 100% of missing S/I/R
- **Feature Expansion**: 58 → 290 columns (5x increase)
- **Data Splits**: Stratified to preserve species distribution
- **MAR Discrepancies**: 23 cases identified for review

### Next Steps → Phase 2

With clean, engineered data ready:
- Exploratory data analysis
- Resistance pattern visualization
- Co-resistance network analysis
- Statistical hypothesis testing
- Species-specific profiling

---

## Technical Details

### Antibiotics in Dataset (23 total)

**β-lactams (11)**:
- Ampicillin
- Amoxicillin/clavulanic acid
- Ceftaroline
- Cefalexin
- Cefalotin
- Cefpodoxime
- Cefotaxime
- Cefovecin
- Ceftiofur
- Ceftazidime/avibactam
- Imipenem (Note: spelled "imepenem" in source data)

**Aminoglycosides (3)**:
- Amikacin
- Gentamicin
- Neomycin

**Fluoroquinolones (4)**:
- Nalidixic acid
- Enrofloxacin
- Marbofloxacin
- Pradofloxacin

**Tetracyclines (2)**:
- Doxycycline
- Tetracycline

**Others (3)**:
- Nitrofurantoin
- Chloramphenicol
- Trimethoprim/sulfamethazole

### Geographic Coverage

**Regions**:
1. BARMM (309 isolates, 53.0%)
2. Region III - Central Luzon (153 isolates, 26.2%)
3. Region VIII - Eastern Visayas (121 isolates, 20.8%)

**Sites**:
- Marawi City (BARMM): 309 isolates
- Pampanga (Central Luzon): 153 isolates
- Ormoc (Eastern Visayas): 121 isolates

## Installation and Setup

### Requirements
```bash
pip install pandas numpy scikit-learn joblib
```

### Running Phase 0 (Data Understanding)
```bash
python phase0_data_analysis.py
```

Generates:
- `data_dictionary.json` - Schema documentation
- `sanity_check_report.txt` - Quality assessment
- `sample_data.csv` - Representative sample

### Running Phase 1 (Preprocessing)
```bash
python phase1_preprocessing.py
```

Generates:
- `processed_data.csv` - Fully processed dataset
- `train_data.csv`, `val_data.csv`, `test_data.csv` - Data splits
- `preprocessing_pipeline.pkl/.json` - Saved pipeline
- `PHASE1_SUMMARY.md` - Processing report

## Contact and Citation

For questions about this analysis or access to additional data, please contact the project maintainers.

---

**Last Updated**: December 8, 2024  
**Phase 0 Version**: 1.0  
**Phase 1 Version**: 1.0  
**Dataset**: raw - data.csv (583 isolates)
