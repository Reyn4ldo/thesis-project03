# Phase 0 — Initial Setup & Data Understanding

## Overview
This repository contains the deliverables for Phase 0 of the antibiotic resistance surveillance thesis project. The objective is to gain comprehensive understanding of the dataset, confirm data schema, and prepare for subsequent analysis phases.

## Dataset Summary
- **Total isolates**: 583 bacterial samples
- **Total fields**: 58 columns
- **Unique species**: 13 bacterial species
- **Sample sources**: 9 different environmental and biological sources
- **Geographic coverage**: 3 administrative regions in the Philippines

## Deliverables

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

## Contact and Citation

For questions about this analysis or access to additional data, please contact the project maintainers.

---

**Last Updated**: December 8, 2024
**Analysis Script Version**: 1.0
**Dataset**: raw - data.csv (583 isolates)
