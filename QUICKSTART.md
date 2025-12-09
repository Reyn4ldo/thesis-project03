# Quick Start Guide

Get up and running with the Antibiotic Resistance Surveillance Project in under 10 minutes!

## Prerequisites

- Python 3.8+ installed
- 4GB RAM minimum
- 2GB free disk space

## Quick Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Reyn4ldo/thesis-project03.git
cd thesis-project03

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install essential dependencies
pip install pandas numpy scikit-learn scipy joblib matplotlib seaborn
```

### 2. Run Basic Analysis

```bash
# Phase 0: Understand the data (30 seconds)
python phase0_data_analysis.py

# Phase 1: Preprocess data (1 minute)
python phase1_preprocessing.py
```

That's it! You've processed the data and are ready for advanced analysis.

## Quick Usage Examples

### Example 1: Data Understanding

```bash
# Generate data quality reports
python phase0_data_analysis.py

# View the report
cat sanity_check_report.txt

# Check sample data
head sample_data.csv
```

### Example 2: Train Machine Learning Models

```bash
# Install ML dependencies first
pip install mlflow xgboost

# Run supervised learning experiments
python phase2_supervised_learning.py

# View results in MLflow UI
mlflow ui
# Open http://localhost:5000 in browser
```

### Example 3: Deploy REST API

```bash
# Install API dependencies
pip install fastapi uvicorn pydantic

# Start the API server
uvicorn operationalization.api:app --reload

# Test the API
curl http://localhost:8000/health

# View interactive docs at http://localhost:8000/docs
```

## Docker Quick Start

If you prefer Docker:

```bash
# Clone repository
git clone https://github.com/Reyn4ldo/thesis-project03.git
cd thesis-project03

# Build and run with Docker Compose
docker-compose up -d

# Check API health
curl http://localhost:8000/health

# View logs
docker-compose logs -f
```

## Project Phases Overview

Run phases in order based on your needs:

| Phase | Command | Duration | Purpose |
|-------|---------|----------|---------|
| **Phase 0** | `python phase0_data_analysis.py` | 30s | Data understanding |
| **Phase 1** | `python phase1_preprocessing.py` | 1m | Data cleaning & feature engineering |
| **Phase 2** | `python phase2_supervised_learning.py` | 5-10m | Train ML models |
| **Phase 3** | `python phase3_exploratory_analysis.py` | 2-3m | Clustering & association rules |
| **Phase 4** | `python phase4_anomaly_detection.py` | 1-2m | Outlier detection |
| **Phase 5** | `python phase5_spatiotemporal.py` | 2-3m | Geographic & temporal analysis |
| **Phase 6** | `python phase6_operationalization.py` | 1m | Deploy production tools |

## What Gets Generated?

### After Phase 0:
- `data_dictionary.json` - Data schema
- `sanity_check_report.txt` - Quality assessment
- `sample_data.csv` - Sample dataset

### After Phase 1:
- `processed_data.csv` - Cleaned data (290 features)
- `train_data.csv`, `val_data.csv`, `test_data.csv` - Data splits
- `preprocessing_pipeline.pkl` - Reusable pipeline

### After Phase 2:
- `models/` - Trained ML models
- MLflow tracking data
- Model evaluation reports

### After Phase 6:
- REST API on port 8000
- Automated batch processing
- Antibiogram reports
- Early warning alerts

## Quick Code Examples

### Use Preprocessing Pipeline

```python
from preprocessing import PreprocessingPipelineWrapper
import pandas as pd

# Load and preprocess data
pipeline = PreprocessingPipelineWrapper()
df = pd.read_csv('raw - data.csv')
processed_df = pipeline.fit_transform(df)

# Save for reuse
pipeline.save('my_pipeline')
```

### Make Predictions

```python
from joblib import load
import numpy as np

# Load a trained model
model = load('models/esbl_classifier_rf.pkl')

# Make prediction
features = np.array([[1, 0, 1, 0, ...]]).reshape(1, -1)
prediction = model.predict(features)
probability = model.predict_proba(features)
```

### Generate Antibiogram

```python
from operationalization import AntibiogramGenerator
import pandas as pd

# Load data
df = pd.read_csv('processed_data.csv')

# Generate antibiogram
generator = AntibiogramGenerator(min_isolates=30)
antibiogram = generator.generate_antibiogram(
    df, 
    species='Escherichia coli',
    site='Hospital_A'
)

# Visualize
generator.plot_antibiogram(antibiogram)
```

### Use REST API

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "isolate": {
      "features": {"feature1": 1.0, "feature2": 0.5}
    },
    "model_name": "esbl_classifier"
  }'

# Get therapy recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "species": "Escherichia coli",
    "source": "urine"
  }'
```

## Minimal Installation (Just API)

If you only need the API:

```bash
# Clone repository
git clone https://github.com/Reyn4ldo/thesis-project03.git
cd thesis-project03

# Install minimal dependencies
pip install -r requirements_api.txt

# Start API
uvicorn operationalization.api:app --host 0.0.0.0 --port 8000
```

## Running Specific Analyses

### Only Data Preprocessing

```bash
pip install pandas numpy scikit-learn scipy joblib
python phase0_data_analysis.py
python phase1_preprocessing.py
```

### Only Machine Learning

```bash
# Requires Phase 1 to be completed first
pip install pandas numpy scikit-learn mlflow xgboost
python phase2_supervised_learning.py
```

### Only Visualization & Exploration

```bash
# Requires Phase 1 to be completed first
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend networkx
python phase3_exploratory_analysis.py
```

## Common Tasks

### View Data Quality Report

```bash
python phase0_data_analysis.py
cat sanity_check_report.txt
```

### Check Processed Data

```bash
python phase1_preprocessing.py
python -c "import pandas as pd; df = pd.read_csv('processed_data.csv'); print(df.info())"
```

### Run Tests

```bash
python tests/test_preprocessing.py
# Expected: All 19 tests pass ✅
```

### View Model Performance

```bash
python phase2_supervised_learning.py
mlflow ui
# Open http://localhost:5000
```

## Troubleshooting

### Issue: Module not found

```bash
# Install missing package
pip install <package-name>
```

### Issue: File not found

```bash
# Make sure you're in project root
cd thesis-project03
ls -lh "raw - data.csv"  # Should exist
```

### Issue: Out of memory

```bash
# Use sample data instead
python phase0_data_analysis.py  # Generates sample_data.csv
# Then modify phase scripts to use sample_data.csv
```

### Issue: Port 8000 in use

```bash
# Use different port
uvicorn operationalization.api:app --port 8001
```

## Next Steps

1. ✅ You've completed quick start!
2. 📖 Read full [INSTALLATION.md](INSTALLATION.md) for detailed setup
3. 📚 Explore phase-specific READMEs in module directories
4. 🔬 Review phase completion summaries (PHASE*_COMPLETE.md)
5. 🚀 Deploy to production with Docker

## Getting Help

- **Detailed Installation**: See [INSTALLATION.md](INSTALLATION.md)
- **Project Overview**: See [README.md](README.md)
- **Module Documentation**: Check `<module>/README.md` files
- **Phase Results**: Review `PHASE*_COMPLETE.md` files

## Project Structure Quick Reference

```
thesis-project03/
├── phase0_data_analysis.py          # Data understanding
├── phase1_preprocessing.py          # Data cleaning
├── phase2_supervised_learning.py    # ML training
├── phase3_exploratory_analysis.py   # Clustering, rules
├── phase4_anomaly_detection.py      # Outlier detection
├── phase5_spatiotemporal.py         # Geographic/temporal
├── phase6_operationalization.py     # Production deployment
├── preprocessing/                   # Phase 1 modules
├── experiments/                     # Phase 2 modules
├── exploratory/                     # Phase 3 modules
├── anomaly/                         # Phase 4 modules
├── spatiotemporal/                  # Phase 5 modules
├── operationalization/              # Phase 6 modules (API)
├── tests/                           # Unit tests
├── Dockerfile                       # Docker build
├── docker-compose.yml               # Docker orchestration
└── requirements_api.txt             # API dependencies
```

---

**Ready to dive deeper?** Check out the [INSTALLATION.md](INSTALLATION.md) for comprehensive setup instructions and troubleshooting.

**Last Updated:** December 2024
