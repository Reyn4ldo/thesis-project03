# Installation Guide - Antibiotic Resistance Surveillance Project

This guide provides step-by-step instructions for setting up and running the Antibiotic Resistance Surveillance Thesis Project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Option 1: Local Installation (Recommended for Development)](#option-1-local-installation-recommended-for-development)
  - [Option 2: Docker Installation (Recommended for Production)](#option-2-docker-installation-recommended-for-production)
- [Running the Project](#running-the-project)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL2 for Docker)
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: At least 2GB free space
- **Python**: Version 3.8 or higher (3.9 recommended)
- **Git**: For cloning the repository

### For Docker Installation (Option 2)

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 1.29 or higher

---

## Installation

### Option 1: Local Installation (Recommended for Development)

This option installs all dependencies directly on your system.

#### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Reyn4ldo/thesis-project03.git

# Navigate to the project directory
cd thesis-project03
```

#### Step 2: Create a Virtual Environment (Recommended)

Using a virtual environment helps isolate project dependencies.

**For Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**For Windows:**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### Step 3: Install Core Dependencies

The project uses different dependency sets for different purposes:

**For basic analysis (Phases 0-1):**
```bash
pip install pandas numpy scikit-learn scipy joblib
```

**For supervised learning (Phase 2):**
```bash
pip install pandas numpy scikit-learn scipy joblib mlflow xgboost
```

**For visualization (Phases 3-5):**
```bash
pip install pandas numpy scikit-learn scipy joblib matplotlib seaborn
```

**For advanced analysis (Phases 3-5):**
```bash
pip install pandas numpy scikit-learn scipy joblib matplotlib seaborn mlxtend networkx python-louvain
```

**For API deployment (Phase 6):**
```bash
pip install -r requirements_api.txt
```

**Install ALL dependencies (Complete Installation):**
```bash
# Core dependencies
pip install pandas==2.1.3 numpy==1.24.3 scikit-learn==1.3.2 scipy==1.11.4 joblib==1.3.2

# Machine learning
pip install xgboost==2.0.2

# Visualization
pip install matplotlib==3.8.2 seaborn==0.13.0

# Experiment tracking
pip install mlflow==2.22.4

# Association rules and network analysis
pip install mlxtend==0.23.0 networkx==3.2.1 python-louvain==0.16

# API (for Phase 6)
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.5.0 python-multipart==0.0.6 requests==2.31.0
```

#### Step 4: Verify Installation

```bash
# Test Python environment
python --version

# Test if key packages are installed
python -c "import pandas, numpy, sklearn; print('Core packages installed successfully!')"
```

---

### Option 2: Docker Installation (Recommended for Production)

This option uses Docker containers for easy deployment.

#### Step 1: Install Docker and Docker Compose

**For Linux:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

**For macOS:**
- Download and install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
- Docker Compose is included with Docker Desktop

**For Windows:**
- Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
- Docker Compose is included with Docker Desktop

#### Step 2: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Reyn4ldo/thesis-project03.git

# Navigate to the project directory
cd thesis-project03
```

#### Step 3: Build Docker Image

```bash
# Build the Docker image
docker build -t amr-surveillance .

# This may take 5-10 minutes on first build
```

#### Step 4: Start Services with Docker Compose

```bash
# Start all services (API + batch worker)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Running the Project

### Phase-by-Phase Execution

The project consists of 6 phases. You can run each phase independently or sequentially.

#### Phase 0: Data Understanding

This phase analyzes the dataset and generates data quality reports.

```bash
python phase0_data_analysis.py
```

**Outputs:**
- `data_dictionary.json` - Schema documentation
- `sanity_check_report.txt` - Data quality assessment
- `sample_data.csv` - Representative sample (50 isolates)

**Duration:** ~30 seconds

---

#### Phase 1: Data Cleaning & Feature Engineering

This phase preprocesses the data and creates engineered features.

```bash
python phase1_preprocessing.py
```

**Outputs:**
- `processed_data.csv` - Full processed dataset (290 columns)
- `train_data.csv` - Training set (407 isolates, 69.8%)
- `val_data.csv` - Validation set (59 isolates, 10.1%)
- `test_data.csv` - Test set (117 isolates, 20.1%)
- `preprocessing_pipeline.pkl` - Saved pipeline
- `PHASE1_SUMMARY.md` - Processing report

**Duration:** ~1 minute

---

#### Phase 2: Supervised Learning

This phase trains and evaluates machine learning models.

```bash
python phase2_supervised_learning.py
```

**Outputs:**
- `models/` directory with trained models
- MLflow experiment tracking
- `PHASE2_COMPLETE.md` - Results summary

**Optional: View MLflow results:**
```bash
mlflow ui
# Then open http://localhost:5000 in your browser
```

**Duration:** ~5-10 minutes (depending on hardware)

---

#### Phase 3: Exploratory Analysis

This phase performs clustering, dimensionality reduction, and association rule mining.

```bash
python phase3_exploratory_analysis.py
```

**Outputs:**
- Cluster analysis reports
- Association rules
- Co-resistance networks
- Visualizations

**Duration:** ~2-3 minutes

---

#### Phase 4: Anomaly Detection

This phase identifies outliers and anomalous resistance patterns.

```bash
python phase4_anomaly_detection.py
```

**Outputs:**
- Anomaly scores
- Triage recommendations
- Outlier reports

**Duration:** ~1-2 minutes

---

#### Phase 5: Spatio-temporal Analysis

This phase analyzes geographic and temporal patterns.

```bash
python phase5_spatiotemporal.py
```

**Outputs:**
- Spatial hotspot reports
- Source attribution analysis
- Comprehensive visualizations

**Duration:** ~2-3 minutes

---

#### Phase 6: Operationalization

This phase sets up production-ready APIs and batch processing.

**Run Demonstration:**
```bash
python phase6_operationalization.py
```

**Start REST API:**
```bash
uvicorn operationalization.api:app --host 0.0.0.0 --port 8000 --reload
```

**Access API Documentation:**
- Open http://localhost:8000/docs in your browser

**Run Batch Pipeline:**
```bash
python operationalization/batch_pipeline.py --data-dir data --model-dir models --output-dir batch_reports --days 7
```

**With Docker:**
```bash
# Start API service
docker-compose up -d api

# Run batch pipeline
docker-compose run batch_worker
```

---

### Running All Phases Sequentially

To run all phases in order:

```bash
# Run all phases
python phase0_data_analysis.py && \
python phase1_preprocessing.py && \
python phase2_supervised_learning.py && \
python phase3_exploratory_analysis.py && \
python phase4_anomaly_detection.py && \
python phase5_spatiotemporal.py && \
python phase6_operationalization.py
```

**Total Duration:** ~15-20 minutes

---

## Verification

### Verify Phase Outputs

After running each phase, verify that output files were created:

**Phase 0:**
```bash
ls -lh data_dictionary.json sanity_check_report.txt sample_data.csv
```

**Phase 1:**
```bash
ls -lh processed_data.csv train_data.csv val_data.csv test_data.csv preprocessing_pipeline.pkl
```

**Phase 2:**
```bash
ls -lh models/
mlflow ui  # Check experiment tracking
```

**Phase 3-5:**
```bash
# Check for generated reports and visualizations
find . -name "*.png" -o -name "*.pdf" -o -name "*_COMPLETE.md"
```

### Run Tests

The project includes unit tests for the preprocessing module:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_preprocessing.py
```

**Expected:** All 19 tests should pass ✅

### Verify API (Phase 6)

If you started the API:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","models_loaded":6,"preprocessing_available":true}

# List available models
curl http://localhost:8000/models

# View interactive API documentation
# Open http://localhost:8000/docs in your browser
```

### Verify Docker Installation

```bash
# Check running containers
docker ps

# Check container logs
docker-compose logs api

# Test API health check
docker-compose exec api curl http://localhost:8000/health
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Python Version Incompatibility

**Problem:** `SyntaxError` or module import errors

**Solution:**
```bash
# Check Python version (must be 3.8+)
python --version

# If using an older version, install Python 3.9
# For Ubuntu/Debian:
sudo apt-get update
sudo apt-get install python3.9 python3.9-venv

# Create virtual environment with Python 3.9
python3.9 -m venv venv
source venv/bin/activate
```

#### Issue 2: Missing Dependencies

**Problem:** `ModuleNotFoundError: No module named 'XXX'`

**Solution:**
```bash
# Install the missing package
pip install <package_name>

# Or reinstall all dependencies
pip install pandas numpy scikit-learn scipy joblib matplotlib seaborn mlflow xgboost mlxtend networkx python-louvain fastapi uvicorn pydantic
```

#### Issue 3: Memory Error

**Problem:** `MemoryError` during Phase 2 or 3

**Solution:**
- Close other applications to free up RAM
- Use a subset of data:
  ```bash
  # Use sample data instead of full dataset
  python phase1_preprocessing.py  # Uses sample_data.csv
  ```
- Reduce batch size in Phase 2

#### Issue 4: File Not Found

**Problem:** `FileNotFoundError: 'raw - data.csv'`

**Solution:**
```bash
# Make sure you're in the project root directory
cd thesis-project03
pwd  # Should show .../thesis-project03

# Check if data file exists
ls -lh "raw - data.csv"
```

#### Issue 5: MLflow UI Not Starting

**Problem:** MLflow UI won't start or shows errors

**Solution:**
```bash
# Install MLflow if not already installed
pip install mlflow

# Start MLflow UI with explicit tracking URI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# If port 5000 is in use, use a different port
mlflow ui --port 5001
```

#### Issue 6: Docker Build Fails

**Problem:** Docker build errors or timeouts

**Solution:**
```bash
# Clean Docker cache and rebuild
docker system prune -a
docker-compose build --no-cache

# Check Docker daemon is running
docker info

# If using Windows/Mac, restart Docker Desktop
```

#### Issue 7: Permission Denied (Linux)

**Problem:** Permission denied when running Docker commands

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and log back in for changes to take effect
# Or run:
newgrp docker

# Verify
docker run hello-world
```

#### Issue 8: Port Already in Use

**Problem:** `Address already in use` when starting API

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill the process or use a different port
uvicorn operationalization.api:app --port 8001
```

#### Issue 9: Matplotlib Display Issues

**Problem:** Matplotlib plots not displaying

**Solution:**
```bash
# For headless environments (servers without GUI)
# Set matplotlib backend
export MPLBACKEND=Agg

# Or in Python code, add at the top:
# import matplotlib
# matplotlib.use('Agg')
```

#### Issue 10: Unicode Errors in Data Files

**Problem:** `UnicodeDecodeError` when loading CSV files

**Solution:**
```python
# Load CSV with explicit encoding
import pandas as pd
df = pd.read_csv('raw - data.csv', encoding='utf-8')

# Or try:
df = pd.read_csv('raw - data.csv', encoding='latin-1')
```

---

## Additional Resources

### Documentation

- **Main README**: `README.md` - Project overview
- **Phase Documentation**: 
  - `preprocessing/README.md` - Phase 1 details
  - `experiments/README.md` - Phase 2 details
  - `exploratory/README.md` - Phase 3 details
  - `anomaly/README.md` - Phase 4 details
  - `spatiotemporal/README.md` - Phase 5 details
  - `operationalization/README.md` - Phase 6 details

### Phase Completion Summaries

- `PHASE0_SUMMARY.md` - Phase 0 results
- `PHASE1_SUMMARY.md` / `IMPLEMENTATION_COMPLETE.md` - Phase 1 results
- `PHASE2_COMPLETE.md` - Phase 2 results
- `PHASE3_COMPLETE.md` - Phase 3 results
- `PHASE4_COMPLETE.md` - Phase 4 results
- `PHASE5_COMPLETE.md` - Phase 5 results
- `PHASE6_COMPLETE.md` - Phase 6 results
- `PROJECT_COMPLETE.md` - Overall project summary

### Getting Help

If you encounter issues not covered in this guide:

1. Check the specific phase README in the module directory
2. Review the phase completion summaries for examples
3. Check the issue tracker on GitHub
4. Review the code comments and docstrings

---

## Next Steps

After successful installation:

1. **Start with Phase 0** to understand the data
2. **Run Phase 1** to preprocess the data
3. **Explore Phase 2-5** based on your analysis needs
4. **Deploy Phase 6** if you need operational tools

For a quick start guide, see [QUICKSTART.md](QUICKSTART.md)

---

**Last Updated:** December 2024  
**Project Version:** 1.0 (All 6 phases complete)
