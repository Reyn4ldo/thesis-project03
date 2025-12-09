# How to Run This Project

## TL;DR - Simplest Way

```bash
# 1. Clone and enter directory
git clone https://github.com/Reyn4ldo/thesis-project03.git
cd thesis-project03

# 2. Install Python packages (SECURE - without MLflow)
pip install -r requirements-secure.txt

# 3. Run the phases
python phase0_data_analysis.py      # Analyzes data (30 sec)
python phase1_preprocessing.py      # Cleans data (1 min)
python phase2_supervised_learning.py # Trains models (5-10 min)
```

Done! Check the generated files and reports.

> ⚠️ **Note**: Using `requirements-secure.txt` excludes MLflow (which has an unfixed security vulnerability). Models still train normally, just without experiment logging. See [SECURITY.md](SECURITY.md) for details.

---

## Need More Details?

- **[QUICKSTART.md](QUICKSTART.md)** - 10-minute getting started guide
- **[INSTALLATION.md](INSTALLATION.md)** - Complete installation with troubleshooting
- **[SECURITY.md](SECURITY.md)** - Security information about MLflow vulnerability
- **[README.md](README.md)** - Full project documentation

## What Each Phase Does

| Phase | Command | Time | What It Does |
|-------|---------|------|--------------|
| 0 | `python phase0_data_analysis.py` | 30s | Analyzes data quality and generates reports |
| 1 | `python phase1_preprocessing.py` | 1m | Cleans data and creates features |
| 2 | `python phase2_supervised_learning.py` | 5-10m | Trains machine learning models |
| 3 | `python phase3_exploratory_analysis.py` | 2-3m | Clustering and pattern discovery |
| 4 | `python phase4_anomaly_detection.py` | 1-2m | Finds outliers and anomalies |
| 5 | `python phase5_spatiotemporal.py` | 2-3m | Geographic and temporal analysis |
| 6 | `python phase6_operationalization.py` | 1m | Sets up production API |

## What Gets Created

After running the phases, you'll have:

- **Data Reports**: `data_dictionary.json`, `sanity_check_report.txt`
- **Processed Data**: `processed_data.csv`, `train_data.csv`, `test_data.csv`
- **ML Models**: `models/` directory with trained models
- **Analysis Results**: Various reports and visualizations

## Using Docker Instead

If you prefer Docker:

```bash
# 1. Clone repository
git clone https://github.com/Reyn4ldo/thesis-project03.git
cd thesis-project03

# 2. Run with Docker
docker-compose up -d

# 3. Check API is running
curl http://localhost:8000/health
```

## Common Questions

**Q: What do I need installed?**  
A: Python 3.8+ and pip. That's it!

**Q: How long does it take?**  
A: About 15-20 minutes to run all phases.

**Q: Can I run just one phase?**  
A: Yes! Each phase is independent, though later phases may need earlier ones completed first.

**Q: Where's the data?**  
A: It's included in the repo as `raw - data.csv` (583 bacterial isolates).

**Q: I got an error!**  
A: Check [INSTALLATION.md](INSTALLATION.md) troubleshooting section. Most issues are missing packages.

## Need Help?

1. Check [INSTALLATION.md](INSTALLATION.md) for detailed troubleshooting
2. Read [QUICKSTART.md](QUICKSTART.md) for quick examples
3. See [README.md](README.md) for complete documentation
4. Look at phase-specific READMEs in each module directory

---

**Last Updated:** December 2024
