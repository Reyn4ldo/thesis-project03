# Documentation Guide - Where to Start?

This project has multiple documentation files. This guide helps you choose the right one.

## Quick Decision Tree

```
Are you new to this project?
│
├─ YES → Want to start IMMEDIATELY (< 5 min)?
│   │
│   ├─ YES → Read HOW_TO_RUN.md
│   │        (3 commands, get running now!)
│   │
│   └─ NO → Have 10 minutes?
│       │
│       ├─ YES → Read QUICKSTART.md
│       │        (Examples & hands-on guide)
│       │
│       └─ NO → Read INSTALLATION.md
│                (Complete guide with troubleshooting)
│
└─ NO → Looking for specific information?
    │
    ├─ General project info → README.md
    ├─ Specific phase details → <module>/README.md
    ├─ Phase results → PHASE*_COMPLETE.md
    └─ Troubleshooting → INSTALLATION.md
```

## Documentation Files Overview

### 🚀 Getting Started (Choose ONE based on your needs)

| File | Who it's for | Time | What you get |
|------|--------------|------|--------------|
| **[HOW_TO_RUN.md](HOW_TO_RUN.md)** | Impatient users | 5 min | 3 commands to start |
| **[QUICKSTART.md](QUICKSTART.md)** | Hands-on learners | 10 min | Examples & common tasks |
| **[INSTALLATION.md](INSTALLATION.md)** | Thorough readers | 30 min | Everything + troubleshooting |

### 📚 Reference Documentation

| File | Purpose |
|------|---------|
| **[README.md](README.md)** | Project overview, all phases, complete information |
| **requirements.txt** | Install all dependencies at once |
| **requirements_api.txt** | Install only API dependencies |

### 📖 Phase-Specific Documentation

Each module has its own README with detailed information:

- **[preprocessing/README.md](preprocessing/README.md)** - Phase 1 details
- **[experiments/README.md](experiments/README.md)** - Phase 2 details
- **[exploratory/README.md](exploratory/README.md)** - Phase 3 details
- **[anomaly/README.md](anomaly/README.md)** - Phase 4 details
- **[spatiotemporal/README.md](spatiotemporal/README.md)** - Phase 5 details
- **[operationalization/README.md](operationalization/README.md)** - Phase 6 details

### 📊 Completion Reports

Results and summaries from completed phases:

- **PHASE0_SUMMARY.md** - Data understanding results
- **PHASE1_SUMMARY.md** / **IMPLEMENTATION_COMPLETE.md** - Preprocessing results
- **PHASE2_COMPLETE.md** - Machine learning results
- **PHASE3_COMPLETE.md** - Exploratory analysis results
- **PHASE4_COMPLETE.md** - Anomaly detection results
- **PHASE5_COMPLETE.md** - Spatio-temporal analysis results
- **PHASE6_COMPLETE.md** - Operationalization results
- **PROJECT_COMPLETE.md** - Overall project summary

## Common Scenarios

### Scenario 1: "I just want to run this NOW"

1. Open **[HOW_TO_RUN.md](HOW_TO_RUN.md)**
2. Copy the 3 commands
3. Done!

### Scenario 2: "I want to understand what I'm doing"

1. Read **[QUICKSTART.md](QUICKSTART.md)** (10 minutes)
2. Follow the examples
3. Reference other docs as needed

### Scenario 3: "I need to deploy this to production"

1. Read **[INSTALLATION.md](INSTALLATION.md)** Docker section
2. Follow **[operationalization/README.md](operationalization/README.md)**
3. Reference **[PHASE6_COMPLETE.md](PHASE6_COMPLETE.md)** for details

### Scenario 4: "Something went wrong"

1. Check **[INSTALLATION.md](INSTALLATION.md)** Troubleshooting section
2. Look at error message
3. Find matching issue and solution

### Scenario 5: "I want to understand the results"

1. Read **[README.md](README.md)** for overview
2. Check **PHASE*_COMPLETE.md** files for specific phase results
3. Read phase-specific READMEs for detailed analysis

### Scenario 6: "I want to use a specific feature"

1. Check **[README.md](README.md)** Project Structure
2. Go to relevant module directory
3. Read module's README.md
4. Look at code examples in **[QUICKSTART.md](QUICKSTART.md)**

## File Sizes & Reading Times

| File | Size | Lines | Reading Time |
|------|------|-------|--------------|
| HOW_TO_RUN.md | 2.9 KB | 92 | 3-5 minutes |
| QUICKSTART.md | 8.5 KB | 368 | 10-15 minutes |
| INSTALLATION.md | 14 KB | 648 | 30-45 minutes |
| README.md | 30 KB | 950+ | 45-60 minutes |

## Installation Quick Reference

```bash
# Simplest - Basic analysis (Phases 0-1)
pip install pandas numpy scikit-learn scipy joblib

# Quick - Most phases (0-3)
pip install pandas numpy scikit-learn scipy joblib matplotlib seaborn

# Complete - All phases
pip install -r requirements.txt

# API only - Phase 6
pip install -r requirements_api.txt
```

## Running Quick Reference

```bash
# Basic analysis
python phase0_data_analysis.py      # 30 seconds
python phase1_preprocessing.py      # 1 minute

# Machine learning
python phase2_supervised_learning.py # 5-10 minutes

# Advanced analysis
python phase3_exploratory_analysis.py # 2-3 minutes
python phase4_anomaly_detection.py   # 1-2 minutes
python phase5_spatiotemporal.py      # 2-3 minutes

# API deployment
python phase6_operationalization.py  # 1 minute
uvicorn operationalization.api:app   # Starts API
```

## Tips

💡 **Start simple**: Begin with HOW_TO_RUN.md even if you're experienced

💡 **Use Docker**: Easiest way to avoid dependency issues

💡 **Check examples**: QUICKSTART.md has copy-paste code examples

💡 **Read summaries first**: PHASE*_COMPLETE.md files give you results without running

💡 **Module READMEs**: Most detailed information about specific functionality

## Still Lost?

If you're still not sure where to start:

1. **Read HOW_TO_RUN.md** (it's only 92 lines!)
2. Run the 3 commands to get something working
3. Come back and explore other docs based on what you need

---

**Remember**: You don't need to read everything! Pick the doc that matches your needs and jump in.

**Last Updated:** December 2024
