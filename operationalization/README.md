# Phase 6: Operationalization & Outputs

Production-ready components for operational AMR surveillance including automated antibiograms, early warning alerts, empiric therapy recommendations, REST API, and batch scoring pipeline.

## Overview

Phase 6 transforms research outputs into operational tools for clinical and public health use:

1. **Automated Antibiogram Generator** - CLSI-compliant antibiograms per species/site
2. **Early Warning System** - Real-time alerts for resistance trends
3. **Empiric Therapy Recommender** - Evidence-based treatment guidance
4. **REST API** - Model scoring and recommendations via HTTP
5. **Batch Pipeline** - Automated weekly surveillance reports
6. **Docker Deployment** - Containerized for production

## Components

### 1. Antibiogram Generator

Generates standardized antibiogram reports following CLSI M39 guidelines.

```python
from operationalization import AntibiogramGenerator

# Initialize
generator = AntibiogramGenerator(min_isolates=30)

# Generate antibiogram for E. coli
antibiogram = generator.generate_antibiogram(
    df,
    species='Escherichia coli',
    site='Hospital_A'
)

# Results include:
# - susceptibility: % susceptible per antibiotic
# - counts: number of isolates tested
# - sir_distribution: full S/I/R breakdown

# Visualize
generator.plot_antibiogram(antibiogram, title='E. coli Antibiogram')

# Save
generator.save_antibiogram(antibiogram, name='ecoli_site_a')
```

**Features:**
- Minimum isolate thresholds (CLSI: 30)
- Species-specific and site-specific reports
- Comparative antibiograms across groups
- Publication-quality visualizations
- Excel and CSV export

### 2. Early Warning System

Monitors surveillance data and generates alerts for concerning trends.

```python
from operationalization import EarlyWarningSystem

# Initialize
ews = EarlyWarningSystem(alert_threshold=0.2)  # 20% increase triggers alert

# Check for resistance increase
alert = ews.check_resistance_increase(
    df,
    antibiotic='ciprofloxacin_sir',
    window_size=30
)

# Check threshold breach
alert = ews.check_threshold_breach(
    df,
    antibiotic='ciprofloxacin_sir',
    threshold=0.3  # 30% resistance
)

# Run complete surveillance check
alerts = ews.run_surveillance_check(df)

# Get alert summary
summary = ews.get_alert_summary()
```

**Alert Types:**
- **Resistance Increase**: Significant trend changes
- **Threshold Breach**: Critical resistance levels exceeded
- **Anomalous Isolates**: Unusual resistance patterns
- **New MDR Patterns**: Emergence of multi-drug resistance

**Severity Levels:**
- Critical (≥90% threshold or ≥50% increase)
- High (≥80% threshold or ≥20% increase)
- Medium (≥50% threshold)

### 3. Empiric Therapy Recommender

Provides evidence-based antibiotic recommendations for empiric therapy.

```python
from operationalization import EmpiricTherapyRecommender

# Initialize
recommender = EmpiricTherapyRecommender(min_susceptibility=0.80)

# Get recommendations
recommendations = recommender.recommend_antibiotics(
    df,
    species='Escherichia coli',
    source='urine',
    contraindications=['penicillin'],  # Patient allergies
    top_n=5
)

# Generate full report
report = recommender.generate_therapy_report(
    df,
    patient_info={
        'species': 'Escherichia coli',
        'source': 'urine',
        'site': 'Hospital_A',
        'contraindications': ['penicillin']
    }
)

# Results include:
# - primary_recommendations: ranked antibiotics
# - alternative_options: backup choices
# - susceptibility_probability: likelihood of success
# - confidence: based on data volume
```

**Recommendation Criteria:**
- ≥80% susceptibility probability (configurable)
- Species-specific when available
- Site-specific antibiogram data
- Patient contraindications excluded
- Confidence levels (high/medium/low)

### 4. REST API

FastAPI-based REST API for model scoring and recommendations.

**Start API:**
```bash
# Development
uvicorn operationalization.api:app --reload

# Production
uvicorn operationalization.api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Endpoints:**

**Health Check:**
```bash
curl http://localhost:8000/health
```

**List Models:**
```bash
curl http://localhost:8000/models
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "isolate": {
      "features": {"feature1": 1.0, "feature2": 0.5},
      "metadata": {"id": "ISO001"}
    },
    "model_name": "esbl_classifier"
  }'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "isolates": [
      {"features": {...}, "metadata": {...}},
      {"features": {...}, "metadata": {...}}
    ],
    "model_name": "esbl_classifier"
  }'
```

**Therapy Recommendations:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "species": "Escherichia coli",
    "source": "urine",
    "contraindications": ["penicillin"]
  }'
```

**Get Antibiogram:**
```bash
curl http://localhost:8000/antibiogram/Escherichia%20coli?site=Hospital_A
```

### 5. Batch Scoring Pipeline

Automated batch processing for weekly surveillance reports.

```bash
# Run batch pipeline
python operationalization/batch_pipeline.py \
  --data-dir data \
  --model-dir models \
  --output-dir batch_reports \
  --days 7
```

**Pipeline Steps:**
1. Load new isolates from last 7 days
2. Run predictions with all models
3. Generate antibiograms for all species/sites
4. Run early warning surveillance checks
5. Generate therapy recommendations
6. Create comprehensive summary report

**Outputs:**
- `batch_reports/antibiograms/` - All antibiogram reports
- `batch_reports/alerts/` - Alert logs
- `batch_reports/weekly_summary_*.json` - Summary report

**Scheduling:**

**Cron (Linux/Mac):**
```bash
# Run every Monday at 1 AM
0 1 * * 1 cd /path/to/project && python operationalization/batch_pipeline.py
```

**Airflow DAG:**
```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

dag = DAG(
    'amr_surveillance',
    default_args={'retries': 1},
    schedule_interval='0 1 * * 1',  # Weekly
    start_date=datetime(2024, 1, 1)
)

batch_task = BashOperator(
    task_id='run_batch_pipeline',
    bash_command='python /path/to/operationalization/batch_pipeline.py',
    dag=dag
)
```

### 6. Docker Deployment

Containerized deployment for production environments.

**Build and Run:**
```bash
# Build image
docker build -t amr-surveillance .

# Run API container
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  amr-surveillance

# Or use docker-compose
docker-compose up -d
```

**Docker Compose:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- `api`: REST API server (port 8000)
- `batch_worker`: Batch processing (scheduled)

## Complete Workflow

### Development Setup

```bash
# Install dependencies
pip install fastapi uvicorn pydantic pandas scikit-learn joblib matplotlib seaborn

# Run Phase 6 demonstration
python phase6_operationalization.py --data sample_data.csv --output-dir phase6_outputs
```

### Production Deployment

```bash
# 1. Build Docker image
docker build -t amr-surveillance .

# 2. Start services
docker-compose up -d

# 3. Verify API
curl http://localhost:8000/health

# 4. Schedule batch pipeline
# Add to crontab or Airflow
```

## Integration Examples

### Python Client

```python
import requests

# API endpoint
API_URL = "http://localhost:8000"

# Make prediction
response = requests.post(
    f"{API_URL}/predict",
    json={
        "isolate": {
            "features": {"feature1": 1.0, "feature2": 0.5},
            "metadata": {"id": "ISO001"}
        },
        "model_name": "esbl_classifier"
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")

# Get therapy recommendations
response = requests.post(
    f"{API_URL}/recommend",
    json={
        "species": "Escherichia coli",
        "source": "urine"
    }
)

recommendations = response.json()
for rec in recommendations['recommendations']:
    print(f"{rec['antibiotic']}: {rec['susceptibility_probability']:.1%}")
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

const API_URL = 'http://localhost:8000';

// Make prediction
async function predict(features, model='esbl_classifier') {
  const response = await axios.post(`${API_URL}/predict`, {
    isolate: { features },
    model_name: model
  });
  
  return response.data;
}

// Get recommendations
async function getRecommendations(species, source) {
  const response = await axios.post(`${API_URL}/recommend`, {
    species,
    source
  });
  
  return response.data.recommendations;
}
```

## Monitoring & Observability

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "models_loaded": 6,
#   "preprocessing_available": true
# }
```

### Logging

All components write logs to their respective output directories:
- Antibiograms: `antibiograms/`
- Alerts: `alerts/`
- Batch reports: `batch_reports/`

### Metrics

Track operational metrics:
- API response times
- Prediction latency
- Alert frequency
- Batch pipeline duration

## Security Considerations

1. **API Authentication**: Add authentication middleware for production
2. **Data Privacy**: Ensure PHI/PII is properly anonymized
3. **Access Control**: Implement role-based access
4. **Audit Logging**: Log all API requests and recommendations
5. **HTTPS**: Use TLS/SSL in production

## Performance

**API:**
- Single prediction: <100ms
- Batch prediction (100 isolates): <500ms
- Recommendation generation: <200ms

**Batch Pipeline:**
- 1000 isolates: ~5 minutes
- Includes all analyses (predictions, antibiograms, alerts, recommendations)

## Troubleshooting

**API won't start:**
- Check port 8000 is available: `lsof -i :8000`
- Verify models are in `models/` directory
- Check logs: `docker-compose logs api`

**Batch pipeline fails:**
- Verify data file exists
- Check model files are present
- Ensure output directory is writable

**Docker issues:**
- Rebuild image: `docker-compose build --no-cache`
- Check volumes: `docker-compose config`
- View logs: `docker-compose logs -f`

## Testing

```bash
# Test API endpoints
python -m pytest tests/test_api.py

# Test batch pipeline
python -m pytest tests/test_batch_pipeline.py

# Integration tests
python -m pytest tests/integration/
```

## Future Enhancements

1. **Dashboard**: Streamlit/Grafana dashboard for visualization
2. **Real-time Messaging**: Slack/email alerts for critical issues
3. **Model Registry**: MLflow model registry for versioning
4. **CI/CD Pipeline**: Automated testing and deployment
5. **A/B Testing**: Compare model versions in production
6. **Advanced Analytics**: Power BI/Tableau integration

## References

- CLSI M39: Analysis and Presentation of Cumulative Antimicrobial Susceptibility Test Data
- WHO AWaRe Classification of Antibiotics
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/

## Support

For issues or questions:
1. Check documentation in this README
2. Review API documentation: http://localhost:8000/docs
3. Check logs in output directories
4. Contact: AMR Surveillance Team
