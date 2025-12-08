# Dockerfile for AMR Surveillance API

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY preprocessing/ ./preprocessing/
COPY experiments/ ./experiments/
COPY exploratory/ ./exploratory/
COPY anomaly/ ./anomaly/
COPY spatiotemporal/ ./spatiotemporal/
COPY operationalization/ ./operationalization/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API
CMD ["uvicorn", "operationalization.api:app", "--host", "0.0.0.0", "--port", "8000"]
