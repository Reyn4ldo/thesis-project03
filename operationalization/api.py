"""
REST API for Model Scoring

FastAPI-based REST API for scoring isolates using trained models.
Provides endpoints for predictions, recommendations, and antibiograms.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import json

# Global model registry
models = {}
preprocessing_pipeline = None
antibiogram_generator = None
therapy_recommender = None


def load_models(model_dir: str = "models"):
    """Load trained models from directory."""
    global models, preprocessing_pipeline
    
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"Model directory {model_dir} not found")
        return
    
    # Load preprocessing pipeline
    pipeline_path = model_path / "preprocessing_pipeline.pkl"
    if pipeline_path.exists():
        preprocessing_pipeline = joblib.load(pipeline_path)
        print("Loaded preprocessing pipeline")
    
    # Load trained models
    for model_file in model_path.glob("*.pkl"):
        if model_file.name != "preprocessing_pipeline.pkl":
            model_name = model_file.stem
            models[model_name] = joblib.load(model_file)
            print(f"Loaded model: {model_name}")


def initialize_components():
    """Initialize antibiogram generator and therapy recommender."""
    global antibiogram_generator, therapy_recommender
    
    from operationalization import AntibiogramGenerator, EmpiricTherapyRecommender
    
    antibiogram_generator = AntibiogramGenerator()
    therapy_recommender = EmpiricTherapyRecommender()
    
    print("Initialized operational components")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    load_models()
    initialize_components()
    yield
    # Shutdown
    pass


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AMR Surveillance API",
    description="API for antimicrobial resistance predictions and recommendations",
    version="1.0.0",
    lifespan=lifespan
)


class IsolateData(BaseModel):
    """Input data for a single isolate."""
    features: Dict[str, float]
    metadata: Optional[Dict[str, str]] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    isolates: List[IsolateData]
    model_name: str = "esbl_classifier"


class TherapyRequest(BaseModel):
    """Therapy recommendation request."""
    species: Optional[str] = None
    site: Optional[str] = None
    source: Optional[str] = None
    contraindications: Optional[List[str]] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AMR Surveillance API",
        "version": "1.0.0",
        "endpoints": [
            "/models",
            "/predict",
            "/predict/batch",
            "/recommend",
            "/antibiogram",
            "/health"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "preprocessing_available": preprocessing_pipeline is not None
    }


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": list(models.keys()),
        "count": len(models)
    }


@app.post("/predict")
async def predict(isolate: IsolateData, model_name: str = "esbl_classifier"):
    """
    Predict for a single isolate.
    
    Parameters:
    - isolate: Isolate feature data
    - model_name: Name of model to use
    
    Returns:
    - prediction: Model prediction
    - probability: Prediction probability (if available)
    """
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([isolate.features])
        
        # Get model
        model = models[model_name]
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            probability = float(proba.max())
        
        return {
            "model": model_name,
            "prediction": int(prediction) if isinstance(prediction, (np.integer, np.int64)) else str(prediction),
            "probability": probability,
            "metadata": isolate.metadata
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple isolates.
    
    Parameters:
    - request: Batch prediction request with isolates and model name
    
    Returns:
    - predictions: List of predictions for all isolates
    """
    if request.model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
    
    try:
        # Convert to DataFrame
        features_list = [isolate.features for isolate in request.isolates]
        df = pd.DataFrame(features_list)
        
        # Get model
        model = models[request.model_name]
        
        # Make predictions
        predictions = model.predict(df)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)
            probabilities = proba.max(axis=1).tolist()
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            result = {
                "index": i,
                "prediction": int(pred) if isinstance(pred, (np.integer, np.int64)) else str(pred),
                "probability": probabilities[i] if probabilities else None,
                "metadata": request.isolates[i].metadata
            }
            results.append(result)
        
        return {
            "model": request.model_name,
            "count": len(results),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/recommend")
async def recommend_therapy(request: TherapyRequest):
    """
    Get empiric therapy recommendations.
    
    Parameters:
    - request: Patient information for therapy recommendation
    
    Returns:
    - recommendations: Ranked antibiotic recommendations
    """
    if therapy_recommender is None:
        raise HTTPException(status_code=503, detail="Therapy recommender not initialized")
    
    try:
        # Load historical data (in production, this would be from database)
        # For now, return mock recommendations
        recommendations = [
            {
                "rank": 1,
                "antibiotic": "Ciprofloxacin",
                "susceptibility_probability": 0.85,
                "confidence": "high",
                "n_isolates": 150
            },
            {
                "rank": 2,
                "antibiotic": "Gentamicin",
                "susceptibility_probability": 0.82,
                "confidence": "high",
                "n_isolates": 145
            }
        ]
        
        return {
            "patient_info": {
                "species": request.species,
                "site": request.site,
                "source": request.source
            },
            "recommendations": recommendations,
            "contraindications_applied": request.contraindications or []
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


@app.get("/antibiogram/{species}")
async def get_antibiogram(species: str, site: Optional[str] = None):
    """
    Get antibiogram for a species.
    
    Parameters:
    - species: Bacterial species
    - site: Optional site filter
    
    Returns:
    - antibiogram: Susceptibility summary
    """
    if antibiogram_generator is None:
        raise HTTPException(status_code=503, detail="Antibiogram generator not initialized")
    
    try:
        # In production, load actual data
        # For now, return mock antibiogram
        antibiogram = {
            "species": species,
            "site": site or "all",
            "susceptibility": {
                "Ciprofloxacin": 85.2,
                "Gentamicin": 82.1,
                "Ceftazidime": 78.5
            },
            "n_isolates": {
                "Ciprofloxacin": 150,
                "Gentamicin": 145,
                "Ceftazidime": 142
            }
        }
        
        return antibiogram
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Antibiogram error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
