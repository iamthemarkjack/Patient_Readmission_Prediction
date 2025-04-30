import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import json
import logging
from typing import List
import os
import pickle
from contextlib import asynccontextmanager

from add_feature import add_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("readmission_api")

# Environment variables with defaults
API_PORT = int(os.environ.get("API_PORT", 8000))
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("pickles", "best_model.pkl"))
MODEL_INFO_PATH = os.environ.get("MODEL_INFO_PATH", os.path.join("metadata", "model_registry_info.json"))
META_DATA_PATH = os.environ.get("META_DATA_PATH", os.path.join("metadata", "preprocessing_metadata.json"))
ENCODERS_PATH = os.environ.get("ENCODERS_PATH", os.path.join("pickles", "encoders.pkl"))
SCALER_PATH = os.environ.get("SCALER_PATH", os.path.join("pickles", "standard_scaler.pkl"))

# Define input schema with validation
class PatientData(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Patient gender")
    primary_diagnosis: str = Field(..., description="Primary diagnosis category")
    num_procedures: int = Field(..., ge=0, description="Number of procedures performed")
    days_in_hospital: int = Field(..., ge=0, description="Length of stay in hospital (days)")
    comorbidity_score: int = Field(..., ge=0, description="Patient comorbidity score")
    discharge_to: str = Field(..., description="Discharge destination")

    class Config:
        schema_extra = {
            "example": {
                "age": 80,
                "gender": "Female",
                "primary_diagnosis": "Circulatory system",
                "num_procedures": 2,
                "days_in_hospital": 3,
                "comorbidity_score": 4,
                "discharge_to": "Home"
            }
        }

class PredictionResponse(BaseModel):
    readmission_probability: float
    will_be_readmitted: bool
    model_version: str
    model_type: str

class ModelContainer:
    def __init__(self):
        self.model = None
        self.model_name = "unknown"
        self.model_version = "unknown"
        self.model_type = "unknown"
        self.f1_score = None
        self.healthy = False

# Global model container instance
model_container = ModelContainer()

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting application, loading model from {MODEL_PATH}")
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model_container.model = pickle.load(f)
            model_container.healthy = True
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        
        # Load model info
        try:
            with open(MODEL_INFO_PATH, 'r') as f:
                model_info = json.load(f)
                model_container.model_name = model_info.get("model_name", "unknown")
                model_container.model_version = str(model_info.get("model_version", "unknown"))
                model_container.model_type = model_info.get("best_model_type", "unknown")
                model_container.f1_score = model_info.get("f1_score", None)
                logger.info(f"Model info loaded: {model_container.model_name} v{model_container.model_version} "
                           f"({model_container.model_type}), F1 score: {model_container.f1_score}")
        except Exception as e:
            logger.warning(f"Could not load model info: {e}")
    except Exception as e:
        logger.error(f"Could not load model: {e}")

    yield  # Application runs during this time
    
    logger.info("Shutting down application")

# Create app instance with lifespan
app = FastAPI(
    title="Patient Readmission Prediction API",
    description="API for predicting patient readmission within 30 days",
    lifespan=lifespan,
    version="1.0.0"
)

# Dependency for checking model availability
async def verify_model():
    if not model_container.healthy or model_container.model is None:
        logger.error("Model is not available")
        raise HTTPException(status_code=503, detail="Model is not available")
    return True

@app.get("/", summary="API root endpoint")
def read_root():
    """Return a welcome message for the API root endpoint."""
    logger.info("Root endpoint accessed")
    return {
        "message": "Welcome to Patient Readmission Prediction API", 
        "status": "active",
        "model": {
            "name": model_container.model_name,
            "version": model_container.model_version,
            "type": model_container.model_type
        }
    }

@app.get("/health", summary="API health check")
async def health_check():
    """Check if the API and model are healthy."""
    model_status = "healthy" if model_container.healthy else "unhealthy"
    logger.debug(f"Health check performed. Model status: {model_status}")
    return {
        "status": "ok",
        "model_status": model_status,
        "model_name": model_container.model_name,
        "model_version": model_container.model_version,
        "model_type": model_container.model_type,
        "f1_score": model_container.f1_score
    }

@app.get("/version", summary="Get model version")
async def version():
    """Get model version information."""
    return {
        "model_name": model_container.model_name,
        "model_version": model_container.model_version,
        "model_type": model_container.model_type,
        "f1_score": model_container.f1_score
    }

@app.post("/predict", response_model=PredictionResponse, summary="Predict patient readmission")
async def predict(patient: PatientData, model_ok: bool = Depends(verify_model)):
    """
    Predict whether a patient will be readmitted within 30 days.
    
    Returns readmission probability and binary prediction.
    """
    logger.info(f"Processing prediction request for patient with primary diagnosis: {patient.primary_diagnosis}")
    try:
        # Convert input to dataframe
        patient_df = pd.DataFrame([patient.model_dump()])
        logger.info(f"Converted patient data to dataframe with shape: {patient_df.shape}")

        patient_df = add_features(patient_df)

        # Read the preprocessing meta data
        try:
            with open(META_DATA_PATH, 'r') as f:
                meta_data = json.load(f)
                cat_cols = meta_data.get("categorical_columns", [])
                num_cols = meta_data.get("numerical_columns", [])
        except Exception as e:
            logger.warning(f"Could not load preprocessing meta data: {e}")
        
        # Open pickled encoder and scaler
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
            logger.info(f"Loaded label encoders for columns: {list(encoders.keys())}")
        
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
            logger.info("Loaded standard scaler")

        for col in cat_cols:
            patient_df[col] = encoders[col].transform(patient_df[[col]])
        
        patient_df[num_cols] = scaler.transform(patient_df[num_cols])

        # Make prediction
        prediction = model_container.model.predict(patient_df)[0]
        logger.info(f"Binary prediction: {prediction}")
        
        # Get probability
        try:
            probability = model_container.model.predict_proba(patient_df)[0][1]
            logger.debug(f"Probability: {probability}")
        except Exception as e:
            logger.warning(f"Could not get probability: {e}. Using binary prediction instead.")
            probability = float(prediction)
        
        result = {
            "readmission_probability": float(probability),
            "will_be_readmitted": bool(prediction),
            "model_version": model_container.model_version,
            "model_type": model_container.model_type
        }
        
        logger.info(f"Prediction complete: {result}")
        return result

    except Exception as e:
        logger.exception(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", summary="Batch predict patient readmissions")
async def predict_batch(patients: List[PatientData], model_ok: bool = Depends(verify_model)):
    """
    Process multiple patient predictions in a single request.
    
    Returns readmission probabilities and binary predictions for each patient.
    """
    logger.info(f"Processing batch prediction request for {len(patients)} patients")
    try:
        # Convert input to dataframe
        patients_df = pd.DataFrame([p.model_dump() for p in patients])
        logger.info(f"Converted batch data to dataframe with shape: {patients_df.shape}")

        patients_df = add_features(patients_df)

        # Read the preprocessing meta data
        try:
            with open(META_DATA_PATH, 'r') as f:
                meta_data = json.load(f)
                cat_cols = meta_data.get("categorical_columns", [])
                num_cols = meta_data.get("numerical_columns", [])
        except Exception as e:
            logger.warning(f"Could not load preprocessing meta data: {e}")
        
        # Open pickled encoder and scaler
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
            logger.info(f"Loaded label encoders for columns: {list(encoders.keys())}")
        
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
            logger.info("Loaded standard scaler")

        for col in cat_cols:
            patients_df[col] = encoders[col].transform(patients_df[[col]])
        
        patients_df[num_cols] = scaler.transform(patients_df[num_cols])

        # Make predictions
        predictions = model_container.model.predict(patients_df)
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Get probabilities
        try:
            probabilities = [p[1] for p in model_container.model.predict_proba(patients_df)]
            logger.info(f"Generated {len(probabilities)} probabilities")
        except Exception as e:
            logger.warning(f"Could not get probabilities: {e}. Using binary predictions instead.")
            probabilities = [float(p) for p in predictions]
        
        results = [
            {
                "readmission_probability": float(prob),
                "will_be_readmitted": bool(pred),
                "model_version": model_container.model_version,
                "model_type": model_container.model_type
            }
            for prob, pred in zip(probabilities, predictions)
        ]
        
        logger.info(f"Batch prediction complete. Processed {len(results)} results")
        return results

    except Exception as e:
        logger.exception(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True, log_level="info")