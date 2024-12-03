from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add src directory to path to import local modules
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes risk with uncertainty estimation",
    version="1.0.0"
)

class HealthIndicators(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int

class PredictionResponse(BaseModel):
    prediction: float
    risk_level: str
    confidence_score: float

# Initialize model and preprocessor variables
model = None
preprocessor = None

@app.on_event("startup")
async def load_model():
    global model, preprocessor
    try:
        model_path = os.path.join(src_path, "..", "models", "diabetes_model.joblib")
        preprocessor_path = os.path.join(src_path, "..", "models", "preprocessor.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
            
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        print("Model and preprocessor loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Diabetes Prediction API", "status": "active"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(health_indicators: HealthIndicators):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([health_indicators.dict()])
        
        # Preprocess input
        processed_data = preprocessor.transform(input_data)
        
        # Get prediction probability
        prediction_proba = model.predict_proba(processed_data)[0][1]
        
        # Determine risk level
        risk_level = "High Risk" if prediction_proba > 0.15 else "Low Risk"
        
        # Calculate confidence score (distance from decision boundary)
        confidence_score = abs(prediction_proba - 0.15) / 0.15
        confidence_score = min(confidence_score, 1.0)
        
        return PredictionResponse(
            prediction=float(prediction_proba),
            risk_level=risk_level,
            confidence_score=float(confidence_score)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "healthy"}
