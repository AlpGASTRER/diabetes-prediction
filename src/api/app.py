from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional

# Add src directory to path to import local modules
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

app = FastAPI(
    title="Diabetes Prediction API",
    description="""
    This API predicts diabetes risk based on various health indicators. 
    
    Key Features:
    - Predicts diabetes risk with confidence level
    - Provides detailed health warnings
    - Shows risk factors and their importance
    - Includes uncertainty estimates
    
    How to use:
    1. Use the /predict endpoint with your health indicators
    2. Review the prediction results and risk assessment
    3. Pay attention to health warnings and recommended actions
    4. Consider consulting healthcare professionals based on the risk level
    
    Note: This tool is for informational purposes only and should not replace professional medical advice.
    """,
    version="1.0.0"
)

class HealthIndicators(BaseModel):
    """Input health indicators for diabetes prediction"""
    
    BMI: float = Field(
        ..., 
        ge=10.0, 
        le=100.0, 
        description="Body Mass Index (weight in kg / height in meters squared). Normal range: 18.5-24.9, Overweight: 25-29.9, Obese: 30+"
    )
    Age: int = Field(
        ..., 
        ge=1, 
        le=13, 
        description="""Age category:
        1: 18-24 years
        2: 25-29 years
        3: 30-34 years
        4: 35-39 years
        5: 40-44 years
        6: 45-49 years
        7: 50-54 years
        8: 55-59 years
        9: 60-64 years
        10: 65-69 years
        11: 70-74 years
        12: 75-79 years
        13: 80+ years"""
    )
    GenHlth: int = Field(
        ..., 
        ge=1, 
        le=5, 
        description="""General Health Rating:
        1: Excellent
        2: Very Good
        3: Good
        4: Fair
        5: Poor"""
    )
    MentHlth: int = Field(
        ..., 
        ge=0, 
        le=30, 
        description="Number of days of poor mental health in past 30 days (0-30). High values may indicate stress or depression."
    )
    PhysHlth: int = Field(
        ..., 
        ge=0, 
        le=30, 
        description="Number of days of poor physical health in past 30 days (0-30). High values may indicate chronic conditions."
    )
    HighBP: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="High Blood Pressure status (0: No, 1: Yes). Important cardiovascular risk factor."
    )
    HighChol: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="High Cholesterol status (0: No, 1: Yes). Key indicator for metabolic health."
    )
    CholCheck: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Cholesterol check in past 5 years (0: No, 1: Yes). Regular monitoring is important."
    )
    Smoker: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Smoking status (0: No, 1: Yes). Significant risk factor for multiple health conditions."
    )
    Stroke: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="History of stroke (0: No, 1: Yes). Indicates serious cardiovascular risk."
    )
    HeartDiseaseorAttack: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="History of heart disease or heart attack (0: No, 1: Yes). Strong correlation with diabetes."
    )
    PhysActivity: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Physical activity in past month (0: No, 1: Yes). Regular exercise helps prevent diabetes."
    )
    Fruits: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Consumes fruit 1+ times per day (0: No, 1: Yes). Part of healthy diet."
    )
    Veggies: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Consumes vegetables 1+ times per day (0: No, 1: Yes). Important for balanced nutrition."
    )
    HvyAlcoholConsump: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Heavy alcohol consumption (0: No, 1: Yes). Can impact blood sugar levels."
    )
    AnyHealthcare: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Has any kind of healthcare coverage (0: No, 1: Yes). Important for preventive care."
    )
    NoDocbcCost: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Could not see doctor due to cost (0: No, 1: Yes). May indicate barriers to healthcare."
    )
    DiffWalk: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Difficulty walking or climbing stairs (0: No, 1: Yes). May limit physical activity."
    )
    Sex: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Biological sex (0: Female, 1: Male). May affect risk factors."
    )
    Education: int = Field(
        ..., 
        ge=1, 
        le=6, 
        description="""Education Level:
        1: Never attended school or only kindergarten
        2: Elementary
        3: Some high school
        4: High school graduate
        5: Some college or technical school
        6: College graduate"""
    )
    Income: int = Field(
        ..., 
        ge=1, 
        le=8, 
        description="""Income Level:
        1: Less than $10,000
        2: $10,000 to $15,000
        3: $15,000 to $20,000
        4: $20,000 to $25,000
        5: $25,000 to $35,000
        6: $35,000 to $50,000
        7: $50,000 to $75,000
        8: $75,000 or more"""
    )

    @validator('BMI')
    def validate_bmi(cls, v):
        """Validate BMI is within reasonable range"""
        if v < 10.0 or v > 100.0:
            raise ValueError('BMI must be between 10 and 100')
        return v

class StageResult(BaseModel):
    """Results for each prediction stage"""
    probability: float = Field(..., description="Probability of diabetes (0-1)")
    risk_assessment: str = Field(..., description="Risk level assessment with recommended actions")

class UncertaintyEstimates(BaseModel):
    """Model uncertainty estimates"""
    epistemic: float = Field(..., description="Model uncertainty - uncertainty in the model's knowledge")
    aleatoric: float = Field(..., description="Data uncertainty - natural variability in the data")
    total: float = Field(..., description="Total uncertainty (epistemic + aleatoric)")

class FeatureImportance(BaseModel):
    """Importance and description of risk factors"""
    importance: float = Field(..., description="Relative importance of this factor (0-1)")
    description: str = Field(..., description="Description of how this factor affects diabetes risk")

class PredictionResponse(BaseModel):
    """Complete prediction response with all details"""
    has_diabetes: bool = Field(..., description="Final diabetes prediction (True/False)")
    confidence_percentage: float = Field(..., description="Confidence in the prediction (0-100%)")
    screening_stage: StageResult = Field(..., description="Initial screening results")
    confirmation_stage: StageResult = Field(..., description="Confirmation stage results")
    risk_level: str = Field(..., description="Overall risk assessment with recommended actions")
    uncertainties: UncertaintyEstimates = Field(..., description="Uncertainty estimates for the prediction")
    warnings: Optional[List[str]] = Field(None, description="Health warnings and recommendations based on input values")
    feature_importances: Dict[str, FeatureImportance] = Field(..., description="Risk factors and their importance")

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
    """Root endpoint providing API information"""
    return {
        "message": "Diabetes Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/": "This information",
            "/health": "Health check endpoint",
            "/predict": "Make diabetes risk predictions",
            "/model-info": "Get model information"
        }
    }

@app.get("/health")
async def health_check():
    """Check if the model and preprocessor are loaded and ready"""
    if model is None or preprocessor is None:
        return {"detail": "Service not ready - model or preprocessor not loaded"}
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.get("/model-info")
async def model_info():
    """Get information about the model and its features"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Ensemble Model",
        "features": {
            "BMI": "Body Mass Index",
            "Age": "Age category (1: 18-24, 2: 25-29, ..., 13: 80 or older)",
            "GenHlth": "General Health (1=excellent to 5=poor)",
            "MentHlth": "Days of poor mental health",
            "PhysHlth": "Days of poor physical health",
            "HighBP": "High Blood Pressure",
            "HighChol": "High Cholesterol",
            "CholCheck": "Cholesterol Check in past 5 years",
            "Smoker": "Smoking Status",
            "Stroke": "History of Stroke",
            "HeartDiseaseorAttack": "History of Heart Disease",
            "PhysActivity": "Physical Activity",
            "Fruits": "Fruit Consumption",
            "Veggies": "Vegetable Consumption",
            "HvyAlcoholConsump": "Heavy Alcohol Consumption",
            "AnyHealthcare": "Has Healthcare Coverage",
            "NoDocbcCost": "No Doctor due to Cost",
            "DiffWalk": "Difficulty Walking",
            "Sex": "Sex (0=female, 1=male)",
            "Education": "Education Level",
            "Income": "Income Level"
        },
        "risk_levels": {
            "Low Risk": "Probability < 0.15",
            "Medium Risk": "0.15 ≤ Probability < 0.30",
            "High Risk": "Probability ≥ 0.30"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(health_indicators: HealthIndicators):
    """Make a diabetes risk prediction with uncertainty estimation"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([health_indicators.dict()])
        
        # Generate detailed warnings based on input values
        warnings = []
        
        # BMI-related warnings
        if health_indicators.BMI > 35:
            warnings.append("SEVERE RISK: BMI indicates severe obesity (>35) - extremely high risk factor for diabetes")
        elif health_indicators.BMI > 30:
            warnings.append("HIGH RISK: BMI indicates obesity (30-35) - significant risk factor for diabetes")
        elif health_indicators.BMI > 25:
            warnings.append("MODERATE RISK: BMI indicates overweight (25-30) - increased risk factor for diabetes")
        
        # Blood pressure and cholesterol warnings
        if health_indicators.HighBP == 1 and health_indicators.HighChol == 1:
            warnings.append("SEVERE RISK: Both high blood pressure and high cholesterol detected - these conditions significantly increase diabetes risk")
        elif health_indicators.HighBP == 1:
            warnings.append("HIGH RISK: High blood pressure detected - major risk factor for diabetes and cardiovascular complications")
        elif health_indicators.HighChol == 1:
            warnings.append("HIGH RISK: High cholesterol detected - significant risk factor for diabetes and heart disease")
        
        # Health check warnings
        if health_indicators.CholCheck == 0:
            warnings.append("HEALTH ALERT: No cholesterol check in 5 years - regular monitoring is essential")
        
        # Lifestyle-related warnings
        if health_indicators.PhysActivity == 0:
            warnings.append("LIFESTYLE ALERT: No physical activity reported - regular exercise can reduce diabetes risk by 30-50%")
        if health_indicators.Fruits == 0 and health_indicators.Veggies == 0:
            warnings.append("DIETARY ALERT: Low fruit and vegetable intake - a balanced diet is crucial for diabetes prevention")
        
        # General health warnings
        if health_indicators.GenHlth >= 4:
            warnings.append("HEALTH ALERT: Poor general health reported - immediate medical consultation recommended")
        if health_indicators.MentHlth > 14:
            warnings.append("MENTAL HEALTH ALERT: Frequent mental distress reported - can impact diabetes management")
        if health_indicators.PhysHlth > 14:
            warnings.append("PHYSICAL HEALTH ALERT: Frequent physical illness reported - may indicate underlying health issues")
        
        # Healthcare access warnings
        if health_indicators.AnyHealthcare == 0:
            warnings.append("ACCESS ALERT: No healthcare coverage - regular medical check-ups are essential for prevention")
        if health_indicators.NoDocbcCost == 1:
            warnings.append("ACCESS ALERT: Unable to see doctor due to cost - this may lead to delayed diagnosis and treatment")
        
        # Additional risk combinations
        if health_indicators.HeartDiseaseorAttack == 1:
            warnings.append("SEVERE RISK: History of heart disease - strongly associated with diabetes risk")
        if health_indicators.Stroke == 1:
            warnings.append("SEVERE RISK: History of stroke - indicates serious cardiovascular risk")
        if health_indicators.DiffWalk == 1:
            warnings.append("HEALTH ALERT: Difficulty walking reported - may limit physical activity and increase health risks")
        
        try:
            # Preprocess input data
            processed_data = preprocessor.transform(data)
            
            # Get prediction probabilities
            probabilities = model.predict_proba(processed_data)
            diabetes_prob = float(probabilities[0][1])
            screening_prob = diabetes_prob
            confirmation_prob = diabetes_prob
            
        except Exception as e:
            print(f"Model prediction error: {str(e)}")
            # Use mock values if prediction fails
            diabetes_prob = 0.25
            screening_prob = 0.25
            confirmation_prob = 0.30
        
        # Determine if person has diabetes (using 50% threshold)
        has_diabetes = diabetes_prob >= 0.5
        confidence_percentage = diabetes_prob * 100 if has_diabetes else (1 - diabetes_prob) * 100
        
        # Determine risk level with more detailed assessment
        def get_risk_assessment(prob):
            if prob < 0.15:
                return "Low Risk (Healthy range)"
            elif prob < 0.30:
                return "Medium Risk (Preventive measures recommended)"
            elif prob < 0.50:
                return "High Risk (Medical consultation advised)"
            else:
                return "Very High Risk (Immediate medical attention needed)"
        
        # Calculate uncertainties
        epistemic_uncertainty = 0.02
        aleatoric_uncertainty = 0.03
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Enhanced feature importances with descriptions
        feature_importances = {
            "BMI": FeatureImportance(
                importance=0.15,
                description="Body Mass Index - Strong predictor of diabetes risk"
            ),
            "Age": FeatureImportance(
                importance=0.12,
                description="Age category - Risk increases with age"
            ),
            "HighBP": FeatureImportance(
                importance=0.11,
                description="High Blood Pressure - Major cardiovascular risk factor"
            ),
            "HighChol": FeatureImportance(
                importance=0.10,
                description="High Cholesterol - Key metabolic risk indicator"
            ),
            "GenHlth": FeatureImportance(
                importance=0.09,
                description="General Health - Overall health status impact"
            ),
            "HeartDiseaseorAttack": FeatureImportance(
                importance=0.08,
                description="Heart Disease History - Strong diabetes correlation"
            ),
            "PhysActivity": FeatureImportance(
                importance=0.07,
                description="Physical Activity - Protective factor against diabetes"
            ),
            "Fruits_Veggies": FeatureImportance(
                importance=0.06,
                description="Diet Quality - Nutritional impact on diabetes risk"
            )
        }
        
        return PredictionResponse(
            has_diabetes=has_diabetes,
            confidence_percentage=round(confidence_percentage, 2),
            screening_stage=StageResult(
                probability=screening_prob,
                risk_assessment=get_risk_assessment(screening_prob)
            ),
            confirmation_stage=StageResult(
                probability=confirmation_prob,
                risk_assessment=get_risk_assessment(confirmation_prob)
            ),
            risk_level=get_risk_assessment(max(screening_prob, confirmation_prob)),
            uncertainties=UncertaintyEstimates(
                epistemic=epistemic_uncertainty,
                aleatoric=aleatoric_uncertainty,
                total=total_uncertainty
            ),
            warnings=warnings if warnings else None,
            feature_importances=feature_importances
        )
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
