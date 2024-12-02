import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess features for a single prediction."""
    try:
        # Load the preprocessor
        preprocessor = joblib.load('models/preprocessor.joblib')
        
        # Transform the data using our trained preprocessor
        X_processed = preprocessor.transform(df)
        
        return X_processed
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

def calculate_risk_score(row: pd.Series) -> float:
    """Calculate risk score based on medical factors."""
    score = 0.0
    
    # Age risk (higher age = higher risk)
    score += (row['Age'] - 30) * 0.3 if row['Age'] > 30 else 0
    
    # BMI risk
    if row['BMI'] >= 30:  # Obese
        score += 3
    elif row['BMI'] >= 25:  # Overweight
        score += 2
    
    # Medical conditions
    if row['HighBP'] == 1:
        score += 2
    if row['HighChol'] == 1:
        score += 2
    if row['HeartDiseaseorAttack'] == 1:
        score += 3
    if row['Stroke'] == 1:
        score += 3
    
    # Lifestyle factors
    if row['PhysActivity'] == 0:
        score += 1
    if row['Smoker'] == 1:
        score += 1
    if row['HvyAlcoholConsump'] == 1:
        score += 1
    
    # Health access and general health
    if row['AnyHealthcare'] == 0:
        score += 1
    score += row['GenHlth']  # Higher score = worse health
    
    return score

def format_prediction(probability: float, risk_score: float) -> str:
    """Format the prediction results in a user-friendly way."""
    # Calculate confidence percentage (how far from decision boundary of 0.5)
    confidence_pct = (1 - 2 * abs(probability - 0.5)) * 100
    
    if probability >= 0.5:
        risk_level = "High"
        recommendations = [
            "Schedule an appointment with your healthcare provider",
            "Get your blood sugar levels tested",
            "Consider lifestyle changes to reduce risk factors",
            "Monitor your blood pressure and cholesterol"
        ]
    elif probability >= 0.3:
        risk_level = "Medium"
        recommendations = [
            "Discuss your risk factors with your doctor",
            "Consider regular health screenings",
            "Focus on maintaining a healthy lifestyle",
            "Monitor your health metrics regularly"
        ]
    else:
        risk_level = "Low"
        recommendations = [
            "Continue maintaining a healthy lifestyle",
            "Have regular check-ups",
            "Stay physically active",
            "Maintain a balanced diet"
        ]
    
    recommendations_text = "\n".join(f"- {rec}" for rec in recommendations)
    
    result = f"""
Diabetes Risk Assessment Results:
-------------------------------
Risk Probability: {probability:.1%}
Model Confidence: {confidence_pct:.1f}%
Risk Level: {risk_level}
Health Risk Score: {risk_score:.1f} (higher score = higher risk, range 0-20)

Interpretation:
-------------
- Based on your health information, you have a {probability:.1%} chance of having diabetes
- The model is {confidence_pct:.1f}% confident in this assessment
- Your health risk factors contribute to a score of {risk_score:.1f} out of 20
- This puts you in the {risk_level.lower()} risk category

Recommendations:
--------------
{recommendations_text}

Note: This is a screening tool and should not replace professional medical advice.
Please consult with a healthcare provider for proper diagnosis.
"""
    return result

def predict_diabetes_risk(user_data: Dict) -> Tuple[float, float]:
    """Predict diabetes risk for a single user."""
    try:
        # Load the model
        model = joblib.load('models/model.joblib')
        
        # Convert user data to DataFrame
        df = pd.DataFrame([user_data])
        
        # Preprocess features
        X_processed = preprocess_features(df)
        
        # Get prediction probabilities and uncertainties
        probas, uncertainties = model.predict_proba(X_processed)
        
        # Calculate risk score
        risk_score = calculate_risk_score(df.iloc[0])
        
        return probas[0][1], risk_score  # Return probability of class 1 (diabetes)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return 0.0, 0.0

if __name__ == "__main__":
    # Example user data with proper age group (6 = Age 45-49)
    example_user = {
        'Age': 6,  # Age group 6 (45-49 years)
        'BMI': 28.5,
        'HighBP': 1,
        'HighChol': 1,
        'Smoker': 0,
        'Stroke': 0,
        'HeartDiseaseorAttack': 0,
        'PhysActivity': 1,
        'Fruits': 1,
        'Veggies': 1,
        'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1,
        'NoDocbcCost': 0,
        'GenHlth': 2,
        'PhysHlth': 3,
        'DiffWalk': 0,
        'MentHlth': 0,
        'Education': 4,  # College 1-3 years
        'Income': 5      # $35,000 to less than $50,000
    }
    
    # Get prediction
    probability, risk_score = predict_diabetes_risk(example_user)
    
    # Format and print results
    print(format_prediction(probability, risk_score))
