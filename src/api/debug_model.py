import os
import joblib
import pandas as pd
import numpy as np

# Get the model and preprocessor paths
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(src_path, "models", "diabetes_model.joblib")
preprocessor_path = os.path.join(src_path, "models", "preprocessor.joblib")

print("Loading model from:", model_path)
print("Loading preprocessor from:", preprocessor_path)

# Load model and preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

print("\nModel type:", type(model))
print("Preprocessor type:", type(preprocessor))

# Create sample input
sample_data = {
    "BMI": 28.5,
    "Age": 2,
    "GenHlth": 2,
    "MentHlth": 0,
    "PhysHlth": 0,
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 1,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "DiffWalk": 0,
    "Sex": 1,
    "Education": 6,
    "Income": 7
}

# Define feature order
input_features = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# Create DataFrame with proper feature order
input_data = pd.DataFrame([sample_data])[input_features]

print("\nInput data shape:", input_data.shape)
print("Input features:", input_data.columns.tolist())
print("Input values:", input_data.values.tolist())

# Try preprocessing
try:
    processed_data = preprocessor.transform(input_data)
    print("\nProcessed data shape:", processed_data.shape)
    print("Processed data:", processed_data)
except Exception as e:
    print("\nError during preprocessing:", str(e))
    raise

# Try prediction
try:
    probabilities = model.predict_proba(processed_data)
    print("\nPrediction probabilities shape:", probabilities.shape)
    print("Prediction probabilities:", probabilities)
except Exception as e:
    print("\nError during prediction:", str(e))
    raise
