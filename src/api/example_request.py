import requests
import json

# Example health indicators data
sample_data = {
    "BMI": 28.5,           # Body Mass Index
    "Age": 2,              # Age category (2: 25-29 years)
    "GenHlth": 2,          # General Health (1=excellent to 5=poor)
    "MentHlth": 0,         # Days of poor mental health
    "PhysHlth": 0,         # Days of poor physical health
    "HighBP": 1,           # High Blood Pressure
    "HighChol": 1,         # High Cholesterol
    "CholCheck": 1,        # Cholesterol Check in past 5 years
    "Smoker": 0,           # Smoking Status
    "Stroke": 0,           # History of Stroke
    "HeartDiseaseorAttack": 0,  # History of Heart Disease
    "PhysActivity": 1,     # Physical Activity
    "Fruits": 1,           # Fruit Consumption
    "Veggies": 1,          # Vegetable Consumption
    "HvyAlcoholConsump": 0,# Heavy Alcohol Consumption
    "AnyHealthcare": 1,    # Has Healthcare Coverage
    "NoDocbcCost": 0,      # No Doctor due to Cost
    "DiffWalk": 0,         # Difficulty Walking
    "Sex": 1,              # Sex (0=female, 1=male)
    "Education": 6,        # Education Level
    "Income": 7            # Income Level
}

def test_health():
    """Test if the API is healthy and ready to make predictions"""
    url = "http://localhost:8000/health"
    try:
        response = requests.get(url)
        print("\nHealth Check:")
        print(response.json())
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
        return False

def test_prediction():
    """Test the prediction endpoint with sample data"""
    if not test_health():
        return

    url = "http://localhost:8000/predict"
    try:
        response = requests.post(url, json=sample_data)
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Results:")
            print("\nScreening Stage:")
            print(f"Probability: {result['screening_stage']['probability']:.2%}")
            print(f"Assessment: {result['screening_stage']['risk_assessment']}")
            
            print("\nConfirmation Stage:")
            print(f"Probability: {result['confirmation_stage']['probability']:.2%}")
            print(f"Assessment: {result['confirmation_stage']['risk_assessment']}")
            
            print(f"\nOverall Risk Level: {result['risk_level']}")
            
            print("\nUncertainty Estimates:")
            print(f"Epistemic: {result['uncertainties']['epistemic']:.4f}")
            print(f"Aleatoric: {result['uncertainties']['aleatoric']:.4f}")
            print(f"Total: {result['uncertainties']['total']:.4f}")
            
            if result.get('warnings'):
                print("\nWarnings:")
                for warning in result['warnings']:
                    print(f"- {warning}")
                    
            if result.get('feature_importances'):
                print("\nTop 5 Most Important Features:")
                sorted_features = sorted(result['feature_importances'].items(), 
                                      key=lambda x: abs(x[1]), reverse=True)[:5]
                for feature, importance in sorted_features:
                    print(f"{feature}: {importance:.4f}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")

if __name__ == "__main__":
    test_prediction()
