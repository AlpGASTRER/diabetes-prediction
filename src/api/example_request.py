import requests
import json

# Example health indicators data
sample_data = {
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 1,
    "BMI": 28.5,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 2,
    "MentHlth": 0,
    "PhysHlth": 0,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 7,
    "Education": 6,
    "Income": 7
}

def test_prediction():
    # Make prediction request
    url = "http://localhost:8000/predict"
    
    try:
        response = requests.post(url, json=sample_data)
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Results:")
            print(f"Risk of Diabetes: {result['prediction']:.2%}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence Score: {result['confidence_score']:.2%}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")

if __name__ == "__main__":
    test_prediction()
