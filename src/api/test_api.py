import requests
import json

def test_health():
    try:
        response = requests.get("http://localhost:8000/health")
        print("\nHealth Check:")
        print(response.json())
    except Exception as e:
        print(f"Error in health check: {str(e)}")

def test_prediction():
    # Test data
    test_data = {
        "Age": 5,  # Age category 5 (45-49 years)
        "Sex": 1,
        "BMI": 32.0,
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
        "GenHlth": 3,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Education": 6,
        "Income": 3
    }

    try:
        print("\nSending prediction request...")
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Results:")
            print(json.dumps({
                "Diabetes Status": "Positive" if result["has_diabetes"] else "Negative",
                "Confidence": f"{result['confidence_percentage']}%",
                "Risk Level": result["risk_level"],
                "Warnings": result["warnings"] if result.get("warnings") else "None",
                "Uncertainties": {
                    "Total": f"{result['uncertainties']['total'] * 100:.1f}%",
                    "Epistemic": f"{result['uncertainties']['epistemic'] * 100:.1f}%",
                    "Aleatoric": f"{result['uncertainties']['aleatoric'] * 100:.1f}%"
                },
                "Top Risk Factors": dict(sorted(
                    result["feature_importances"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3])
            }, indent=2))
        else:
            print("Error Response:")
            print(response.text)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    test_health()
    test_prediction()
