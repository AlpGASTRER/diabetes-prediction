import requests
import json

def print_section(title, content, indent=0):
    """Helper function to print sections with proper formatting"""
    indent_str = " " * indent
    print(f"\n{indent_str}{'-' * 20} {title} {'-' * 20}")
    if isinstance(content, dict):
        for key, value in content.items():
            if isinstance(value, dict):
                print(f"{indent_str}{key}:")
                for subkey, subvalue in value.items():
                    print(f"{indent_str}  {subkey}: {subvalue}")
            else:
                print(f"{indent_str}{key}: {value}")
    elif isinstance(content, list):
        for item in content:
            print(f"{indent_str}• {item}")
    else:
        print(f"{indent_str}{content}")

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
        "PhysActivity": 0,  # Added to trigger lifestyle warning
        "Fruits": 0,        # Added to trigger dietary warning
        "Veggies": 0,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 15,    # Added to trigger mental health warning
        "PhysHlth": 0,
        "DiffWalk": 1,     # Added to trigger mobility warning
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
            
            # Print main prediction
            print_section("DIABETES PREDICTION", {
                "Status": "Positive" if result["has_diabetes"] else "Negative",
                "Confidence": f"{result['confidence_percentage']}%",
                "Risk Level": result["risk_level"]
            })
            
            # Print health warnings
            print_section("HEALTH WARNINGS", result.get("warnings", ["No warnings"]))
            
            # Print uncertainties
            print_section("PREDICTION UNCERTAINTY", {
                "Total Uncertainty": f"{result['uncertainties']['total'] * 100:.1f}%",
                "Model Uncertainty": f"{result['uncertainties']['epistemic'] * 100:.1f}%",
                "Data Uncertainty": f"{result['uncertainties']['aleatoric'] * 100:.1f}%"
            })
            
            # Print risk factors
            print_section("RISK FACTORS (Ordered by Impact)")
            sorted_factors = sorted(
                result["feature_importances"].items(),
                key=lambda x: x[1]['importance'],
                reverse=True
            )
            for factor, info in sorted_factors:
                print(f"\n• {factor}:")
                print(f"  Impact: {info['importance'] * 100:.1f}%")
                print(f"  Description: {info['description']}")
            
        else:
            print("Error Response:")
            print(response.text)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        if 'response' in locals() and response.status_code == 200:
            print("\nRaw response for debugging:")
            print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_health()
    test_prediction()
