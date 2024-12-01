import pandas as pd
import numpy as np

def load_and_validate_data(data):
    """Load and validate input data"""
    required_columns = [
        'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 
        'GenHlth', 'PhysHlth', 'DiffWalk', 'Age'
    ]
    
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        raise ValueError("Input must be a dictionary or pandas DataFrame")
    
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return data

def format_prediction_output(predictions, probabilities):
    """Format model predictions and probabilities"""
    return {
        'prediction': int(predictions[0]),
        'probability': float(probabilities[0][1]),
        'prediction_label': 'Diabetes' if predictions[0] == 1 else 'No Diabetes'
    }

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss for better handling of hard examples"""
    try:
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
            
        if y_true.shape != y_pred.shape:
            raise ValueError("Shape mismatch between y_true and y_pred")
            
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        pt = np.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
        
        focal_weight = (1 - pt) ** gamma
        weighted_loss = -alpha_t * focal_weight * np.log(pt)
        
        return np.mean(weighted_loss)
        
    except Exception as e:
        print(f"Error in focal loss calculation: {str(e)}")
        return float('inf')

def uncertainty_score(probas):
    """Calculate prediction uncertainty using entropy"""
    try:
        if not isinstance(probas, np.ndarray):
            probas = np.array(probas)
        
        if probas.ndim != 2:
            raise ValueError("Input probabilities must be 2-dimensional")
            
        row_sums = np.sum(probas, axis=1)
        if not np.allclose(row_sums, 1, rtol=1e-5):
            probas = probas / row_sums[:, np.newaxis]
        
        probas = np.clip(probas, 1e-10, 1.0)
        
        entropy = -np.sum(probas * np.log2(probas), axis=1)
        max_entropy = -np.log2(1/probas.shape[1])
        
        return np.clip(entropy / max_entropy, 0, 1)
        
    except Exception as e:
        print(f"Error in uncertainty calculation: {str(e)}")
        return np.ones(len(probas))
