import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, log_loss
)
import argparse
import joblib
from pathlib import Path
import json

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from preprocessor import PreProcessor
from ensemble_model import DiabetesEnsemblePredictor
import visualization as viz
import matplotlib.pyplot as plt
from typing import Dict

def load_data(data_path: str) -> pd.DataFrame:
    """Load and validate the dataset."""
    try:
        data = pd.read_csv(data_path)
        required_columns = [
            'Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'Smoker',
            'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
            'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
            'GenHlth', 'PhysHlth', 'DiffWalk', 'Age'
        ]
        
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure all columns have proper data types
        categorical_cols = [
            'Diabetes_binary', 'HighBP', 'HighChol', 'Smoker',
            'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
            'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
            'DiffWalk'
        ]
        
        numeric_cols = ['BMI', 'Age', 'GenHlth', 'PhysHlth']
        
        # Convert categorical columns to int
        for col in categorical_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
            
        # Convert numeric columns to float
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
        # Handle missing values
        data = data.dropna()
            
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")

def evaluate_predictions(y_true: np.ndarray,
                       probas: np.ndarray,
                       uncertainties: Dict,
                       thresholds: Dict[str, float]) -> Dict:
    """Evaluate model performance with comprehensive metrics."""
    # Get binary predictions using thresholds
    y_pred_screen = (probas[:, 1] >= thresholds['screening']).astype(int)
    y_pred_confirm = (probas[:, 1] >= thresholds['confirmation']).astype(int)
    
    # Calculate metrics for both stages
    metrics = {
        'screening_metrics': {
            'accuracy': float(accuracy_score(y_true, y_pred_screen)),
            'precision': float(precision_score(y_true, y_pred_screen)),
            'recall': float(recall_score(y_true, y_pred_screen)),
            'f1': float(f1_score(y_true, y_pred_screen))
        },
        'confirmation_metrics': {
            'accuracy': float(accuracy_score(y_true, y_pred_confirm)),
            'precision': float(precision_score(y_true, y_pred_confirm)),
            'recall': float(recall_score(y_true, y_pred_confirm)),
            'f1': float(f1_score(y_true, y_pred_confirm))
        },
        'probability_metrics': {
            'roc_auc': float(roc_auc_score(y_true, probas[:, 1])),
            'average_precision': float(average_precision_score(y_true, probas[:, 1])),
            'brier_score': float(brier_score_loss(y_true, probas[:, 1]))
        }
    }
    
    # Calculate uncertainty metrics
    if isinstance(uncertainties, dict):
        metrics['uncertainty_metrics'] = {
            'mean_epistemic': float(np.mean(uncertainties['epistemic'])),
            'mean_aleatoric': float(np.mean(uncertainties['aleatoric'])),
            'mean_total': float(np.mean(uncertainties['total'])),
            'correlation_with_error': float(np.corrcoef(
                uncertainties['total'],
                np.abs(y_true - probas[:, 1])
            )[0, 1])
        }
    
    return metrics

def train_model(data_path: str, visualize: bool = False, test_mode: bool = False) -> None:
    """Train the diabetes prediction model."""
    print("Loading data...")
    data = load_data(data_path)
    
    if test_mode:
        print("\nRunning in test mode with smaller dataset...")
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=42)
        for train_idx, _ in sss.split(data, data['Diabetes_binary']):
            data = data.iloc[train_idx]
        print(f"Test dataset size: {len(data)}")
    
    # Create metrics directory if it doesn't exist
    Path('metrics').mkdir(exist_ok=True)
    
    # Split features and target
    X = data.drop('Diabetes_binary', axis=1)
    y = data['Diabetes_binary']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"\nClass distribution in training set:")
    print(f"Non-diabetic (0): {sum(y_train == 0)}")
    print(f"Diabetic (1): {sum(y_train == 1)}")
    
    # Preprocess data
    print("\nPreprocessing training data...")
    preprocessor = PreProcessor()
    X_train_processed, y_train = preprocessor.fit_transform(X_train, y_train)
    
    # Save preprocessor
    print("\nSaving preprocessor...")
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    # Train model
    print("\nTraining ensemble model...")
    model = DiabetesEnsemblePredictor()
    model.fit(X_train_processed, y_train)
    
    # Save model
    print("\nSaving model...")
    joblib.dump(model, 'models/model.joblib')
    
    # Preprocess test data
    print("\nEvaluating model...")
    X_test_processed = preprocessor.transform(X_test)
    
    # Get predictions and uncertainties
    probas, uncertainties = model.predict_proba(X_test_processed)
    
    # Evaluate model
    metrics = evaluate_predictions(
        y_test,
        probas,
        uncertainties,
        {'screening': model.screening_threshold, 'confirmation': model.confirmation_threshold}
    )
    
    # Save metrics
    with open('metrics/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print metrics
    print("\nModel Performance:\n")
    print("Screening Stage Metrics (Threshold: {:.1f})".format(model.screening_threshold))
    print(f"Accuracy: {metrics['screening_metrics']['accuracy']:.3f}")
    print(f"Precision: {metrics['screening_metrics']['precision']:.3f}")
    print(f"Recall: {metrics['screening_metrics']['recall']:.3f}")
    print(f"F1 Score: {metrics['screening_metrics']['f1']:.3f}\n")
    
    print("Confirmation Stage Metrics (Threshold: {:.1f})".format(model.confirmation_threshold))
    print(f"Accuracy: {metrics['confirmation_metrics']['accuracy']:.3f}")
    print(f"Precision: {metrics['confirmation_metrics']['precision']:.3f}")
    print(f"Recall: {metrics['confirmation_metrics']['recall']:.3f}")
    print(f"F1 Score: {metrics['confirmation_metrics']['f1']:.3f}\n")
    
    print("Probability Metrics:")
    print(f"ROC AUC: {metrics['probability_metrics']['roc_auc']:.3f}")
    print(f"Average Precision: {metrics['probability_metrics']['average_precision']:.3f}")
    print(f"Brier Score: {metrics['probability_metrics']['brier_score']:.3f}\n")
    
    if visualize:
        print("\nGenerating visualizations...")
        # Plot feature importance
        feature_importance = pd.DataFrame({
            'Feature': preprocessor.get_feature_names(),
            'Avg_Importance': np.mean([
                model.models['screening']['lgb'].feature_importances_,
                model.models['screening']['xgb'].feature_importances_,
                model.models['confirmation']['lgb'].feature_importances_,
                model.models['confirmation']['xgb'].feature_importances_
            ], axis=0),
            'Std_Importance': np.std([
                model.models['screening']['lgb'].feature_importances_,
                model.models['screening']['xgb'].feature_importances_,
                model.models['confirmation']['lgb'].feature_importances_,
                model.models['confirmation']['xgb'].feature_importances_
            ], axis=0)
        })
        viz.plot_feature_importance(feature_importance)
        
        # Plot uncertainty distribution
        viz.plot_uncertainty_distribution(probas[:, 1], uncertainties)
        
        # Plot calibration curve
        viz.plot_calibration_curve(y_test, probas[:, 1])
        
        # Plot class distribution
        viz.plot_class_distribution(y_train, y_train)
        
        # Plot metrics summary
        viz.plot_metrics_summary(metrics)
    
    print("\nTraining complete! Model and preprocessor saved in 'models' directory.")

def predict(data_path: str) -> None:
    """Make predictions on new data."""
    # Load model and preprocessor
    try:
        preprocessor = joblib.load('models/preprocessor.joblib')
        model = joblib.load('models/model.joblib')
    except FileNotFoundError:
        raise RuntimeError("Model files not found. Please train the model first.")
    
    # Load and preprocess data
    data = load_data(data_path)
    if 'Diabetes_binary' in data.columns:
        data = data.drop('Diabetes_binary', axis=1)
    
    X_proc = preprocessor.transform(data)
    
    # Make predictions
    probas, uncertainties = model.predict_proba(X_proc)
    
    # Calculate confidence scores
    def calculate_confidence(proba, uncertainty):
        # Distance from decision boundary (0.6)
        decision_distance = abs(proba - 0.6)
        # Normalize distance to [0, 1]
        normalized_distance = min(decision_distance / 0.4, 1.0)
        # Combine with uncertainty (weighted average)
        return 0.7 * (1 - uncertainty) + 0.3 * normalized_distance

    confidences = np.array([calculate_confidence(p, u) for p, u in zip(probas[:, 1], uncertainties['total'])])
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Probability': probas[:, 1],
        'Total_Uncertainty': uncertainties['total'],
        'Epistemic_Uncertainty': uncertainties['epistemic'],
        'Aleatoric_Uncertainty': uncertainties['aleatoric'],
        'Screening_Confidence': uncertainties['screening_confidence'],
        'Confirmation_Confidence': uncertainties['confirmation_confidence'],
        'Prediction': (probas[:, 1] >= 0.6).astype(int),
        'Confidence': confidences,
        'Risk_Level': pd.cut(confidences, 
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low Confidence', 'Medium Confidence', 'High Confidence', 'Very High Confidence'])
    })
    
    # Save results
    results.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to 'predictions.csv'")
    
    # Create visualization
    viz.create_interactive_dashboard(
        probas[:, 1], uncertainties['total'], None,
        {'screening': 0.2, 'confirmation': 0.6}  # Updated thresholds
    )

def small_test():
    """Run a small test of the pipeline with a subset of data."""
    print("\nRunning small test of the pipeline...")
    test_data_path = "diabetes_binary_health_indicators_BRFSS2015.csv"
    
    try:
        # Test training with small dataset
        train_model(test_data_path, visualize=False, test_mode=True)
        print("[PASS] Training test passed")
        
        # Test prediction
        predict(test_data_path)
        print("[PASS] Prediction test passed")
        
        print("\nAll tests passed successfully!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Diabetes Prediction Model with Uncertainty Estimation')
    parser.add_argument('--mode', choices=['train', 'predict', 'test'], default='train',
                      help='Mode to run the model in (train/predict/test)')
    parser.add_argument('--data', type=str, required=False,
                      help='Path to the dataset')
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualizations during training')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        small_test()
    elif args.mode == 'train':
        if not args.data:
            raise ValueError("Data path is required for training mode")
        train_model(args.data, args.visualize)
    elif args.mode == 'predict':
        if not args.data:
            raise ValueError("Data path is required for prediction mode")
        predict(args.data)

if __name__ == '__main__':
    main()
