import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, log_loss, balanced_accuracy_score
)
import argparse
import joblib
from pathlib import Path
import json
import matplotlib.pyplot as plt

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
from typing import Dict

def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare the diabetes dataset."""
    try:
        # Read the CSV file
        data = pd.read_csv(data_path)
        
        # Split features and target
        X = data.drop('Diabetes_binary', axis=1)
        y = data['Diabetes_binary']
        
        # Print dataset information
        print("\nTraining set size:", len(X))
        print("Test set size:", int(len(X) * 0.15))
        
        print("\nClass distribution in training set:")
        print("Non-diabetic (0):", sum(y == 0))
        print("Diabetic (1):", sum(y == 1))
        
        return train_test_split(X, y, test_size=0.15, random_state=42)
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def create_balanced_test_set(X, y, test_size=0.2, random_state=42):
    """Create a balanced test set while maintaining original distribution in training."""
    # Convert to numpy arrays if needed
    X = np.array(X) if isinstance(X, pd.DataFrame) else X
    y = np.array(y) if isinstance(y, pd.Series) else y
    
    # Find the minority class
    minority_class = 1 if np.sum(y == 1) < np.sum(y == 0) else 0
    minority_count = np.sum(y == minority_class)
    
    # Calculate how many samples we want in test set for each class
    n_test_per_class = int(min(
        minority_count * test_size,  # Don't take more than test_size% of minority class
        len(y) * test_size / 2  # Don't take more than half of desired test size per class
    ))
    
    # Split each class separately
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    y_0 = y[y == 0]
    y_1 = y[y == 1]
    
    # Random state for reproducibility
    rng = np.random.RandomState(random_state)
    
    # Randomly select equal numbers from each class for test set
    test_idx_0 = rng.choice(len(X_0), n_test_per_class, replace=False)
    test_idx_1 = rng.choice(len(X_1), n_test_per_class, replace=False)
    
    # Create test set
    X_test = np.vstack([X_0[test_idx_0], X_1[test_idx_1]])
    y_test = np.hstack([y_0[test_idx_0], y_1[test_idx_1]])
    
    # Create training set from remaining samples
    train_idx_0 = np.setdiff1d(np.arange(len(X_0)), test_idx_0)
    train_idx_1 = np.setdiff1d(np.arange(len(X_1)), test_idx_1)
    
    X_train = np.vstack([X_0[train_idx_0], X_1[train_idx_1]])
    y_train = np.hstack([y_0[train_idx_0], y_1[train_idx_1]])
    
    # Shuffle both sets
    train_shuffle = rng.permutation(len(X_train))
    test_shuffle = rng.permutation(len(X_test))
    
    X_train = X_train[train_shuffle]
    y_train = y_train[train_shuffle]
    X_test = X_test[test_shuffle]
    y_test = y_test[test_shuffle]
    
    return X_train, X_test, y_train, y_test

def evaluate_predictions(y_true: np.ndarray,
                       probas: np.ndarray,
                       uncertainties: Dict,
                       thresholds: Dict[str, float]) -> Dict:
    """Evaluate model performance with comprehensive metrics."""
    # Get binary predictions using thresholds
    screen_mask = (probas[:, 1] >= thresholds['screening'])
    confirm_mask = (probas[:, 1] >= thresholds['confirmation'])
    
    # Screening predictions: All samples that pass screening threshold
    y_pred_screen = screen_mask.astype(int)
    
    # Confirmation predictions: Only samples that pass screening get confirmation predictions
    y_pred_confirm = np.zeros_like(y_true)
    y_pred_confirm[screen_mask] = confirm_mask[screen_mask].astype(int)
    
    # Calculate metrics for both stages
    metrics = {
        'screening_metrics': {
            'accuracy': float(accuracy_score(y_true, y_pred_screen)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred_screen)),
            'precision': float(precision_score(y_true, y_pred_screen)),
            'recall': float(recall_score(y_true, y_pred_screen)),
            'f1': float(f1_score(y_true, y_pred_screen)),
            'weighted_precision': float(precision_score(y_true, y_pred_screen, average='weighted')),
            'weighted_recall': float(recall_score(y_true, y_pred_screen, average='weighted')),
            'weighted_f1': float(f1_score(y_true, y_pred_screen, average='weighted'))
        },
        'confirmation_metrics': {
            'accuracy': float(accuracy_score(y_true, y_pred_confirm)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred_confirm)),
            'precision': float(precision_score(y_true, y_pred_confirm)),
            'recall': float(recall_score(y_true, y_pred_confirm)),
            'f1': float(f1_score(y_true, y_pred_confirm)),
            'weighted_precision': float(precision_score(y_true, y_pred_confirm, average='weighted')),
            'weighted_recall': float(recall_score(y_true, y_pred_confirm, average='weighted')),
            'weighted_f1': float(f1_score(y_true, y_pred_confirm, average='weighted'))
        },
        'probability_metrics': {
            'roc_auc': float(roc_auc_score(y_true, probas[:, 1])),
            'average_precision': float(average_precision_score(y_true, probas[:, 1])),
            'brier_score': float(brier_score_loss(y_true, probas[:, 1]))
        },
        'uncertainty_metrics': {
            'mean_epistemic': float(np.mean(uncertainties['epistemic'])),
            'mean_aleatoric': float(np.mean(uncertainties['aleatoric'])),
            'mean_total': float(np.mean(uncertainties['total'])),
            'correlation_with_error': float(np.corrcoef(
                uncertainties['total'],
                np.abs(y_true - probas[:, 1])
            )[0, 1])
        },
        'class_distribution': {
            'negative_class': int(np.sum(y_true == 0)),
            'positive_class': int(np.sum(y_true == 1)),
            'class_ratio': float(np.sum(y_true == 1) / len(y_true))
        }
    }
    
    return metrics

def train_model(data_path: str, visualize: bool = False, test_mode: bool = False) -> None:
    """Train the diabetes prediction model."""
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    if test_mode:
        print("\nRunning in test mode with smaller dataset...")
        # Take an even smaller subset (1%) for faster testing
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.99, random_state=42)
        for train_idx, _ in sss.split(X_train, y_train):
            X_train = X_train.iloc[train_idx]
            y_train = y_train.iloc[train_idx]
        
        # Also reduce test set size
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.99, random_state=42)
        for test_idx, _ in sss.split(X_test, y_test):
            X_test = X_test.iloc[test_idx]
            y_test = y_test.iloc[test_idx]
            
        print(f"Test mode dataset size - Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Create metrics directory if it doesn't exist
    Path('metrics').mkdir(exist_ok=True)
    
    # Create plots subdirectory in metrics if it doesn't exist
    Path('metrics/plots').mkdir(exist_ok=True)
    
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
    
    if visualize:
        print("\nGenerating pre-training visualizations...")
        viz.plot_class_distribution(y_train, y_train, title="Initial Class Distribution")
        plt.close()
    
    model.fit(X_train_processed, y_train, test_mode=test_mode)
    
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
    print("Class Distribution in Test Set:")
    print(f"Class 0 (No Diabetes): {metrics['class_distribution']['negative_class']}")
    print(f"Class 1 (Diabetes): {metrics['class_distribution']['positive_class']}")
    print(f"Positive Class Ratio: {metrics['class_distribution']['class_ratio']:.3f}\n")
    
    print("Screening Stage Metrics (Threshold: {:.1f})".format(model.screening_threshold))
    print(f"Accuracy: {metrics['screening_metrics']['accuracy']:.3f}")
    print(f"Balanced Accuracy: {metrics['screening_metrics']['balanced_accuracy']:.3f}")
    print(f"Precision: {metrics['screening_metrics']['precision']:.3f}")
    print(f"Weighted Precision: {metrics['screening_metrics']['weighted_precision']:.3f}")
    print(f"Recall: {metrics['screening_metrics']['recall']:.3f}")
    print(f"Weighted Recall: {metrics['screening_metrics']['weighted_recall']:.3f}")
    print(f"F1 Score: {metrics['screening_metrics']['f1']:.3f}")
    print(f"Weighted F1: {metrics['screening_metrics']['weighted_f1']:.3f}\n")
    
    print("Confirmation Stage Metrics (Threshold: {:.1f})".format(model.confirmation_threshold))
    print(f"Accuracy: {metrics['confirmation_metrics']['accuracy']:.3f}")
    print(f"Balanced Accuracy: {metrics['confirmation_metrics']['balanced_accuracy']:.3f}")
    print(f"Precision: {metrics['confirmation_metrics']['precision']:.3f}")
    print(f"Weighted Precision: {metrics['confirmation_metrics']['weighted_precision']:.3f}")
    print(f"Recall: {metrics['confirmation_metrics']['recall']:.3f}")
    print(f"Weighted Recall: {metrics['confirmation_metrics']['weighted_recall']:.3f}")
    print(f"F1 Score: {metrics['confirmation_metrics']['f1']:.3f}")
    print(f"Weighted F1: {metrics['confirmation_metrics']['weighted_f1']:.3f}\n")
    
    print("Probability Metrics:")
    print(f"ROC AUC: {metrics['probability_metrics']['roc_auc']:.3f}")
    print(f"Average Precision: {metrics['probability_metrics']['average_precision']:.3f}")
    print(f"Brier Score: {metrics['probability_metrics']['brier_score']:.3f}\n")
    
    if visualize:
        try:
            print("\nGenerating visualizations...")
            # Get feature importances from all models
            feature_names = preprocessor.get_feature_names()
            print(f"Feature names: {feature_names}")
            
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'LightGBM_Screen': model.get_feature_importance('lgb', 'screening'),
                'XGBoost_Screen': model.get_feature_importance('xgb', 'screening'),
                'CatBoost_Screen': model.get_feature_importance('cat', 'screening'),
                'LightGBM_Confirm': model.get_feature_importance('lgb', 'confirmation'),
                'XGBoost_Confirm': model.get_feature_importance('xgb', 'confirmation'),
                'CatBoost_Confirm': model.get_feature_importance('cat', 'confirmation')
            })
            
            print("\nFeature importance DataFrame:")
            print(feature_importance)
            
            print("\nSaving plots to metrics/plots directory...")
            
            print("1. Feature importance plot...")
            viz.plot_feature_importance(feature_importance)
            
            print("2. Uncertainty distribution plot...")
            viz.plot_uncertainty_distribution(probas[:, 1], uncertainties)
            
            print("3. Class distribution plot...")
            viz.plot_class_distribution(y_train, y_train)
            
            print("4. Metrics summary plot...")
            viz.plot_metrics_summary(metrics)
            
            print("5. Calibration analysis plot...")
            viz.plot_calibration_analysis(y_test, probas, n_bins=10)
            
            print("6. Class performance plot...")
            viz.plot_class_performance(y_test, model.predict(X_test_processed), probas)
            
            print("All visualizations generated successfully!")
            
        except Exception as e:
            print(f"\nError generating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nTraining complete! Model and preprocessor saved in 'models' directory.")

def predict(data_path: str) -> None:
    """Make predictions on new data."""
    try:
        # Load data
        data = pd.read_csv(data_path)
        
        # Load preprocessor and model
        preprocessor = joblib.load('models/preprocessor.joblib')
        model = joblib.load('models/model.joblib')
        
        # Ensure we have all required features
        for col in preprocessor.required_columns:
            if col not in data.columns:
                print(f"Warning: Adding missing column {col} with default value 0")
                data[col] = 0
        
        # Preprocess the data
        X = preprocessor.transform(data)
        
        # Get predictions and uncertainties
        probas, uncertainties = model.predict_proba(X)
        
        # Extract positive class probabilities (class 1)
        positive_probas = probas[:, 1]
        
        # Get binary predictions using model thresholds
        screening_preds = (positive_probas >= model.screening_threshold).astype(int)
        confirmation_preds = (positive_probas >= model.confirmation_threshold).astype(int)
        
        # Handle different uncertainty types
        uncertainty_values = uncertainties.get('epistemic', 
                           uncertainties.get('total', 
                           np.zeros_like(positive_probas)))
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Screening_Prediction': screening_preds,
            'Confirmation_Prediction': confirmation_preds,
            'Probability': positive_probas,
            'Uncertainty': uncertainty_values
        })
        
        # Save predictions
        results.to_csv('predictions.csv', index=False)
        print("\nPredictions saved to 'predictions.csv'")
        
        # Display sample predictions
        print("\nSample predictions:")
        print(results.head())
        
        return results
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        print("\nDebug information:")
        print(f"Data shape: {data.shape if 'data' in locals() else 'Not loaded'}")
        print(f"Preprocessed shape: {X.shape if 'X' in locals() else 'Not processed'}")
        if 'probas' in locals():
            print(f"Probabilities shape: {probas.shape}")
            print(f"Probability range: [{probas.min():.3f}, {probas.max():.3f}]")
        if 'uncertainties' in locals():
            print("Uncertainty types:", list(uncertainties.keys()))
        raise e

def small_test():
    """Run a small test of the pipeline with a subset of data."""
    print("\nRunning small test of the pipeline...")
    test_data_path = "diabetes_binary_health_indicators_BRFSS2015.csv"
    
    try:
        # Test training with small dataset
        train_model(test_data_path, visualize=False, test_mode=True)
        print("[PASS] Training test passed")
        
        # Test prediction with same file
        print("\nTesting prediction...")
        data = pd.read_csv(test_data_path)
        data = data.head(10)  # Only use first 10 rows for prediction test
        data.to_csv('test_predictions_input.csv', index=False)
        predict('test_predictions_input.csv')
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
        train_model('test_diabetes_binary_5k.csv', args.visualize, test_mode=True)
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
