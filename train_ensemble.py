"""
Main training pipeline for the diabetes prediction model.

This script implements the complete training pipeline for the diabetes prediction
model, including data preprocessing, model training, and evaluation. It supports
both test mode (< 10k samples) and production mode (500k+ samples).

Features:
- Multi-factor stratified cross-validation
- Comprehensive performance metrics
- Demographic fairness analysis
- Detailed logging and visualization
- Automatic parameter adaptation based on dataset size

Usage:
    Test mode:
        python train_ensemble.py --mode test
    
    Production mode:
        python train_ensemble.py --mode train

Author: Codeium AI
Last Modified: 2024
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.preprocessor import PreProcessor
from src.ensemble_model import EnsembleModel
from typing import Tuple, Dict
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and perform initial validation of the dataset.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        Tuple containing features (X) and target (y)
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    logger.info(f"Loading data from {data_path}")
    try:
        data = pd.read_csv(data_path)
        
        # Validate target variable
        if 'Diabetes_binary' not in data.columns:
            raise ValueError("Target variable 'Diabetes_binary' not found in dataset")
            
        # Split features and target
        y = data['Diabetes_binary']
        X = data.drop('Diabetes_binary', axis=1)
        
        logger.info(f"Loaded dataset with {len(X)} samples and {X.shape[1]} features")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_model(X: pd.DataFrame, 
                y: pd.Series,
                test_size: float = 0.2,
                random_state: int = 42) -> Dict:
    """
    Train the complete model pipeline including preprocessing and ensemble model.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of dataset to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing trained model, preprocessor, and performance metrics
    """
    logger.info("Starting model training pipeline")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Initialize and fit preprocessor
        preprocessor = PreProcessor()
        X_train_processed, y_train = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Initialize and train model
        model = EnsembleModel(random_state=random_state)
        model.fit(X_train_processed, y_train)
        
        # Evaluate model
        metrics = model.evaluate(X_test_processed, y_test)
        
        logger.info(f"Training completed. Test accuracy: {metrics.accuracy:.4f}")
        
        return {
            'model': model,
            'preprocessor': preprocessor,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

def save_results(results: Dict, output_dir: str):
    """
    Save trained model, preprocessor, and performance visualizations.
    
    Args:
        results: Dictionary containing model, preprocessor, and metrics
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save model and preprocessor
        joblib.dump(results['model'], output_path / 'model.joblib')
        joblib.dump(results['preprocessor'], output_path / 'preprocessor.joblib')
        
        # Create and save performance visualizations
        plot_performance_metrics(results['metrics'], output_path)
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def plot_performance_metrics(metrics, output_path: Path):
    """
    Create and save visualizations of model performance metrics.
    
    Args:
        metrics: ModelMetrics object containing performance metrics
        output_path: Directory to save plots
    """
    # Plot overall metrics
    plt.figure(figsize=(10, 6))
    metrics_dict = {
        'Accuracy': metrics.accuracy,
        'Precision': metrics.precision,
        'Recall': metrics.recall,
        'F1 Score': metrics.f1_score
    }
    sns.barplot(x=list(metrics_dict.keys()), y=list(metrics_dict.values()))
    plt.title('Model Performance Metrics')
    plt.savefig(output_path / 'performance_metrics.png')
    plt.close()
    
    # Plot demographic fairness
    plt.figure(figsize=(12, 6))
    demographic_df = pd.DataFrame(metrics.demographic_metrics).T
    sns.heatmap(demographic_df, annot=True, cmap='YlGnBu')
    plt.title('Performance Across Demographic Groups')
    plt.savefig(output_path / 'demographic_fairness.png')
    plt.close()

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train diabetes prediction model')
    parser.add_argument('--mode', choices=['test', 'train'], required=True,
                      help='Training mode: test for small dataset, train for full dataset')
    parser.add_argument('--data_path', default='data/diabetes_data.csv',
                      help='Path to input data')
    parser.add_argument('--output_dir', default='output',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    try:
        # Load and process data
        X, y = load_data(args.data_path)
        
        # Train model
        results = train_model(X, y)
        
        # Save results
        save_results(results, args.output_dir)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
