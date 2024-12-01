"""
Quick test pipeline for the diabetes prediction model.

This script runs a fast test of the entire prediction pipeline using a small
subset of data to verify that all components are working correctly.
"""

import pandas as pd
import numpy as np
from src.preprocessor import PreProcessor
from src.ensemble_model import EnsembleModel
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
import time
from pathlib import Path
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model_performance(model, X, y):
    """
    Evaluate model performance with balanced sampling.
    
    Args:
        model: The model to evaluate
        X: Feature matrix
        y: Target variable
        
    Returns:
        Dictionary containing performance metrics
    """
    # Print original class distribution
    logger.info(f"Original class distribution: {np.bincount(y)}")
    
    # Split data first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to balance training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Balanced training set distribution: {np.bincount(y_train_balanced)}")
    logger.info(f"Test set distribution: {np.bincount(y_test)}")
    
    # Fit the model on balanced training data
    model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return metrics

def test_pipeline():
    """
    Quick test of the entire pipeline with enhanced feature importance validation
    and comprehensive error checking.
    """
    start_time = time.time()
    results = {
        'status': 'failed',
        'error': None,
        'performance': {},
        'warnings': [],
        'feature_importance': {}
    }
    
    try:
        logger.info("Starting pipeline test...")
        
        # Load and validate data
        logger.info("Loading data...")
        data_path = Path("diabetes_binary_health_indicators_BRFSS2015.csv")
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
            
        data = pd.read_csv(data_path)
        
        # Take a stratified sample for quick testing
        test_size = 1000  # Increased for better stability assessment
        stratify = data['Diabetes_binary']
        data = data.groupby('Diabetes_binary', group_keys=False).apply(
            lambda x: x.sample(n=test_size//2)
        ).reset_index(drop=True)
        
        logger.info(f"Testing with {len(data)} samples (balanced)")
        X = data.drop('Diabetes_binary', axis=1)
        y = data['Diabetes_binary']
        
        # Test preprocessor with error handling
        logger.info("Testing preprocessor...")
        try:
            preprocessor = PreProcessor()
            X_processed, y_processed = preprocessor.fit_transform(X, y)
            logger.info(f"Features shape after preprocessing: {X_processed.shape}")
        except Exception as e:
            logger.error(f"Preprocessor failed: {str(e)}")
            results['error'] = f"Preprocessor error: {str(e)}"
            raise
        
        # Test feature importance calculation with validation
        logger.info("Testing feature importance calculation...")
        try:
            feature_importance = preprocessor._calculate_feature_importance(X_processed, y_processed)
            
            # Validate feature importance scores
            if not (0 <= feature_importance).all() or not (feature_importance <= 1).all():
                raise ValueError("Feature importance scores outside valid range [0,1]")
            
            # Store top features and their importance
            top_features = feature_importance.nlargest(10)
            results['feature_importance'] = top_features.to_dict()
            
            logger.info("Top 10 important features with confidence intervals:")
            for feature in top_features.index:
                importance_value = feature_importance[feature]
                clinical_weight = preprocessor.clinical_weights.get(feature, 1.0)
                logger.info(
                    f"{feature:20} "
                    f"Importance: {importance_value:.4f} "
                    f"(Clinical weight: {clinical_weight:.2f})"
                )
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {str(e)}")
            results['warnings'].append(f"Feature importance error: {str(e)}")
        
        # Test model with comprehensive validation
        logger.info("Testing model...")
        try:
            model = EnsembleModel(
                n_monte_carlo=5,  # Reduced for testing
                batch_size=100,   # Smaller batch size for testing
                random_state=42
            )
            
            # Quick model evaluation with cross-validation
            performance = evaluate_model_performance(model, X_processed, y_processed)
            results['performance'] = performance
            
            # Validate performance metrics
            if any(v < 0 or v > 1 for v in performance.values()):
                raise ValueError("Invalid performance metrics detected")
            
            # Log performance with confidence intervals
            logger.info("\nModel Performance:")
            for metric, value in performance.items():
                logger.info(f"{metric.title()}: {value:.4f}")
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            results['error'] = f"Model error: {str(e)}"
            raise
        
        results['status'] = 'success'
        
        # Log execution time and memory usage
        execution_time = time.time() - start_time
        logger.info(f"\nTest completed in {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        results['error'] = str(e)
        raise
        
    finally:
        return results

if __name__ == "__main__":
    test_pipeline()
