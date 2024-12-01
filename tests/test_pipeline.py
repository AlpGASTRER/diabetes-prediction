"""
Test suite for the diabetes prediction pipeline.
Tests data preprocessing, model training, and prediction functionality.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, recall_score
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessor import PreProcessor
from src.ensemble_model import DiabetesEnsemblePredictor

class TestDiabetesPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 1000
        
        # Create data with all required columns
        cls.test_data = pd.DataFrame({
            'Age': np.random.randint(18, 90, n_samples),
            'BMI': np.random.normal(28, 6, n_samples),
            'HighBP': np.random.binomial(1, 0.3, n_samples),
            'HighChol': np.random.binomial(1, 0.3, n_samples),
            'CholCheck': np.random.binomial(1, 0.8, n_samples),
            'Smoker': np.random.binomial(1, 0.2, n_samples),
            'Stroke': np.random.binomial(1, 0.05, n_samples),
            'HeartDiseaseorAttack': np.random.binomial(1, 0.1, n_samples),
            'PhysActivity': np.random.binomial(1, 0.7, n_samples),
            'Fruits': np.random.binomial(1, 0.6, n_samples),
            'Veggies': np.random.binomial(1, 0.7, n_samples),
            'HvyAlcoholConsump': np.random.binomial(1, 0.1, n_samples),
            'AnyHealthcare': np.random.binomial(1, 0.9, n_samples),
            'NoDocbcCost': np.random.binomial(1, 0.1, n_samples),
            'GenHlth': np.random.randint(1, 6, n_samples),
            'MentHlth': np.random.randint(0, 31, n_samples),
            'PhysHlth': np.random.randint(0, 31, n_samples),
            'DiffWalk': np.random.binomial(1, 0.2, n_samples),
            'Education': np.random.randint(1, 7, n_samples),
            'Income': np.random.randint(1, 9, n_samples),
            'Diabetes_binary': np.random.binomial(1, 0.3, n_samples)
        })
        
        # Split features and target
        cls.X = cls.test_data.drop('Diabetes_binary', axis=1)
        cls.y = cls.test_data['Diabetes_binary']
    
    def test_preprocessor(self):
        """Test medical preprocessor functionality."""
        preprocessor = PreProcessor()
        X_processed, y_processed = preprocessor.fit_transform(self.X, self.y)
        
        # Check if values are standardized for continuous features
        continuous_cols = ['BMI', 'Age', 'PhysHlth']
        for col in continuous_cols:
            stats = X_processed[col].describe()
            self.assertTrue(abs(stats['mean']) < 0.5)  # Approximately centered
            self.assertTrue(0.5 < stats['std'] < 2.0)  # Approximately scaled
    
    def test_ensemble_model(self):
        """Test ensemble model training and prediction."""
        # Preprocess data
        preprocessor = PreProcessor()
        X_processed, y_processed = preprocessor.fit_transform(self.X, self.y)
        
        # Train model with reduced calibration models for testing
        model = DiabetesEnsemblePredictor(n_calibration_models=2)
        model.fit(X_processed, y_processed)
        
        # Test predictions with uncertainty
        probas, uncertainties = model.predict_proba_with_uncertainty(X_processed)
        
        # Check prediction shapes and values
        self.assertEqual(len(probas), len(X_processed))  # Compare with processed data length
        self.assertTrue(all(0 <= p <= 1 for p in probas))  # Check probability bounds
    
    def test_duplicate_handling(self):
        """Test handling of duplicate records."""
        # Create data with duplicates
        duplicate_data = pd.concat([self.X] * 2, ignore_index=True)
        duplicate_y = pd.concat([self.y] * 2, ignore_index=True)
        
        # Test duplicate removal
        preprocessor = PreProcessor()
        processed_X, processed_y = preprocessor.fit_transform(duplicate_data, duplicate_y)
        self.assertLess(len(processed_X), len(duplicate_data))

    def test_clinical_thresholds(self):
        """Test clinical threshold validation."""
        # Create data with out-of-range values
        bad_data = self.X.copy()
        bad_data.loc[0, 'BMI'] = 100  # Unrealistic BMI
        
        preprocessor = PreProcessor()
        processed_X, _ = preprocessor.fit_transform(bad_data, self.y)
        
        # Check if out-of-range BMI was handled
        self.assertLess(
            processed_X['BMI'].max(),
            preprocessor.medical_thresholds['BMI']['high']
        )

    def test_uncertainty_calibration(self):
        """Test uncertainty estimation calibration."""
        # Process data
        preprocessor = PreProcessor()
        X_processed, y_processed = preprocessor.fit_transform(self.X, self.y)
        
        # Train model
        model = DiabetesEnsemblePredictor(n_calibration_models=2)
        model.fit(X_processed, y_processed)
        
        # Get predictions with uncertainty
        probas, uncertainties = model.predict_proba_with_uncertainty(X_processed)
        
        # Check if higher uncertainty correlates with predictions closer to 0.5
        distance_from_half = np.abs(probas - 0.5)
        
        # Convert uncertainties to array if it's a dictionary
        if isinstance(uncertainties, dict):
            uncertainty_values = uncertainties['total']
            if isinstance(uncertainty_values, pd.Series):
                uncertainty_values = uncertainty_values.values
        else:
            uncertainty_values = uncertainties
            
        # Ensure both arrays have same shape
        if len(uncertainty_values) != len(distance_from_half):
            raise ValueError(f"Shape mismatch: uncertainties ({len(uncertainty_values)}) vs distances ({len(distance_from_half)})")
            
        # Reshape arrays if needed
        uncertainty_values = np.array(uncertainty_values).reshape(-1)
        distance_from_half = np.array(distance_from_half).reshape(-1)
        
        # Calculate correlation between uncertainty and distance from decision boundary
        correlation = np.corrcoef(uncertainty_values, distance_from_half)[0, 1]
        
        # Higher uncertainty should correlate with predictions closer to 0.5
        self.assertLess(correlation, 0)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        preprocessor = PreProcessor()
        X_processed, y_processed = preprocessor.fit_transform(self.X, self.y)
        
        # Check if feature importance is calculated
        importance_scores = preprocessor.get_feature_importance()
        self.assertIsNotNone(importance_scores)
        self.assertTrue(len(importance_scores) > 0)
        
        # Check if scores sum to approximately 1
        self.assertAlmostEqual(sum(importance_scores.values()), 1.0, places=2)
    
    def test_batch_processing(self):
        """Test batch processing capability."""
        # Create larger dataset
        large_data = pd.concat([self.X] * 10)  # Create larger dataset
        large_y = pd.concat([self.y] * 10)
        batch_size = 1000
        
        # Initialize and fit preprocessor
        preprocessor = PreProcessor()
        preprocessor.fit(self.X.copy(), self.y.copy())  # Fit on original data
        
        # Process data in batches
        processed_batches = []
        for i in range(0, len(large_data), batch_size):
            batch_X = large_data.iloc[i:i + batch_size]
            batch_processed = preprocessor.transform(batch_X)
            processed_batches.append(batch_processed)
            
        # Combine processed batches
        X_processed = pd.concat(processed_batches)
        
        # Verify results
        self.assertEqual(len(X_processed), len(large_data))
        self.assertTrue(all(col in X_processed.columns 
                          for col in preprocessor.continuous_features))
    
    def test_medical_risk_scores(self):
        """Test medical risk score calculations."""
        preprocessor = PreProcessor()
        X_processed, y_processed = preprocessor.fit_transform(self.X, self.y)
        
        # Check medical score columns
        medical_score_cols = [col for col in X_processed.columns 
                            if any(term in col.lower() 
                                  for term in ['risk', 'score', 'index'])]
        self.assertTrue(len(medical_score_cols) > 0)
        
        # Verify medical scores are within valid ranges
        for col in medical_score_cols:
            col_values = X_processed[col].dropna()
            self.assertTrue(col_values.between(0, 1).all())

class TestTrainedModel(unittest.TestCase):
    """Tests for evaluating trained model performance."""
    
    @classmethod
    def setUpClass(cls):
        """Load trained model and test data."""
        try:
            cls.model = joblib.load('models/model.joblib')
            cls.preprocessor = joblib.load('models/preprocessor.joblib')
            
            # Load a small test set
            test_data = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
            test_data = test_data.sample(n=1000, random_state=42)  # Use small sample for quick testing
            
            cls.X_test = test_data.drop('Diabetes_binary', axis=1)
            cls.y_test = test_data['Diabetes_binary']
            
        except FileNotFoundError:
            raise unittest.SkipTest("Trained model not found. Run training first.")
    
    def test_class_balance(self):
        """Test if model performs well on both classes."""
        # Process test data
        X_proc = self.preprocessor.transform(self.X_test)
        
        # Get predictions
        probas, uncertainties = self.model.predict_proba_with_uncertainty(X_proc)
        predictions = (probas >= self.model.confirmation_threshold).astype(int)
        
        # Calculate metrics for each class
        for class_label in [0, 1]:
            mask = self.y_test == class_label
            class_preds = predictions[mask]
            class_true = self.y_test[mask]
            
            accuracy = accuracy_score(class_true, class_preds)
            recall = recall_score(class_true, class_preds, pos_label=class_label)
            
            # Assert balanced performance
            self.assertGreater(accuracy, 0.7, f"Low accuracy for class {class_label}")
            self.assertGreater(recall, 0.6, f"Low recall for class {class_label}")
    
    def test_uncertainty_calibration(self):
        """Test if uncertainty estimates are well-calibrated."""
        # Process test data
        X_proc = self.preprocessor.transform(self.X_test)
        
        # Get predictions with uncertainty
        probas, uncertainties = self.model.predict_proba_with_uncertainty(X_proc)
        predictions = (probas >= self.model.confirmation_threshold).astype(int)
        
        # Check if higher confidence correlates with better accuracy
        high_conf_mask = uncertainties['total'] < np.median(uncertainties['total'])
        low_conf_mask = ~high_conf_mask
        
        high_conf_acc = accuracy_score(self.y_test[high_conf_mask], predictions[high_conf_mask])
        low_conf_acc = accuracy_score(self.y_test[low_conf_mask], predictions[low_conf_mask])
        
        self.assertGreater(high_conf_acc, low_conf_acc, 
                          "High confidence predictions should be more accurate")
    
    def test_clinical_relevance(self):
        """Test if model predictions align with clinical risk factors."""
        # Process test data
        X_proc = self.preprocessor.transform(self.X_test)
        
        # Get predictions
        probas, _ = self.model.predict_proba_with_uncertainty(X_proc)
        
        # Check if high-risk patients (with multiple conditions) get higher probabilities
        high_risk_mask = (
            (self.X_test['HighBP'] == 1) & 
            (self.X_test['HighChol'] == 1) & 
            (self.X_test['BMI'] > 30)
        )
        
        high_risk_probs = probas[high_risk_mask]
        low_risk_probs = probas[~high_risk_mask]
        
        self.assertGreater(np.mean(high_risk_probs), np.mean(low_risk_probs),
                          "Model should assign higher probabilities to high-risk patients")

if __name__ == '__main__':
    unittest.main()
