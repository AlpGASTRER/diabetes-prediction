import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, Optional
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, recall_score, precision_score, brier_score_loss
from scipy.stats import entropy
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class DiabetesEnsemblePredictor:
    """Two-stage ensemble model for diabetes prediction with uncertainty estimation"""
    
    def __init__(self, 
                 screening_threshold: float = 0.12,  # Adjusted for better balance
                 confirmation_threshold: float = 0.15,  # Adjusted for better balance
                 n_calibration_models: int = 25,  # Increased for better calibration
                 mc_iterations: int = 35,
                 threshold: float = 0.5):
        """Initialize the ensemble predictor."""
        self.screening_threshold = screening_threshold
        self.confirmation_threshold = confirmation_threshold
        self.n_calibration_models = n_calibration_models
        self.mc_iterations = mc_iterations
        self.threshold = threshold
        
        # Initialize model containers
        self.models = {}
        self.calibrated_models = {}
        self.model_weights = {}
        
        for stage in ['screening', 'confirmation']:
            self.models[stage] = {}
            self.calibrated_models[stage] = {}
            self.model_weights[stage] = {}
        
        # Initialize performance tracking for dynamic weights
        self.model_performance = {'screening': {}, 'confirmation': {}}
        self.range_performance = {'screening': {}, 'confirmation': {}}
        
        # Define prediction ranges for range-specific weights
        self.pred_ranges = [
            (0.0, 0.2),   # Very low risk
            (0.2, 0.4),   # Low risk
            (0.4, 0.6),   # Medium risk
            (0.6, 0.8),   # High risk
            (0.8, 1.0)    # Very high risk
        ]
        
        self.is_fitted = False
    
    def _create_base_models(self, stage: str, test_mode: bool = False) -> Dict:
        """Create base models for the ensemble with balanced parameters."""
        if test_mode:
            # Lightweight models for testing
            models = {
                'lgb': lgb.LGBMClassifier(
                    n_estimators=10,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42
                ),
                'xgb': xgb.XGBClassifier(
                    n_estimators=10,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                ),
                'cat': CatBoostClassifier(
                    iterations=10,
                    learning_rate=0.1,
                    depth=3,
                    random_seed=42,
                    verbose=False
                )
            }
            return models
            
        if stage == 'screening':
            models = {
                'lgb': lgb.LGBMClassifier(
                    n_estimators=500,  # Increased for better convergence
                    learning_rate=0.03,  # Reduced for better generalization
                    num_leaves=63,  # Reduced to prevent overfitting
                    max_depth=8,  # Adjusted for balance
                    min_child_samples=20,
                    min_split_gain=1e-5,
                    subsample=0.85,
                    feature_fraction=0.85,
                    scale_pos_weight=6,  # Adjusted for better balance
                    force_row_wise=True,
                    random_state=42,
                    boosting_type='gbdt',
                    data_sample_strategy='goss',
                    verbose=-1,
                    n_jobs=-1
                ),
                'xgb': xgb.XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=8,
                    min_child_weight=20,
                    gamma=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=6,
                    tree_method='hist',
                    grow_policy='lossguide',
                    random_state=42,
                    n_jobs=-1
                ),
                'cat': CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.03,
                    depth=8,
                    min_data_in_leaf=20,
                    l2_leaf_reg=3.0,
                    bootstrap_type='Bernoulli',
                    subsample=0.85,
                    class_weights=[1, 6],
                    random_seed=42,
                    verbose=False,
                    thread_count=-1
                )
            }
        else:  # Confirmation stage
            models = {
                'lgb': lgb.LGBMClassifier(
                    n_estimators=500,  # Increased for better convergence
                    learning_rate=0.03,  # Reduced for better generalization
                    num_leaves=63,  # Reduced to prevent overfitting
                    max_depth=8,  # Adjusted for balance
                    min_child_samples=20,
                    min_split_gain=1e-5,
                    subsample=0.85,
                    feature_fraction=0.85,
                    scale_pos_weight=6,  # Adjusted for better balance
                    force_row_wise=True,
                    random_state=42,
                    boosting_type='gbdt',
                    data_sample_strategy='goss',
                    verbose=-1,
                    n_jobs=-1
                ),
                'xgb': xgb.XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=8,
                    min_child_weight=20,
                    gamma=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=6,
                    tree_method='hist',
                    grow_policy='lossguide',
                    random_state=42,
                    n_jobs=-1
                ),
                'cat': CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.03,
                    depth=8,
                    min_data_in_leaf=20,
                    l2_leaf_reg=3.0,
                    bootstrap_type='Bernoulli',
                    subsample=0.85,
                    class_weights=[1, 6],
                    random_seed=42,
                    verbose=False,
                    thread_count=-1
                )
            }
        return models
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], 
            test_mode: bool = False) -> 'DiabetesEnsemblePredictor':
        """Fit the ensemble model to the training data."""
        if not self.is_fitted:
            # Convert inputs to numpy arrays if needed
            X = np.array(X) if isinstance(X, pd.DataFrame) else X
            y = np.array(y) if isinstance(y, pd.Series) else y
            
            print(f"Input shapes - X: {X.shape}, y: {y.shape}")
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            print(f"After split - X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"Class distribution before ADASYN - 0: {sum(y_train == 0)}, 1: {sum(y_train == 1)}")
            
            # Use preprocessor's ADASYN for better class balancing
            from preprocessor import PreProcessor
            from imblearn.over_sampling import RandomOverSampler
            preprocessor = PreProcessor()
            
            # Initialize performance tracking for dynamic weights
            self.model_performance = {'screening': {}, 'confirmation': {}}
            self.range_performance = {'screening': {}, 'confirmation': {}}
            
            # Define prediction ranges for range-specific weights
            self.pred_ranges = [
                (0.0, 0.2),   # Very low risk
                (0.2, 0.4),   # Low risk
                (0.4, 0.6),   # Medium risk
                (0.6, 0.8),   # High risk
                (0.8, 1.0)    # Very high risk
            ]
            
            # Train screening models
            print("\nTraining screening models...")
            try:
                if test_mode:
                    # Use simple random oversampling for test mode
                    oversampler = RandomOverSampler(sampling_strategy=1.0, random_state=42)
                    X_train_screen, y_train_screen = oversampler.fit_resample(X_train, y_train)
                    print("Using RandomOverSampler for test mode")
                else:
                    # Use ADASYN for production mode
                    X_train_screen, y_train_screen = preprocessor.adasyn.fit_resample(X_train, y_train)
                print(f"After resampling - X_screen: {X_train_screen.shape}, y_screen: {y_train_screen.shape}")
                print(f"Class distribution after resampling - 0: {sum(y_train_screen == 0)}, 1: {sum(y_train_screen == 1)}")
            except Exception as e:
                print(f"Resampling error: {str(e)}")
                raise e
            
            self._train_stage_models('screening', X_train_screen, y_train_screen, X_val, y_val, test_mode)
            
            print("\nTraining confirmation models...")
            try:
                if test_mode:
                    # Use simple random oversampling for test mode
                    X_train_confirm, y_train_confirm = oversampler.fit_resample(X_train, y_train)
                    print("Using RandomOverSampler for test mode")
                else:
                    # Use ADASYN for production mode
                    X_train_confirm, y_train_confirm = preprocessor.adasyn.fit_resample(X_train, y_train)
                print(f"After resampling - X: {X_train_confirm.shape}, y: {y_train_confirm.shape}")
                print(f"Class distribution after resampling - 0: {sum(y_train_confirm == 0)}, 1: {sum(y_train_confirm == 1)}")
            except Exception as e:
                print(f"Resampling error: {str(e)}")
                raise e
            
            self._train_stage_models('confirmation', X_train_confirm, y_train_confirm, X_val, y_val, test_mode)
            
            self.is_fitted = True
        return self
    
    def _train_stage_models(self, stage: str, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray, test_mode: bool = False) -> None:
        """Train models for a specific stage and compute their performance metrics."""
        
        for name, model in self._create_base_models(stage, test_mode).items():
            print(f"\nTraining {name} model...")
            model.fit(X_train, y_train)
            self.models[stage][name] = model
            
            # Get base model performance
            val_probs = model.predict_proba(X_val)
            self.model_performance[stage][name] = roc_auc_score(y_val, val_probs[:, 1])
            
            # Calculate range-specific performance
            self.range_performance[stage][name] = {}
            for range_min, range_max in self.pred_ranges:
                range_mask = (val_probs[:, 1] >= range_min) & (val_probs[:, 1] < range_max)
                if np.any(range_mask):
                    range_score = roc_auc_score(
                        y_val[range_mask], 
                        val_probs[range_mask, 1]
                    ) if len(np.unique(y_val[range_mask])) > 1 else 0.5
                    self.range_performance[stage][name][(range_min, range_max)] = range_score
            
            # Train calibrated models for uncertainty
            calibrated_models = []
            for i in range(self.n_calibration_models):
                print(f"Training calibrated model {i+1}/{self.n_calibration_models}")
                calibrated = CalibratedClassifierCV(
                    estimator=model,
                    cv='prefit',
                    n_jobs=-1
                )
                calibrated.fit(X_val, y_val)
                calibrated_models.append(calibrated)
            self.calibrated_models[stage][name] = calibrated_models
    
    def _get_model_weights(self, stage: str, probas: np.ndarray) -> np.ndarray:
        """Calculate dynamic weights based on model performance and prediction ranges."""
        base_weights = np.array([
            self.model_performance[stage][name] for name in ['lgb', 'xgb', 'cat']
        ])
        
        # Normalize base weights
        base_weights = base_weights / np.sum(base_weights)
        
        # Get range-specific weights
        range_weights = np.zeros_like(base_weights)
        for i, name in enumerate(['lgb', 'xgb', 'cat']):
            model_range_scores = []
            for prob in probas[i]:
                for (range_min, range_max), score in self.range_performance[stage][name].items():
                    if range_min <= prob[1] < range_max:
                        model_range_scores.append(score)
                        break
            range_weights[i] = np.mean(model_range_scores) if model_range_scores else base_weights[i]
        
        # Normalize range weights
        range_weights = range_weights / np.sum(range_weights)
        
        # Combine base and range weights
        final_weights = (0.7 * base_weights + 0.3 * range_weights)
        return final_weights / np.sum(final_weights)
    
    def _predict_proba_stage(self, X: np.ndarray, stage: str) -> np.ndarray:
        """Get weighted ensemble predictions for a stage with improved uncertainty handling."""
        predictions = []
        uncertainties = []
        
        # Get predictions from each model
        for name in ['lgb', 'xgb', 'cat']:
            model_preds = np.mean([
                model.predict_proba(X) for model in self.calibrated_models[stage][name]
            ], axis=0)
            predictions.append(model_preds)
            
            # Calculate per-model uncertainty
            model_std = np.std([
                model.predict_proba(X)[:, 1] for model in self.calibrated_models[stage][name]
            ], axis=0)
            uncertainties.append(model_std)
        
        predictions = np.stack(predictions)  # Shape: (n_models, n_samples, 2)
        uncertainties = np.stack(uncertainties)  # Shape: (n_models, n_samples)
        
        # Calculate weighted average predictions
        weights = self._get_model_weights(stage, predictions)
        weights = weights[:, np.newaxis, np.newaxis]  # Shape: (n_models, 1, 1)
        weighted_preds = np.sum(predictions * weights, axis=0)
        
        # Handle high uncertainty cases
        high_uncertainty = np.mean(uncertainties, axis=0) > 0.2
        if np.any(high_uncertainty):
            # For high uncertainty cases:
            # 1. Use majority voting for the prediction
            votes = predictions[:, high_uncertainty, 1] > 0.5
            vote_preds = np.mean(votes, axis=0)
            
            # 2. Adjust confidence based on vote agreement
            vote_agreement = np.abs(vote_preds - 0.5) * 2  # Scale to [0, 1]
            confidence_adjustment = 0.5 + (vote_agreement * 0.3)  # Max adjustment of 0.3
            
            # Apply adjusted predictions
            weighted_preds[high_uncertainty, 1] = vote_preds
            weighted_preds[high_uncertainty, 0] = 1 - vote_preds
            
            # Scale probabilities based on confidence
            weighted_preds[high_uncertainty] *= confidence_adjustment[:, np.newaxis]
            # Ensure probabilities sum to 1
            weighted_preds[high_uncertainty] /= weighted_preds[high_uncertainty].sum(axis=1, keepdims=True)
        
        return weighted_preds
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, Dict]:
        """Make predictions with uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Convert input to numpy array
            X = np.array(X) if isinstance(X, pd.DataFrame) else X
            
            if X.ndim != 2:
                raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
            
            # Get screening predictions
            screen_probs = self._predict_proba_stage(X, 'screening')
            if screen_probs.ndim != 2 or screen_probs.shape[1] != 2:
                raise ValueError(f"Invalid screening probabilities shape: {screen_probs.shape}")
            
            screen_mask = screen_probs[:, 1] >= self.screening_threshold
            
            # Initialize confirmation probabilities with screening probabilities
            confirm_probs = np.copy(screen_probs)
            
            # Get confirmation predictions only for screened samples
            if np.any(screen_mask):
                X_confirm = X[screen_mask]
                conf_preds = self._predict_proba_stage(X_confirm, 'confirmation')
                
                if conf_preds.shape[1] != 2:
                    raise ValueError(f"Invalid confirmation probabilities shape: {conf_preds.shape}")
                    
                confirm_probs[screen_mask] = conf_preds
            
            # Combine predictions based on screening results
            final_probs = np.where(screen_mask[:, np.newaxis], confirm_probs, screen_probs)
            
            # Validate final probabilities
            if np.any(np.isnan(final_probs)):
                print("Warning: NaN values found in probabilities, replacing with 0.5")
                final_probs = np.nan_to_num(final_probs, nan=0.5)
            
            if not np.allclose(final_probs.sum(axis=1), 1.0, rtol=1e-5):
                print("Warning: Probabilities don't sum to 1, normalizing")
                final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
            
            # Calculate uncertainties
            epistemic, aleatoric, total = self._calculate_uncertainties(X)
            
            uncertainties = {
                'epistemic': epistemic,
                'aleatoric': aleatoric,
                'total': total,
                'screening_confidence': 1 - 2 * np.abs(screen_probs[:, 1] - 0.5),
                'confirmation_confidence': 1 - 2 * np.abs(confirm_probs[:, 1] - 0.5)
            }
            
            return final_probs, uncertainties
            
        except Exception as e:
            print(f"\nError in predict_proba: {str(e)}")
            print(f"Input shape: {X.shape}")
            if 'screen_probs' in locals():
                print(f"Screening probabilities shape: {screen_probs.shape}")
            if 'conf_preds' in locals():
                print(f"Confirmation probabilities shape: {conf_preds.shape}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        try:
            # Get probabilities and handle potential errors
            probas, uncertainties = self.predict_proba(X)
            
            # Get predictions using confirmation threshold for better precision
            predictions = (probas[:, 1] >= self.confirmation_threshold).astype(int)
            
            # Add warning for highly uncertain predictions
            high_uncertainty_mask = uncertainties['total'] > 0.2
            if np.any(high_uncertainty_mask):
                n_uncertain = np.sum(high_uncertainty_mask)
                print(f"\nWarning: {n_uncertain} predictions have high uncertainty (>0.2)")
                print("Consider these predictions with caution.")
            
            return predictions
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(f"Input shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")
            # Return safe default predictions
            return np.zeros(len(X) if hasattr(X, '__len__') else 1, dtype=int)
    
    def get_feature_importance(self, model_name, stage):
        """Get feature importance for a specific model and stage."""
        try:
            if not self.is_fitted:
                print("Warning: Model not fitted yet, returning zero array")
                return np.zeros(len(self.models[stage][model_name].feature_importances_))
                
            if stage not in self.models or model_name not in self.models[stage]:
                print(f"Warning: Model {model_name} for stage {stage} not found")
                # Try to infer feature count from another model
                for s in self.models:
                    for m in self.models[s]:
                        if hasattr(self.models[s][m], 'feature_importances_'):
                            return np.zeros(len(self.models[s][m].feature_importances_))
                return np.zeros(20)  # Default fallback
            
            model = self.models[stage][model_name]
            
            # Handle different model types
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'get_score'):
                scores = model.get_score(importance_type='gain')
                # Convert dictionary to array maintaining order
                importances = np.zeros(model.n_features_)
                for feat, score in scores.items():
                    idx = int(feat.replace('f', ''))
                    importances[idx] = score
                return importances
            else:
                print(f"Warning: Model {model_name} does not support feature importance")
                return np.zeros(model.n_features_ if hasattr(model, 'n_features_') else 20)
                
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            # Try to return an appropriate size array
            if hasattr(model, 'n_features_'):
                return np.zeros(model.n_features_)
            elif hasattr(model, 'feature_importances_'):
                return np.zeros(len(model.feature_importances_))
            return np.zeros(20)  # Default fallback

    def _calculate_uncertainties(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate multiple uncertainty metrics efficiently."""
        try:
            all_predictions = []
            stage_predictions = {stage: {name: [] for name in ['lgb', 'xgb', 'cat']} 
                               for stage in ['screening', 'confirmation']}
            
            # First collect and cache predictions from all models
            for stage in ['screening', 'confirmation']:
                for name in ['lgb', 'xgb', 'cat']:
                    try:
                        preds = np.array([
                            model.predict_proba(X) for model in self.calibrated_models[stage][name]
                        ])
                        if preds.ndim != 3:  # Should be (n_models, n_samples, 2)
                            raise ValueError(f"Invalid predictions shape from {name}: {preds.shape}")
                        stage_predictions[stage][name] = preds
                    except Exception as e:
                        print(f"Error getting predictions for {stage} {name}: {str(e)}")
                        # Return zero arrays if prediction fails
                        return (np.zeros(X.shape[0]), np.zeros(X.shape[0]), np.zeros(X.shape[0]))
            
            # Now use MC iterations to combine predictions
            for _ in range(self.mc_iterations):
                stage_preds = []
                for stage in ['screening', 'confirmation']:
                    model_preds = []
                    for name in ['lgb', 'xgb', 'cat']:
                        # Randomly sample from stored predictions
                        idx = np.random.randint(len(stage_predictions[stage][name]))
                        model_preds.append(stage_predictions[stage][name][idx])
                    # Average predictions from different models
                    model_preds = np.stack(model_preds)  # Shape: (n_models, n_samples, 2)
                    stage_preds.append(np.mean(model_preds, axis=0))  # Shape: (n_samples, 2)
                # Average predictions from different stages
                stage_preds = np.stack(stage_preds)  # Shape: (n_stages, n_samples, 2)
                all_predictions.append(np.mean(stage_preds, axis=0))  # Shape: (n_samples, 2)
            
            predictions = np.stack(all_predictions)  # Shape: (mc_iterations, n_samples, 2)
            
            # Epistemic uncertainty (model uncertainty)
            epistemic = np.std(predictions, axis=0)  # Shape: (n_samples, 2)
            
            # Aleatoric uncertainty (data uncertainty)
            mean_p = np.mean(predictions, axis=0)  # Shape: (n_samples, 2)
            aleatoric = np.sqrt(np.mean(predictions * (1 - predictions), axis=0))  # Shape: (n_samples, 2)
            
            # Total uncertainty (combine epistemic and aleatoric)
            total = np.sqrt(epistemic**2 + aleatoric**2)  # Shape: (n_samples, 2)
            
            # Return only the uncertainties for positive class (index 1)
            return epistemic[:, 1], aleatoric[:, 1], total[:, 1]
            
        except Exception as e:
            print(f"Error in uncertainty calculation: {str(e)}")
            print(f"Input shape: {X.shape}")
            if 'predictions' in locals():
                print(f"MC predictions shape: {predictions.shape}")
            # Return zero arrays if calculation fails
            return np.zeros(X.shape[0]), np.zeros(X.shape[0]), np.zeros(X.shape[0])
