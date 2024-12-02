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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class DiabetesEnsemblePredictor:
    """Two-stage ensemble model for diabetes prediction with uncertainty estimation"""
    
    def __init__(self, 
                 screening_threshold: float = 0.15,  # Lowered to more stable region
                 confirmation_threshold: float = 0.25,  # Further lowered based on test results
                 n_calibration_models: int = 15,  # Increased for better uncertainty estimation
                 mc_iterations: int = 20):  # Increased for more stable MC dropout
        """Initialize the ensemble predictor.
        
        Args:
            screening_threshold: Threshold for screening stage (default: 0.15)
                Lowered to more stable region
            confirmation_threshold: Threshold for confirmation stage (default: 0.25)
                Further lowered based on test results
            n_calibration_models: Number of calibrated models for uncertainty (default: 15)
                Increased for better uncertainty estimation
            mc_iterations: Number of Monte Carlo iterations for uncertainty (default: 20)
                Increased for more stable MC dropout
        """
        self.screening_threshold = screening_threshold
        self.confirmation_threshold = confirmation_threshold
        self.n_calibration_models = n_calibration_models
        self.mc_iterations = mc_iterations
        
        # Initialize model containers
        self.models = {}
        self.calibrated_models = {}
        self.model_weights = {}
        
        for stage in ['screening', 'confirmation']:
            self.models[stage] = {}
            self.calibrated_models[stage] = {}
            self.model_weights[stage] = {}
        
        self.is_fitted = False
    
    def _create_base_models(self, stage: str) -> Dict:
        """Create base models for the ensemble."""
        if stage == 'screening':
            models = {
                'lgb': lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    num_leaves=127,
                    max_depth=8,
                    min_child_samples=100,
                    min_split_gain=1e-2,
                    subsample=0.8,
                    feature_fraction=0.8,
                    class_weight='balanced',
                    scale_pos_weight=3,  # Balanced weight for stability
                    force_row_wise=True,
                    random_state=42,
                    boosting_type='gbdt',
                    data_sample_strategy='goss',
                    verbose=-1,
                    n_jobs=-1
                ),
                'xgb': xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=8,
                    min_child_weight=100,
                    gamma=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=6,  # Moderate weight
                    tree_method='hist',
                    grow_policy='lossguide',
                    random_state=42,
                    n_jobs=-1
                ),
                'cat': CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.1,
                    depth=8,
                    min_data_in_leaf=100,
                    l2_leaf_reg=3,
                    bootstrap_type='Bernoulli',
                    subsample=0.8,
                    class_weights=[1, 6],  # Moderate weight
                    random_seed=42,
                    verbose=False,
                    thread_count=-1
                )
            }
        else:  # confirmation stage
            models = {
                'lgb': lgb.LGBMClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    num_leaves=63,
                    max_depth=6,
                    min_child_samples=50,
                    subsample=0.8,
                    feature_fraction=0.8,
                    class_weight='balanced',
                    scale_pos_weight=4,  # Increased weight
                    reg_alpha=0.1,  # L1 regularization for stability
                    reg_lambda=0.1,  # L2 regularization for stability
                    force_row_wise=True,
                    random_state=42,
                    boosting_type='gbdt',
                    data_sample_strategy='goss',
                    verbose=-1,
                    n_jobs=-1
                ),
                'xgb': xgb.XGBClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=6,
                    min_child_weight=50,
                    gamma=0.2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=6,  # Increased weight
                    reg_alpha=0.1,  # Added regularization
                    reg_lambda=0.1,
                    tree_method='hist',
                    grow_policy='lossguide',
                    random_state=42,
                    n_jobs=-1
                ),
                'cat': CatBoostClassifier(
                    iterations=150,
                    learning_rate=0.1,
                    depth=6,
                    min_data_in_leaf=50,
                    l2_leaf_reg=5,
                    bootstrap_type='Bernoulli',
                    subsample=0.8,
                    class_weights=[1, 6],  # Increased weight
                    random_seed=42,
                    verbose=False,
                    thread_count=-1
                )
            }
        return models
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'DiabetesEnsemblePredictor':
        """Train the two-stage ensemble model with SMOTE sampling."""
        
        # Convert inputs to numpy if needed
        X = np.array(X) if isinstance(X, pd.DataFrame) else X
        y = np.array(y) if isinstance(y, pd.Series) else y
        
        print(f"Input shapes - X: {X.shape}, y: {y.shape}")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        print(f"After split - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Class distribution before SMOTE - 0: {sum(y_train == 0)}, 1: {sum(y_train == 1)}")
        
        # Train screening models with SMOTE
        print("\nTraining screening models...")
        smote = SMOTE(random_state=42, sampling_strategy=0.6)  # Increased for better recall
        try:
            X_train_screen, y_train_screen = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - X_screen: {X_train_screen.shape}, y_screen: {y_train_screen.shape}")
            print(f"Class distribution after SMOTE - 0: {sum(y_train_screen == 0)}, 1: {sum(y_train_screen == 1)}")
        except Exception as e:
            print(f"SMOTE error: {str(e)}")
            print(f"X_train dtype: {X_train.dtype}, y_train dtype: {y_train.dtype}")
            raise e
        
        # Train screening models
        for name, model in self._create_base_models('screening').items():
            print(f"\nTraining {name} model...")
            # Train base model
            model.fit(X_train_screen, y_train_screen)
            self.models['screening'][name] = model
            
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
            self.calibrated_models['screening'][name] = calibrated_models
        
        # Get predictions from screening stage
        screen_probs = self._predict_proba_stage(X_train, 'screening')
        confirmed_idx = screen_probs[:, 1] >= self.screening_threshold
        
        print("\nTraining confirmation models...")
        print(f"Samples passing screening: {sum(confirmed_idx)}")
        
        # Train confirmation models on screened data with SMOTE
        if np.sum(confirmed_idx) > 0:
            X_confirm = X_train[confirmed_idx]
            y_confirm = y_train[confirmed_idx]
            
            print(f"Confirmation data - X: {X_confirm.shape}, y: {y_confirm.shape}")
            print(f"Class distribution before SMOTE - 0: {sum(y_confirm == 0)}, 1: {sum(y_confirm == 1)}")
            
            # Apply SMOTE with more aggressive sampling for confirmation
            smote_confirm = SMOTE(random_state=42, sampling_strategy=0.7)
            try:
                X_confirm_res, y_confirm_res = smote_confirm.fit_resample(X_confirm, y_confirm)
                print(f"After SMOTE - X: {X_confirm_res.shape}, y: {y_confirm_res.shape}")
                print(f"Class distribution after SMOTE - 0: {sum(y_confirm_res == 0)}, 1: {sum(y_confirm_res == 1)}")
            except Exception as e:
                print(f"SMOTE error: {str(e)}")
                print(f"X_confirm dtype: {X_confirm.dtype}, y_confirm dtype: {y_confirm.dtype}")
                raise e
            
            for name, model in self._create_base_models('confirmation').items():
                print(f"\nTraining {name} confirmation model...")
                # Train base model
                model.fit(X_confirm_res, y_confirm_res)
                self.models['confirmation'][name] = model
                
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
                self.calibrated_models['confirmation'][name] = calibrated_models
        
        self.is_fitted = True
        return self
    
    def _predict_proba_stage(self, X: np.ndarray, stage: str) -> np.ndarray:
        """Get weighted ensemble predictions for a stage."""
        predictions = []
        weights = []
        
        for name in ['lgb', 'xgb', 'cat']:
            model_preds = np.mean([
                model.predict_proba(X) for model in self.calibrated_models[stage][name]
            ], axis=0)
            predictions.append(model_preds)
            weights.append(1/3)  # Equal weights for simplicity
        
        # Stack predictions and calculate weighted average
        predictions = np.stack(predictions)
        weights = np.array(weights)[:, np.newaxis, np.newaxis]
        return np.sum(predictions * weights, axis=0)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, Dict]:
        """Make predictions with uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X) if isinstance(X, pd.DataFrame) else X
        
        # Get screening predictions
        screen_probs = self._predict_proba_stage(X, 'screening')
        screen_mask = screen_probs[:, 1] >= self.screening_threshold
        
        # Initialize confirmation probabilities
        confirm_probs = np.zeros_like(screen_probs)
        confirm_probs[:] = screen_probs  # Default to screening probs
        
        # Get confirmation predictions only for screened samples
        if np.any(screen_mask):
            conf_preds = self._predict_proba_stage(X[screen_mask], 'confirmation')
            confirm_probs[screen_mask] = conf_preds
        
        # Use confirmation probabilities for screened samples
        final_probs = np.where(screen_mask[:, np.newaxis], confirm_probs, screen_probs)
        
        # Calculate detailed uncertainties
        epistemic, aleatoric, total = self._calculate_uncertainties(X)
        
        uncertainties = {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total,
            'screening_confidence': 1 - 2 * np.abs(screen_probs[:, 1] - 0.5),
            'confirmation_confidence': 1 - 2 * np.abs(confirm_probs[:, 1] - 0.5)
        }
        
        return final_probs, uncertainties
    
    def _calculate_uncertainties(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate multiple uncertainty metrics.
        
        Returns:
            Tuple containing:
            - Epistemic uncertainty (model uncertainty)
            - Aleatoric uncertainty (data uncertainty)
            - Total uncertainty (sum of epistemic and aleatoric)
        """
        all_predictions = []
        
        # Collect predictions from all models and MC iterations
        for _ in range(self.mc_iterations):
            stage_preds = []
            for stage in ['screening', 'confirmation']:
                for name in ['lgb', 'xgb', 'cat']:
                    preds = np.mean([
                        model.predict_proba(X) for model in self.calibrated_models[stage][name]
                    ], axis=0)
                    stage_preds.append(preds)
            all_predictions.append(np.mean(stage_preds, axis=0))
        
        predictions = np.array(all_predictions)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = np.std(predictions, axis=0)
        
        # Aleatoric uncertainty (data uncertainty)
        mean_p = np.mean(predictions, axis=0)
        aleatoric = np.mean(predictions * (1 - predictions), axis=0)
        
        # Total uncertainty (sum of epistemic and aleatoric)
        total = epistemic + aleatoric
        
        return epistemic[:, 1], aleatoric[:, 1], total[:, 1]
