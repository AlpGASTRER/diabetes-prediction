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
import warnings
warnings.filterwarnings('ignore')

class DiabetesEnsemblePredictor:
    """Two-stage ensemble model for diabetes prediction with uncertainty estimation"""
    
    def __init__(self, 
                 screening_threshold: float = 0.2,  # Lowered from 0.3 for better recall
                 confirmation_threshold: float = 0.6,  # Lowered from 0.7 for better balance
                 n_calibration_models: int = 5,  # Increased for better uncertainty
                 mc_iterations: int = 10):  # MC Dropout iterations
        """Initialize the ensemble predictor.
        
        Args:
            screening_threshold: Threshold for screening stage (default: 0.2)
                Lower threshold to catch more potential cases
            confirmation_threshold: Threshold for confirmation stage (default: 0.6)
                Balanced threshold considering both precision and recall
            n_calibration_models: Number of calibrated models for uncertainty (default: 5)
            mc_iterations: Number of Monte Carlo iterations for uncertainty (default: 10)
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
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'DiabetesEnsemblePredictor':
        """Train the two-stage ensemble model."""
        X = X.values if hasattr(X, 'values') else X
        y = y.values if hasattr(y, 'values') else y
        
        # Split data for two stages
        X_screen, X_confirm, y_screen, y_confirm = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train both stages
        for stage, (X_stage, y_stage) in zip(
            ['screening', 'confirmation'],
            [(X_screen, y_screen), (X_confirm, y_confirm)]
        ):
            print(f"\nTraining {stage} models...")
            self._train_stage(stage, X_stage, y_stage)
        
        self.is_fitted = True
        return self
    
    def predict_proba_with_uncertainty(self, 
                                     X: Union[np.ndarray, pd.DataFrame]
                                     ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get predictions with detailed uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = X.values if hasattr(X, 'values') else X
        
        # Get screening predictions
        screen_probs = self._get_stage_predictions(X, 'screening')
        screen_mask = screen_probs >= self.screening_threshold
        
        # Get confirmation predictions only for screened samples
        confirm_probs = np.zeros_like(screen_probs)
        if np.any(screen_mask):
            confirm_probs[screen_mask] = self._get_stage_predictions(
                X[screen_mask], 'confirmation'
            )
        
        # Combine predictions
        final_probs = np.where(screen_mask, confirm_probs, screen_probs)
        
        # Calculate detailed uncertainties
        epistemic, aleatoric, total = self._calculate_uncertainties(X)
        
        uncertainties = {
            'epistemic': epistemic,  # Model uncertainty
            'aleatoric': aleatoric,  # Data uncertainty
            'total': total,         # Total predictive uncertainty
            'screening_confidence': np.abs(screen_probs - 0.5) * 2,  # Confidence in screening
            'confirmation_confidence': np.abs(confirm_probs - 0.5) * 2  # Confidence in confirmation
        }
        
        return final_probs, uncertainties
    
    def _train_stage(self, stage: str, X: np.ndarray, y: np.ndarray) -> None:
        """Train all models for a specific stage."""
        # Base parameters for each model type
        base_params = {
            'screening': {
                'scale_pos_weight': 1.0,  # Balanced weight for fair treatment
                'early_stopping_rounds': 10,
                'eval_metric': ['auc', 'aucpr']  # Added PR-AUC for imbalanced data
            },
            'confirmation': {
                'scale_pos_weight': 1.0,  # Balanced weight for fair treatment
                'early_stopping_rounds': 10,
                'eval_metric': ['auc', 'aucpr']  # Added PR-AUC for imbalanced data
            }
        }
        
        # Create validation set for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Calculate class weights based on data distribution
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        weight_ratio = neg_count / pos_count if pos_count > 0 else 1.0
        
        # Update weights based on actual class distribution
        base_params[stage]['scale_pos_weight'] = weight_ratio
        
        # Train base models with early stopping
        self.models[stage] = {
            'xgb': self._train_xgboost(X_train, y_train, X_val, y_val, stage, base_params[stage]),
            'lgb': self._train_lightgbm(X_train, y_train, X_val, y_val, stage, base_params[stage]),
            'cat': self._train_catboost(X_train, y_train, X_val, y_val, stage, base_params[stage])
        }
        
        # Train calibrated models
        self._calibrate_models(stage, X, y)
        
        # Update model weights
        self._update_model_weights(stage, X_val, y_val)
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, stage, params):
        """Train XGBoost model with early stopping."""
        model = xgb.XGBClassifier(**params)
        
        eval_set = [(X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        return model
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, stage, params):
        """Train LightGBM with early stopping."""
        model = lgb.LGBMClassifier(
            num_leaves=31 if stage == 'confirmation' else 15,
            learning_rate=0.05,
            n_estimators=1000,
            objective='binary',
            random_state=42,
            scale_pos_weight=params['scale_pos_weight']
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=params['early_stopping_rounds'])]
        )
        return model
    
    def _train_catboost(self, X_train, y_train, X_val, y_val, stage, params):
        """Train CatBoost with early stopping."""
        model = CatBoostClassifier(
            depth=6 if stage == 'confirmation' else 4,
            learning_rate=0.05,
            iterations=1000,
            random_seed=42,
            verbose=False,
            eval_metric='AUC',
            scale_pos_weight=params['scale_pos_weight']
        )
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=params['early_stopping_rounds'],
            verbose=False
        )
        return model
    
    def _calibrate_models(self, stage: str, X: np.ndarray, y: np.ndarray) -> None:
        """Calibrate models using stratified k-fold."""
        print(f"Calibrating {stage} models...")
        self.calibrated_models[stage] = {name: [] for name in ['xgb', 'lgb', 'cat']}
        
        # Use stratified splits for calibration
        for i in range(self.n_calibration_models):
            X_cal, X_val, y_cal, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42+i, stratify=y
            )
            
            for name, model in self.models[stage].items():
                calibrated = CalibratedClassifierCV(
                    model, cv='prefit', method='sigmoid'
                )
                calibrated.fit(X_cal, y_cal)
                self.calibrated_models[stage][name].append(calibrated)
    
    def _update_model_weights(self, stage: str, X: np.ndarray, y: np.ndarray) -> None:
        """Update model weights using validation performance."""
        metrics = {}
        threshold = self.screening_threshold if stage == 'screening' else self.confirmation_threshold
        
        for name in ['xgb', 'lgb', 'cat']:
            preds = self._get_model_predictions(X, stage, name)
            metrics[name] = (
                recall_score(y, preds > threshold)
                if stage == 'screening'
                else precision_score(y, preds > threshold, zero_division=1)  # Return 1.0 when no positive predictions
            )
        
        # Softmax normalization of weights
        exp_metrics = np.exp(list(metrics.values()))
        sum_exp = np.sum(exp_metrics)
        self.model_weights[stage] = {
            name: float(exp/sum_exp)
            for name, exp in zip(metrics.keys(), exp_metrics)
        }
    
    def _get_model_predictions(self, X: np.ndarray, stage: str, model_name: str) -> np.ndarray:
        """Get averaged predictions from calibrated models."""
        return np.mean([
            model.predict_proba(X)[:, 1]
            for model in self.calibrated_models[stage][model_name]
        ], axis=0)
    
    def _get_stage_predictions(self, X: np.ndarray, stage: str) -> np.ndarray:
        """Get weighted ensemble predictions for a stage."""
        predictions = []
        weights = []
        
        for name in ['xgb', 'lgb', 'cat']:
            predictions.append(self._get_model_predictions(X, stage, name))
            weights.append(self.model_weights[stage][name])
        
        return np.average(predictions, weights=weights, axis=0)
    
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
                for name in ['xgb', 'lgb', 'cat']:
                    preds = self._get_model_predictions(X, stage, name)
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
        
        return epistemic, aleatoric, total
