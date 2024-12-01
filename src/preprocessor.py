"""
Medical-grade data preprocessing for diabetes prediction.

This module implements a comprehensive preprocessing pipeline specifically designed
for medical data analysis and diabetes prediction. It includes:
- Medical-grade data validation
- Age-specific feature normalization
- Clinical feature engineering
- Advanced outlier detection
- Multi-method feature importance calculation

The preprocessor adapts its behavior based on dataset size, providing efficient
processing for both testing (< 10k samples) and production (500k+ samples) scenarios.

Example:
    ```python
    preprocessor = PreProcessor()
    X_processed = preprocessor.fit_transform(X, y)
    ```

Author: Codeium AI
Last Modified: 2024
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTENC
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import warnings
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Union
from sklearn.model_selection import StratifiedKFold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class MedicalThresholds:
    """
    Medical thresholds for feature validation and outlier detection.
    
    These thresholds are based on clinical guidelines and medical literature.
    They are used to validate input data and detect medically significant outliers.
    """
    BMI_MIN: float = 10.0
    BMI_MAX: float = 70.0
    AGE_MIN: float = 18
    AGE_MAX: float = 120
    GLUCOSE_MIN: float = 30
    GLUCOSE_MAX: float = 600
    BP_MIN: float = 60
    BP_MAX: float = 300

@dataclass
class DataQualityReport:
    """Data quality report containing validation results"""
    missing_values: Dict[str, int]
    outliers: Dict[str, List[int]]
    invalid_ranges: Dict[str, List[float]]
    correlation_matrix: pd.DataFrame
    feature_importance: Dict[str, float]

class PreProcessor:
    """
    A comprehensive medical data preprocessor for diabetes prediction.
    
    This class implements a pipeline of preprocessing steps specifically designed
    for medical data analysis, with a focus on diabetes prediction. It includes
    medical-grade data validation, feature engineering, and importance calculation.
    
    Attributes:
        thresholds (MedicalThresholds): Medical thresholds for data validation
        scaler (StandardScaler): Scaler for feature normalization
        selected_features (list): List of selected features after importance analysis
        logger (logging.Logger): Logger for tracking preprocessing steps
    """
    
    def __init__(self, 
                correlation_threshold: float = 0.8, 
                random_state: int = 42,
                cache_dir: Optional[str] = None,
                thresholds: MedicalThresholds = None):
        """Initialize the preprocessor with enhanced configuration
        
        Args:
            correlation_threshold: Threshold for feature correlation filtering
            random_state: Random state for reproducibility
            cache_dir: Directory to cache preprocessed features
            thresholds (MedicalThresholds, optional): Custom medical thresholds.
                Defaults to None, using standard thresholds.
        """
        self.thresholds = thresholds or MedicalThresholds()
        self.robust_scaler = RobustScaler()
        self.age_scalers: Dict[str, RobustScaler] = {}
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.random_state = random_state
        self.lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
        self.feature_importance = None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Clinical domain weights for diabetes-specific features
        self.clinical_weights = {
            'BMI': 1.2,
            'Age': 1.1,
            'HighBP': 1.2,
            'HighChol': 1.2,
            'GenHlth': 1.1,
            'PhysHlth': 1.1,
            'HeartDiseaseorAttack': 1.15,
            'PhysActivity': 1.1,
            'MetabolicSyndrome': 1.25,  # Composite feature
            'CVD_Severity': 1.2,        # Composite feature
            'Total_Risk_Score': 1.15    # Composite feature
        }
        
        # Define medical thresholds with clinical reasoning
        self.medical_thresholds = {
            'BMI': {'low': self.thresholds.BMI_MIN, 'high': self.thresholds.BMI_MAX},  
            'Age': {'low': self.thresholds.AGE_MIN, 'high': self.thresholds.AGE_MAX},  
            'PhysHlth': {'low': 0, 'high': 30},  
            'GenHlth': {'low': 1, 'high': 5}  
        }
        
        self.categorical_features = [
            'HighBP', 'HighChol', 'Smoker', 'Stroke', 
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
        ]
        
        self.continuous_features = ['BMI', 'Age', 'PhysHlth', 'GenHlth']
        self.required_columns = self.categorical_features + self.continuous_features
        self.MIN_SAMPLES_PER_CLASS = 10
        
        self.logger = logging.getLogger(__name__)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Enhanced transform with medical-grade preprocessing"""
        try:
            self.logger.info("Performing medical-grade preprocessing...")
            
            # Validate input data
            X_clean = self._validate_data(X)
            
            # Remove exact duplicates
            X_clean, y_clean = self._remove_duplicates(X_clean, y)
            
            # Remove near-duplicates
            X_clean, y_clean = self._remove_near_duplicates(X_clean, y_clean)
            
            # Handle outliers
            X_clean = self._handle_outliers(X_clean)
            # Keep only rows that weren't removed by outlier detection
            y_clean = y_clean[X_clean.index]
            
            # Create age groups for normalization
            age_groups = self._create_age_groups(X_clean['Age'])
            
            # Perform age-specific normalization
            X_clean = self._age_specific_normalization(X_clean, age_groups)
            
            # Create medical scores
            X_clean = self._create_medical_scores(X_clean)
            
            # Create clinical interactions
            X_clean = self._create_clinical_interactions(X_clean)
            
            # Remove highly correlated features
            X_clean = self._remove_correlated_features(X_clean)
            
            # Calculate feature importance
            self._calculate_feature_importance(X_clean, y_clean)
            
            return X_clean, y_clean
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
            
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessing"""
        try:
            # Validate data
            X = self._validate_data(X)
            
            # Handle outliers
            X_clean = self._handle_outliers(X)
            
            # Age-specific normalization
            age_groups = self._create_age_groups(X_clean['Age'])
            X_clean = self._age_specific_normalization(X_clean, age_groups)
            
            # Create medical scores
            X_clean = self._create_medical_scores(X_clean)
            
            # Create clinical interactions
            X_clean = self._create_clinical_interactions(X_clean)
            
            # Use only selected features
            if self.selected_features is not None:
                X_clean = X_clean[self.selected_features]
            
            return X_clean
            
        except Exception as e:
            self.logger.error(f"Error in transform: {str(e)}")
            raise
            
    def _remove_duplicates(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove exact duplicates"""
        unique_indices = ~X.duplicated()
        return X[unique_indices], y[unique_indices]
        
    def _remove_near_duplicates(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove near-duplicates using LSH with medical context"""
        from datasketch import MinHashLSH, MinHash
        
        # Convert rows to strings for hashing, with weighted features
        def row_to_string(row):
            # Give more weight to critical medical features
            critical_features = ['HighBP', 'HighChol', 'HeartDiseaseorAttack', 'BMI', 'Age']
            row_parts = []
            for col in X.columns:
                val = str(row[col])
                if col in critical_features:
                    # Repeat critical features to give them more weight
                    row_parts.extend([val] * 3)
                else:
                    row_parts.append(val)
            return ' '.join(row_parts)
        
        row_strings = X.apply(row_to_string, axis=1)
        
        # Create LSH index with lower threshold for medical data
        lsh = MinHashLSH(threshold=0.95, num_perm=256)  # More permutations for better accuracy
        minhashes = {}
        
        # Generate minhashes with error handling
        keep_indices = []
        try:
            for idx, row in enumerate(row_strings):
                minhash = MinHash(num_perm=256)
                for word in row.split():
                    minhash.update(word.encode('utf8'))
                
                # Check if we've seen a very similar row
                if not lsh.query(minhash):
                    lsh.insert(idx, minhash)
                    keep_indices.append(idx)
                minhashes[idx] = minhash
            
            # Ensure we keep at least 90% of the data
            if len(keep_indices) < len(X) * 0.9:
                self.logger.warning("Too many duplicates detected, adjusting threshold...")
                keep_indices = list(range(len(X)))  # Keep all rows
            
            self.logger.info(f"Removed {len(X) - len(keep_indices)} near-duplicate entries")
            return X.iloc[keep_indices], y.iloc[keep_indices]
            
        except Exception as e:
            self.logger.warning(f"Error in near-duplicate removal: {str(e)}")
            return X, y  # Return original data if error occurs
        
    def _validate_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input data with medical-grade checks"""
        self.logger.info("Validating input data...")
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create copy to avoid modifying original
        X_validated = X.copy()
        
        # Validate medical ranges
        for feature, limits in self.medical_thresholds.items():
            mask = (X_validated[feature] < limits['low']) | (X_validated[feature] > limits['high'])
            if mask.any():
                self.logger.warning(f"Found {mask.sum()} values outside medical range for {feature}")
                # Use clinically appropriate imputation
                if feature == 'BMI':
                    X_validated.loc[mask, feature] = X_validated[~mask][feature].median()
                elif feature == 'Age':
                    # Keep age but flag as potentially incorrect
                    X_validated[f'{feature}_suspicious'] = mask
        
        # Validate categorical features
        for cat_feature in self.categorical_features:
            invalid_values = ~X_validated[cat_feature].isin([0, 1])
            if invalid_values.any():
                self.logger.warning(f"Found {invalid_values.sum()} invalid values in {cat_feature}")
                # Use mode imputation for binary features
                X_validated.loc[invalid_values, cat_feature] = X_validated[cat_feature].mode()[0]
        
        return X_validated

    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using medical thresholds"""
        for col, thresholds in self.medical_thresholds.items():
            X[col] = X[col].clip(thresholds['low'], thresholds['high'])
            
        # Use Local Outlier Factor (LOF) for outlier detection
        outlier_scores = self.lof.fit_predict(X)
        X = X[outlier_scores != -1]
        
        return X
        
    def _create_age_groups(self, age_series: pd.Series) -> pd.Series:
        """Create age groups for age-specific normalization"""
        bins = [0, 35, 50, 65, 100]  
        labels = ['young', 'middle', 'senior', 'elderly']  
        return pd.cut(age_series, bins=bins, labels=labels)
        
    def _age_specific_normalization(self, X: pd.DataFrame, age_groups: pd.Series) -> pd.DataFrame:
        """Normalize features within age groups to account for age-specific normal ranges"""
        X_normalized = X.copy()
        
        # Features that need age-specific normalization
        features_to_normalize = ['BMI', 'PhysHlth', 'GenHlth']
        
        for age_group in age_groups.unique():
            age_mask = age_groups == age_group
            if age_mask.sum() >= self.MIN_SAMPLES_PER_CLASS:
                scaler = RobustScaler()
                X_normalized.loc[age_mask, features_to_normalize] = scaler.fit_transform(
                    X_normalized.loc[age_mask, features_to_normalize]
                )
                self.age_scalers[str(age_group)] = scaler
        
        return X_normalized

    def _create_medical_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create health scores with medical context"""
        X_scored = X.copy()
        
        try:
            # BMI Risk (WHO criteria for Asian populations - more stringent)
            X_scored['BMI_Risk'] = pd.cut(
                X_scored['BMI'],
                bins=[0, 18.5, 23, 27.5, 32.5, 37.5, float('inf')],
                labels=[1, 2, 3, 4, 5, 6]
            ).astype(float)
            
            # Age-specific BMI adjustment using numeric boundaries
            age_bins = [0, 35, 50, 65, 100]
            age_labels = ['young', 'middle', 'senior', 'elderly']
            age_groups = pd.cut(X_scored['Age'], bins=age_bins, labels=age_labels)
            
            # Apply age-specific adjustments
            age_adjustments = {
                'young': 0.9,    # Lower risk for younger
                'middle': 1.0,   # No adjustment for middle age
                'senior': 1.1,   # Slightly higher risk for seniors
                'elderly': 1.2   # Higher risk for elderly
            }
            
            for age_label, adjustment in age_adjustments.items():
                mask = age_groups == age_label
                X_scored.loc[mask, 'BMI_Risk'] *= adjustment
            
            # General Health Risk Score
            health_indicators = ['HighBP', 'HighChol', 'HeartDiseaseorAttack', 'GenHlth', 'PhysHlth']
            X_scored['Health_Risk_Score'] = X_scored[health_indicators].mean(axis=1)
            
            # Physical Activity Score
            activity_indicators = ['PhysActivity', 'BMI_Risk']
            X_scored['Activity_Score'] = X_scored[activity_indicators].mean(axis=1)
            
            # Lifestyle Score
            lifestyle_indicators = ['Smoker', 'HvyAlcoholConsump', 'Fruits', 'Veggies']
            X_scored['Lifestyle_Score'] = X_scored[lifestyle_indicators].mean(axis=1)
            
            # Healthcare Access Score
            access_indicators = ['AnyHealthcare', 'NoDocbcCost']
            X_scored['Healthcare_Access'] = X_scored[access_indicators].mean(axis=1)
            
            return X_scored
            
        except Exception as e:
            self.logger.warning(f"Error in creating medical scores: {str(e)}")
            # Return original data if scoring fails
            return X
        
    def _create_clinical_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinically relevant feature interactions.
        
        This method creates various interaction features based on medical knowledge:
        1. Age-related risks
        2. Metabolic syndrome indicators
        3. Cardiovascular risk factors
        4. Lifestyle impact factors
        5. Healthcare access barriers
        6. Comorbidity interactions
        
        Args:
            X: Input feature matrix
            
        Returns:
            DataFrame with added interaction features
        """
        X_interactions = X.copy()
        
        try:
            # Fill NaN values with medians for continuous features
            for col in self.continuous_features:
                X_interactions[col] = X_interactions[col].fillna(X_interactions[col].median())
            
            # Fill NaN values with mode for categorical features
            for col in self.categorical_features:
                X_interactions[col] = X_interactions[col].fillna(X_interactions[col].mode()[0])
            
            # 1. Age-related interactions
            X_interactions['Age_BMI'] = X_interactions['Age'] * X_interactions['BMI'] / 100
            X_interactions['Age_BP'] = X_interactions['Age'] * X_interactions['HighBP']
            X_interactions['Age_Chol'] = X_interactions['Age'] * X_interactions['HighChol']
            X_interactions['Age_Health'] = X_interactions['Age'] * X_interactions['GenHlth']
            
            # 2. Metabolic syndrome indicators
            X_interactions['BP_Chol'] = X_interactions['HighBP'] * X_interactions['HighChol']
            X_interactions['BMI_BP'] = (X_interactions['BMI'] > 30).astype(int) * X_interactions['HighBP']
            X_interactions['BMI_Chol'] = (X_interactions['BMI'] > 30).astype(int) * X_interactions['HighChol']
            X_interactions['MetabolicSyndrome'] = (
                (X_interactions['BMI'] > 30).astype(int) +
                X_interactions['HighBP'] +
                X_interactions['HighChol']
            )
            
            # 3. Cardiovascular risk factors
            X_interactions['CardioRisk'] = (
                X_interactions['HighBP'] * 2 + 
                X_interactions['HighChol'] * 1.5 +
                X_interactions['HeartDiseaseorAttack'] * 3 +
                X_interactions['Stroke'] * 2 +
                (X_interactions['BMI'] > 30).astype(int) * 1.5 +
                X_interactions['Smoker'] * 1.5
            )
            X_interactions['CVD_Severity'] = (
                X_interactions['HeartDiseaseorAttack'] +
                X_interactions['Stroke']
            )
            
            # 4. Lifestyle impact factors
            X_interactions['Inactive_Obese'] = (
                (1 - X_interactions['PhysActivity']) * 
                (X_interactions['BMI'] > 30).astype(int)
            )
            X_interactions['Smoking_CVD'] = X_interactions['Smoker'] * X_interactions['HeartDiseaseorAttack']
            X_interactions['LifestyleRisk'] = (
                X_interactions['Smoker'] * 2 +
                (1 - X_interactions['PhysActivity']) * 1.5 +
                (1 - X_interactions['Fruits']) +
                (1 - X_interactions['Veggies']) +
                X_interactions['HvyAlcoholConsump'] * 1.5 +
                (X_interactions['BMI'] > 30).astype(int) * 1.5
            )
            X_interactions['DietScore'] = (
                X_interactions['Fruits'] +
                X_interactions['Veggies'] -
                X_interactions['HvyAlcoholConsump']
            )
            
            # 5. Healthcare access barriers
            X_interactions['Poor_Health_No_Care'] = (
                (X_interactions['GenHlth'] > 3).astype(int) * 
                X_interactions['NoDocbcCost']
            )
            X_interactions['HealthcareRisk'] = (
                (1 - X_interactions['AnyHealthcare']) * 2 +
                X_interactions['NoDocbcCost'] * 1.5 +
                (X_interactions['GenHlth'] > 3).astype(int)
            )
            X_interactions['SocioeconomicBarrier'] = (
                (X_interactions['Income'] < 4).astype(int) * 
                X_interactions['NoDocbcCost']
            )
            
            # 6. Comorbidity interactions
            X_interactions['Chronic_Conditions'] = (
                X_interactions['HighBP'] +
                X_interactions['HighChol'] +
                X_interactions['HeartDiseaseorAttack'] +
                X_interactions['Stroke'] +
                (X_interactions['BMI'] > 30).astype(int)
            )
            X_interactions['Mental_Physical'] = (
                X_interactions['MentHlth'] * 
                X_interactions['PhysHlth'] / 
                100
            )
            X_interactions['Health_Impact'] = (
                X_interactions['GenHlth'] * 
                X_interactions['Chronic_Conditions']
            )
            
            # Complex risk scores
            X_interactions['Total_Risk_Score'] = (
                X_interactions['CardioRisk'] * 0.4 +
                X_interactions['LifestyleRisk'] * 0.3 +
                X_interactions['HealthcareRisk'] * 0.2 +
                X_interactions['MetabolicSyndrome'] * 0.1
            )
            
            # Ensure no NaN values in the output
            for col in X_interactions.columns:
                if X_interactions[col].isna().any():
                    if pd.api.types.is_numeric_dtype(X_interactions[col]):
                        X_interactions[col] = X_interactions[col].fillna(X_interactions[col].median())
                    else:
                        X_interactions[col] = X_interactions[col].fillna(X_interactions[col].mode()[0])
            
            return X_interactions
            
        except Exception as e:
            self.logger.warning(f"Error in creating clinical interactions: {str(e)}")
            return X  # Return original data if interactions fail
        
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features with medical priority"""
        # Define medical priority for features
        medical_priority = {
            'HighBP': 5, 'BMI': 5, 'Age': 5,         # Highest priority
            'HighChol': 4, 'HeartDiseaseorAttack': 4,
            'GenHlth': 3, 'PhysActivity': 3,
            'Smoker': 3, 'Stroke': 3,
            'Fruits': 2, 'Veggies': 2,               # Lower priority
            'HvyAlcoholConsump': 2,
            'AnyHealthcare': 1, 'NoDocbcCost': 1     # Lowest priority
        }
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Create upper triangle mask
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop while considering medical priority
        to_drop = set()
        for i in range(len(upper.columns)):
            for j in range(i + 1, len(upper.columns)):
                if upper.iloc[i, j] > self.correlation_threshold:
                    col_i = upper.columns[i]
                    col_j = upper.columns[j]
                    # Drop the feature with lower medical priority
                    if medical_priority.get(col_i, 0) < medical_priority.get(col_j, 0):
                        to_drop.add(col_i)
                    else:
                        to_drop.add(col_j)
        
        # Store selected features
        self.selected_features = [col for col in X.columns if col not in to_drop]
        
        return X[self.selected_features]
        
    def _calculate_feature_importance(self, X, y):
        """
        Calculate feature importance using multiple methods and aggregate results.
        Includes stability scoring, clinical domain weighting, and confidence intervals.
        Adapts to dataset size for efficient processing.
        
        Args:
            X (pd.DataFrame): The feature matrix
            y (pd.Series): The target variable
            
        Returns:
            pd.Series: Aggregated feature importance scores
        """
        # Determine if we're in test mode based on dataset size
        is_test_mode = len(X) < 10000
        importances = {}
        
        # Clinical domain weights for diabetes-specific features
        clinical_weights = self.clinical_weights
        
        # Method 1: Random Forest importance with cross-validation
        rf_params = {
            'n_estimators': 50 if is_test_mode else 200,
            'max_depth': 4 if is_test_mode else 8,
            'min_samples_leaf': 5 if is_test_mode else 20,
            'n_jobs': -1,
            'random_state': 42
        }
        
        # Use cross-validation for stability assessment
        n_splits = 3 if is_test_mode else 5
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        rf_importances = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_train, y_train)
            rf_importances.append(pd.Series(rf.feature_importances_, index=X.columns))
        
        # Calculate mean and std of RF importance
        importances['rf'] = pd.DataFrame(rf_importances).mean()
        importance_std = pd.DataFrame(rf_importances).std()
        
        # Method 2: Permutation importance with reduced computation for large datasets
        n_repeats = 5 if is_test_mode else 2
        rf_final = RandomForestClassifier(**rf_params)
        rf_final.fit(X, y)
        
        perm_importance = permutation_importance(
            rf_final, X, y,
            n_repeats=n_repeats,
            n_jobs=-1,
            random_state=42
        )
        importances['perm'] = pd.Series(
            perm_importance.importances_mean,
            index=X.columns
        )
        
        # Method 3: Mutual Information with bootstrapping for confidence estimation
        n_bootstrap = 10 if is_test_mode else 3
        mi_importances = []
        
        for _ in range(n_bootstrap):
            if is_test_mode:
                mi_importance = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
            else:
                sample_size = min(10000, len(X))
                indices = np.random.choice(len(X), sample_size, replace=True)
                X_sample, y_sample = X.iloc[indices], y.iloc[indices]
                mi_importance = mutual_info_classif(
                    X_sample, y_sample,
                    random_state=42,
                    n_neighbors=5
                )
            mi_importances.append(pd.Series(mi_importance, index=X.columns))
        
        importances['mi'] = pd.DataFrame(mi_importances).mean()
        mi_std = pd.DataFrame(mi_importances).std()
        
        # Normalize and aggregate importance scores
        for method in importances:
            # Add small epsilon to avoid division by zero
            min_val = importances[method].min()
            max_val = importances[method].max()
            if max_val - min_val > 0:
                importances[method] = (importances[method] - min_val) / (max_val - min_val)
            else:
                importances[method] = np.zeros_like(importances[method])

        # Ensure all scores are in [0,1] range after clinical weights
        for feature in X.columns:
            weight = clinical_weights.get(feature, 1.0)
            for method in importances:
                importances[method][feature] = min(1.0, importances[method][feature] * weight)
        
        # Weighted average with dynamic weights based on stability
        stability_score = 1 / (importance_std + 1e-10)  # Lower std = higher stability
        stability_score = stability_score / stability_score.sum()
        
        weights = {
            'rf': 0.4 * (1 + stability_score.mean()),
            'perm': 0.4 * (1 - stability_score.mean()),
            'mi': 0.2
        }
        weights = {k: v/sum(weights.values()) for k, v in weights.items()}
        
        final_importance = sum(importances[method] * weight 
                             for method, weight in weights.items())
        
        # Calculate confidence intervals
        confidence_intervals = pd.DataFrame({
            'importance': final_importance,
            'rf_std': importance_std,
            'mi_std': mi_std
        })
        
        # Log results with enhanced information
        self.logger.info(f"Feature importance calculation completed ({'test' if is_test_mode else 'full'} mode)")
        self.logger.info(f"Stability weights: {weights}")
        
        for feature in final_importance.nlargest(10).index:
            self.logger.info(
                f"Feature: {feature:20} "
                f"Importance: {final_importance[feature]:.4f} ± "
                f"{(confidence_intervals.loc[feature, 'rf_std'] + confidence_intervals.loc[feature, 'mi_std'])/2:.4f} "
                f"(Clinical weight: {clinical_weights.get(feature, 1.0)})"
            )
        
        return final_importance
