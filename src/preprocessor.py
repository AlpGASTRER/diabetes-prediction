import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from typing import Union, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class PreProcessor:
    """Medical-grade preprocessor for diabetes prediction"""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = None
        self.selected_features = None
        self.feature_names_ = None
        self.continuous_features_ = None
        self.original_features = None
        self.feature_order_ = None
        
        # Define expected column names
        self.required_columns = [
            'BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth',
            'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk',
            'Education', 'Income', 'Sex'
        ]
        
        # Column name mappings (dataset -> our names)
        self.column_mapping = {
            'Diabetes_binary': 'Diabetes',
            'HighBP': 'HighBP',
            'HighChol': 'HighChol',
            'CholCheck': 'CholCheck',
            'BMI': 'BMI',
            'Smoker': 'Smoker',
            'Stroke': 'Stroke',
            'HeartDiseaseorAttack': 'HeartDiseaseorAttack',
            'PhysActivity': 'PhysActivity',
            'Fruits': 'Fruits',
            'Veggies': 'Veggies',
            'HvyAlcoholConsump': 'HvyAlcoholConsump',
            'AnyHealthcare': 'AnyHealthcare',
            'NoDocbcCost': 'NoDocbcCost',
            'GenHlth': 'GenHlth',
            'MentHlth': 'MentHlth',
            'PhysHlth': 'PhysHlth',
            'DiffWalk': 'DiffWalk',
            'Age': 'Age',
            'Education': 'Education',
            'Income': 'Income'
        }
        
        # Define feature groups
        self.continuous_features = ['BMI', 'PhysHlth', 'MentHlth']  # Removed Age
        self.binary_features = [
            'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
        ]
        self.categorical_features = ['GenHlth', 'Education', 'Income', 'Age']  # Added Age
        
        # BRFSS 2015 Age Groups (1-14)
        self.age_groups = {
            1: "Age 18-24",
            2: "Age 25-29",
            3: "Age 30-34",
            4: "Age 35-39",
            5: "Age 40-44",
            6: "Age 45-49",
            7: "Age 50-54",
            8: "Age 55-59",
            9: "Age 60-64",
            10: "Age 65-69",
            11: "Age 70-74",
            12: "Age 75-79",
            13: "Age 80+",
            14: "Don't know/Refused/Missing"
        }
        
        # Medical thresholds based on clinical guidelines
        self.medical_thresholds = {
            'BMI': {'low': 16, 'high': 50},
            'Age': {'low': 18, 'high': 100},
            'PhysHlth': {'low': 0, 'high': 30},
            'GenHlth': {'low': 1, 'high': 5}
        }
        
        self.feature_importance = None
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=False,
            n_jobs=-1
        )
        self.adasyn = ADASYN(
            sampling_strategy='auto',  # Will be adjusted dynamically
            random_state=42,
            n_neighbors=15,  # Increased for better density estimation
            n_jobs=-1
        )
        
    def fit(self, X, y=None):
        """Fit the preprocessor."""
        try:
            # Convert numpy array to DataFrame if necessary
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
            # Store original feature names
            self.original_features = X.columns.tolist()
            self.feature_names_ = self.required_columns
            
            # Validate required columns and add missing ones
            missing_cols = set(self.required_columns) - set(X.columns)
            if missing_cols:
                print(f"Warning: Missing required columns: {missing_cols}")
                # Add missing columns with default values
                for col in missing_cols:
                    print(f"Adding {col} with default value 0")
                    X[col] = 0
            
            # Ensure columns are in the same order
            X = X[self.required_columns]
            
            # Handle missing values
            X_clean = self._handle_missing_values(X)
            
            # Scale features
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(X_clean)
            
            return self
            
        except Exception as e:
            print(f"Error in fit: {str(e)}")
            raise

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, pd.Series]:
        """Fit the preprocessor and transform the data."""
        # Validate input data
        X = self._validate_data(X)
        
        # Store feature names
        self.feature_names_ = X.columns.tolist()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Fit and transform scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Store feature order
        self.feature_order_ = X.columns.tolist()
        
        # Ensure no NaN values exist after scaling
        if np.isnan(X_scaled).any():
            print("Warning: NaN values found after scaling, filling with 0")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        return X_scaled, y

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform the input data."""
        try:
            # Convert numpy array to DataFrame if needed
            if isinstance(X, np.ndarray):
                if self.feature_names_ is None:
                    raise ValueError("feature_names_ not set. Call fit_transform first.")
                X = pd.DataFrame(X, columns=self.feature_names_)
            elif isinstance(X, pd.DataFrame):
                X = X.copy()
            else:
                raise ValueError("Input must be a pandas DataFrame or numpy array")

            # Drop target variable if present
            if 'Diabetes_binary' in X.columns:
                X = X.drop('Diabetes_binary', axis=1)
            
            # Ensure we have all required columns in the correct order
            missing_cols = set(self.required_columns) - set(X.columns)
            if missing_cols:
                for col in missing_cols:
                    print(f"Warning: Adding missing column {col} with default value 0")
                    X[col] = 0
            
            # Ensure columns are in the same order as during training
            if self.feature_order_ is None:
                raise ValueError("feature_order_ not set. Call fit_transform first.")
            
            # Select and order columns
            try:
                X = X[self.feature_order_]
            except KeyError as e:
                print(f"Error: Missing columns from feature_order_. Available columns: {X.columns.tolist()}")
                print(f"Required columns: {self.feature_order_}")
                raise
            
            # Handle missing values
            X = self._handle_missing_values(X)
            
            # Convert to numpy array before scaling
            X_array = X.values
            
            # Scale features if scaler exists
            if self.scaler is not None:
                try:
                    X_array = self.scaler.transform(X_array)
                except Exception as e:
                    print(f"Error during scaling: {str(e)}")
                    print(f"Input shape: {X_array.shape}")
                    print(f"Expected features: {len(self.feature_order_)}")
                    raise
            
            # Ensure no NaN values exist after scaling
            if np.isnan(X_array).any():
                print("Warning: NaN values found after scaling, filling with 0")
                X_array = np.nan_to_num(X_array, nan=0.0)
            
            return X_array
            
        except Exception as e:
            print(f"Error in transform: {str(e)}")
            print(f"Input type: {type(X)}")
            if isinstance(X, (pd.DataFrame, np.ndarray)):
                print(f"Input shape: {X.shape}")
            raise

    def _validate_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input data structure and values."""
        # Check required columns
        missing_cols = set(self.required_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check binary features
        for col in self.binary_features:
            if not X[col].isin([0, 1, np.nan]).all():
                raise ValueError(f"Binary feature {col} contains non-binary values")
                
        # Check categorical features
        for col in self.categorical_features:
            if col == 'GenHlth':
                valid_range = range(1, 6)
            elif col == 'Education':
                valid_range = range(1, 7)
            elif col == 'Income':
                valid_range = range(1, 9)
            elif col == 'Age':
                valid_range = range(1, 15)  # BRFSS 2015 Age Groups
            else:
                continue
                
            invalid_mask = ~X[col].isin(valid_range) & ~X[col].isna()
            if invalid_mask.any():
                raise ValueError(f"Invalid values in {col}")
                
        # Check continuous features
        for col in self.continuous_features:
            if col in self.medical_thresholds:
                thresholds = self.medical_thresholds[col]
                invalid_mask = (
                    (X[col] < thresholds['low']) | 
                    (X[col] > thresholds['high'])
                ) & ~X[col].isna()
                
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    print(f"Warning: {invalid_count} invalid values found in {col}")
                    # Replace invalid values with NaN for later imputation
                    X.loc[invalid_mask, col] = np.nan
                    
        return X

    def _remove_duplicates(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicates efficiently."""
        return X.drop_duplicates(subset=self.required_columns)
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        # Fill missing values with appropriate defaults
        defaults = {
            'BMI': X['BMI'].dropna().median(),
            'Age': X['Age'].dropna().mode()[0],  # Use mode for Age
            'GenHlth': X['GenHlth'].dropna().mode()[0],
            'MentHlth': X['MentHlth'].dropna().median(),
            'PhysHlth': X['PhysHlth'].dropna().median(),
            'HighBP': X['HighBP'].dropna().mode()[0],
            'HighChol': X['HighChol'].dropna().mode()[0],
            'Smoker': X['Smoker'].dropna().mode()[0],
            'Stroke': X['Stroke'].dropna().mode()[0],
            'HeartDiseaseorAttack': X['HeartDiseaseorAttack'].dropna().mode()[0],
            'PhysActivity': X['PhysActivity'].dropna().mode()[0],
            'Fruits': X['Fruits'].dropna().mode()[0],
            'Veggies': X['Veggies'].dropna().mode()[0],
            'HvyAlcoholConsump': X['HvyAlcoholConsump'].dropna().mode()[0],
            'AnyHealthcare': X['AnyHealthcare'].dropna().mode()[0],
            'NoDocbcCost': X['NoDocbcCost'].dropna().mode()[0],
            'DiffWalk': X['DiffWalk'].dropna().mode()[0]
        }
        
        # Fill missing values
        for col, default in defaults.items():
            if col in X.columns:
                X[col] = X[col].fillna(default)
        
        return X

    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using LOF."""
        try:
            # Only detect outliers in continuous features
            outlier_scores = self.lof.predict(X)
            outlier_mask = outlier_scores == -1
            
            if outlier_mask.any():
                print(f"Found {outlier_mask.sum()} outliers")
                # For outliers, replace with median of non-outlier values
                for i in range(X.shape[1]):
                    non_outlier_median = np.nanmedian(X[~outlier_mask, i])
                    X[outlier_mask, i] = non_outlier_median
                    
            return X
            
        except Exception as e:
            print(f"Error in outlier detection: {str(e)}")
            return X  # Return original data if outlier detection fails

    def _create_medical_scores(self, X):
        """Create medical risk scores based on clinical guidelines."""
        try:
            X = X.copy()
            
            # Ensure all required features exist
            required_features = ['BMI', 'HighBP', 'HighChol', 'Age']
            missing_features = [f for f in required_features if f not in self.feature_names_]
            if missing_features:
                print(f"Warning: Missing features for risk score calculation: {missing_features}")
                return X
            
            # Basic risk score based on key medical factors with safe indexing
            risk_score = np.zeros(X.shape[0])
            
            # BMI risk (0-3 points)
            bmi_idx = self.feature_names_.index('BMI')
            risk_score += (X[:, bmi_idx] >= 30).astype(int) * 3
            
            # Blood pressure risk (0-2 points)
            bp_idx = self.feature_names_.index('HighBP')
            risk_score += (X[:, bp_idx] == 1).astype(int) * 2
            
            # Cholesterol risk (0-2 points)
            chol_idx = self.feature_names_.index('HighChol')
            risk_score += (X[:, chol_idx] == 1).astype(int) * 2
            
            # Age risk (0-3 points) - Using BRFSS age groups
            age_idx = self.feature_names_.index('Age')
            age_values = X[:, age_idx].astype(int)
            
            # Age 45-54 (groups 6-7): 2 points
            middle_age_mask = (age_values >= 6) & (age_values <= 7)
            risk_score += middle_age_mask.astype(int) * 2
            
            # Age 55+ (groups 8-13): 3 points
            older_age_mask = (age_values >= 8) & (age_values <= 13)
            risk_score += older_age_mask.astype(int) * 3
            
            # Add risk score as new feature
            X = np.hstack((X, risk_score.reshape(-1, 1)))
            self.feature_names_.append('BasicRiskScore')
            
            # Create risk categories using qcut with unique bins
            try:
                # Create numeric risk categories (1-5)
                risk_categories = pd.qcut(
                    risk_score,
                    q=5,
                    labels=False,  # Use numeric labels
                    duplicates='drop'
                ) + 1  # Add 1 to make it 1-based instead of 0-based
            except ValueError as e:
                # If qcut fails due to too few unique values, use cut instead
                unique_values = np.unique(risk_score)
                n_bins = min(5, len(unique_values))
                risk_categories = pd.cut(
                    risk_score,
                    bins=n_bins,
                    labels=False,  # Use numeric labels
                    duplicates='drop'
                ) + 1  # Add 1 to make it 1-based
            
            # Add risk categories and handle any NaN values
            risk_categories = np.nan_to_num(risk_categories, nan=3)  # Use moderate risk (3) for NaN
            X = np.hstack((X, risk_categories.reshape(-1, 1)))
            self.feature_names_.append('RiskCategory')
            
            return X
            
        except Exception as e:
            print(f"Error in creating medical scores: {str(e)}")
            # Return original data if score creation fails
            return X

    def _handle_class_imbalance(self, X, y):
        """Handle class imbalance with risk-stratified sampling."""
        try:
            # Calculate risk scores if not already present
            if 'BasicRiskScore' not in self.feature_names_:
                X = self._create_medical_scores(X)
            
            # Get risk score index
            risk_score_idx = self.feature_names_.index('BasicRiskScore')
            
            # Create risk strata
            risk_scores = X[:, risk_score_idx]
            risk_strata = pd.qcut(risk_scores, q=5, labels=['VL', 'L', 'M', 'H', 'VH'])
            
            # Initialize resampled data containers
            X_resampled_parts = []
            y_resampled_parts = []
            
            # Process each risk stratum separately
            for stratum in ['VL', 'L', 'M', 'H', 'VH']:
                stratum_mask = risk_strata == stratum
                if not np.any(stratum_mask):
                    continue
                
                X_stratum = X[stratum_mask]
                y_stratum = y[stratum_mask]
                
                # Skip if too few samples
                if len(y_stratum) < 10 or len(np.unique(y_stratum)) < 2:
                    X_resampled_parts.append(X_stratum)
                    y_resampled_parts.append(y_stratum)
                    continue
                
                # Calculate sampling ratio based on risk level
                if stratum in ['VL', 'L']:
                    sampling_ratio = 0.6  # Less aggressive for low risk
                elif stratum == 'M':
                    sampling_ratio = 0.7  # Moderate for medium risk
                else:
                    sampling_ratio = 0.8  # More aggressive for high risk
                
                try:
                    # Apply ADASYN with stratum-specific sampling
                    self.adasyn.sampling_strategy = sampling_ratio
                    X_resampled, y_resampled = self.adasyn.fit_resample(X_stratum, y_stratum)
                    
                    X_resampled_parts.append(X_resampled)
                    y_resampled_parts.append(y_resampled)
                    
                except Exception as e:
                    print(f"ADASYN failed for {stratum} stratum: {str(e)}")
                    # Fall back to original data for this stratum
                    X_resampled_parts.append(X_stratum)
                    y_resampled_parts.append(y_stratum)
            
            # Combine all resampled strata
            X_resampled = np.vstack(X_resampled_parts)
            y_resampled = np.hstack(y_resampled_parts)
            
            # Shuffle the combined data
            shuffle_idx = np.random.permutation(len(y_resampled))
            X_resampled = X_resampled[shuffle_idx]
            y_resampled = y_resampled[shuffle_idx]
            
            print("\nResampling Summary:")
            print(f"Original class distribution - 0: {sum(y == 0)}, 1: {sum(y == 1)}")
            print(f"Resampled class distribution - 0: {sum(y_resampled == 0)}, 1: {sum(y_resampled == 1)}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"Error in class imbalance handling: {str(e)}")
            return X, y  # Return original data if resampling fails

    def _create_clinical_interactions(self, X):
        """Create clinical interaction features."""
        X_new = X.copy()
        
        # Store original values for required features
        original_values = {col: X_new[:, self.feature_names_.index(col)].copy() for col in self.feature_names_ if col in self.feature_names_}
        
        # Create interaction terms if features are present
        if all(col in self.feature_names_ for col in ['HighBP', 'HighChol']):
            X_new = np.hstack((X_new, (X_new[:, self.feature_names_.index('HighBP')] * X_new[:, self.feature_names_.index('HighChol')]).reshape(-1, 1)))
            self.feature_names_.append('BP_Chol')
        
        if all(col in self.feature_names_ for col in ['HeartDiseaseorAttack', 'HighBP']):
            X_new = np.hstack((X_new, (X_new[:, self.feature_names_.index('HeartDiseaseorAttack')] * X_new[:, self.feature_names_.index('HighBP')]).reshape(-1, 1)))
            self.feature_names_.append('Heart_BP')
        
        if all(col in self.feature_names_ for col in ['BMI', 'HighBP']):
            X_new = np.hstack((X_new, (X_new[:, self.feature_names_.index('BMI')] * X_new[:, self.feature_names_.index('HighBP')]).reshape(-1, 1)))
            self.feature_names_.append('BMI_BP')
        
        # Restore original values
        for col in self.feature_names_:
            if col in original_values:
                X_new[:, self.feature_names_.index(col)] = original_values[col]
        
        return X_new
    
    def _remove_correlated_features(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Remove highly correlated features efficiently while preserving medical relevance."""
        try:
            # Define features that should never be dropped due to medical importance
            critical_features = [
                'BMI', 'Age', 'HighBP', 'HighChol', 'HeartDiseaseorAttack',
                'GenHlth', 'BasicRiskScore', 'RiskCategory'
            ]
            
            # Calculate correlation matrix only for numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            corr_matrix = X[numeric_cols].corr().abs()
            
            # Calculate correlation with target if provided
            if y is not None:
                target_corr = pd.Series(
                    [mutual_info_classif(X[[col]], y, random_state=42)[0] for col in numeric_cols],
                    index=numeric_cols
                )
                print("\nFeature-Target Correlations:")
                for feat, corr in target_corr.sort_values(ascending=False).items():
                    print(f"{feat}: {corr:.3f}")
            
            # Find features to drop
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = set()
            correlations_found = []
            
            # Iterate through feature pairs with correlation > threshold
            for i in range(len(upper.columns)):
                for j in range(i):
                    corr_value = upper.iloc[j, i]
                    if corr_value > 0.4:  # Lower correlation threshold
                        feat1, feat2 = upper.columns[i], upper.columns[j]
                        correlations_found.append((feat1, feat2, corr_value))
                        
                        # Skip if both features are critical
                        if feat1 in critical_features and feat2 in critical_features:
                            continue
                            
                        # If one feature is critical, drop the other
                        if feat1 in critical_features:
                            to_drop.add(feat2)
                            continue
                        if feat2 in critical_features:
                            to_drop.add(feat1)
                            continue
                        
                        # If target correlation is available, keep feature with higher correlation
                        if y is not None:
                            if target_corr[feat1] > target_corr[feat2]:
                                to_drop.add(feat2)
                            else:
                                to_drop.add(feat1)
                        else:
                            # If no target correlation, keep the first feature
                            to_drop.add(feat2)
            
            # Print correlation findings
            if correlations_found:
                print("\nCorrelated Feature Pairs (>0.4):")
                for feat1, feat2, corr in sorted(correlations_found, key=lambda x: x[2], reverse=True):
                    status = "kept" if feat1 not in to_drop and feat2 not in to_drop else \
                            "dropped" if feat1 in to_drop and feat2 in to_drop else \
                            f"kept {feat1 if feat1 not in to_drop else feat2}"
                    print(f"{feat1} - {feat2}: {corr:.3f} ({status})")
            
            # Store and return selected features
            self.selected_features = [col for col in X.columns if col not in to_drop]
            
            # Print summary
            print(f"\nFeature Selection Summary:")
            print(f"Original features: {len(X.columns)}")
            print(f"Selected features: {len(self.selected_features)}")
            print(f"Dropped features: {sorted(to_drop)}")
            
            return X[self.selected_features]
            
        except Exception as e:
            print(f"Error in removing correlated features: {str(e)}")
            return X  # Return original data if correlation analysis fails

    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calculate feature importance using Random Forest."""
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        rf.fit(X, y)
        
        # Store feature importance as a dictionary
        self.feature_importance = dict(zip(X.columns, rf.feature_importances_))
        
    def get_feature_importance(self) -> dict:
        """Get feature importance scores."""
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Call fit first.")
        return self.feature_importance

    def _detect_outliers(self, X: pd.DataFrame) -> np.ndarray:
        """Detect outliers using LOF."""
        try:
            # Get continuous data
            continuous_data = X.copy()
            
            # Handle any remaining missing values
            for i in range(continuous_data.shape[1]):
                if np.isnan(continuous_data[:, i]).any():
                    continuous_data[:, i] = np.nan_to_num(continuous_data[:, i], nan=np.nanmedian(continuous_data[:, i]))
            
            # Fit and predict outliers
            outlier_labels = self.lof.fit_predict(continuous_data)
            return outlier_labels == -1
            
        except Exception as e:
            print(f"Error in outlier detection: {str(e)}")
            return np.zeros(len(X), dtype=bool)  # Return no outliers if detection fails

    def get_feature_names(self):
        """Get list of feature names after preprocessing."""
        return [
            'BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth', 'HighBP', 'HighChol',
            'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
            'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk',
            'Education', 'Income', 'BasicRiskScore', 'RiskCategory'
        ]
