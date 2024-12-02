import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
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
        self.required_columns = [
            'BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth',
            'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
            'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
            'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
        ]
        
        # Define feature groups
        self.continuous_features = ['BMI', 'PhysHlth', 'MentHlth', 'Age']
        self.binary_features = [
            'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
        ]
        self.categorical_features = ['GenHlth', 'Education', 'Income']
        
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
            novelty=False,  # Set to False for fit_predict
            n_jobs=-1
        )
        self.adasyn = ADASYN(
            sampling_strategy='auto',  # Will be adjusted dynamically
            random_state=42,
            n_neighbors=15,  # Increased for better density estimation
            n_jobs=-1
        )
        
    def fit(self, X, y=None):
        """Fit the preprocessor to the data."""
        try:
            # Store feature names and required columns
            self.feature_names_ = list(X.columns)
            
            # Validate required columns
            missing_cols = set(self.required_columns) - set(X.columns)
            if missing_cols:
                print(f"Warning: Missing required columns: {missing_cols}")
                for col in missing_cols:
                    print(f"Adding {col} with default value 0")
                    X[col] = 0
            
            # Handle missing values
            X_clean = self._handle_missing_values(X)
            
            # Define and store continuous features
            self.continuous_features_ = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth']
            
            # Initialize and fit scaler on all continuous features at once
            self.scaler = StandardScaler()
            continuous_features_present = [col for col in self.continuous_features_ if col in X_clean.columns]
            if continuous_features_present:
                self.scaler.fit(X_clean[continuous_features_present])
            
            return self
        except Exception as e:
            print(f"Error in fit: {str(e)}")
            raise

    def fit_transform(self, X, y=None):
        """Fit and transform the data."""
        try:
            # Store original feature names and data
            self.feature_names_ = list(X.columns)
            X_clean = X.copy()
            
            # Handle missing values
            print("Handling missing values...")
            X_clean = self._handle_missing_values(X_clean)
            
            # Ensure all required columns are present
            missing_cols = set(self.required_columns) - set(X_clean.columns)
            if missing_cols:
                print(f"Warning: Missing required columns: {missing_cols}")
                for col in missing_cols:
                    print(f"Adding {col} with default value 0")
                    X_clean[col] = 0
            
            # Define and store continuous features
            self.continuous_features_ = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth']
            
            # Initialize and fit scaler on all continuous features at once
            self.scaler = StandardScaler()
            continuous_features_present = [col for col in self.continuous_features_ if col in X_clean.columns]
            if continuous_features_present:
                X_clean[continuous_features_present] = self.scaler.fit_transform(X_clean[continuous_features_present]).astype('float32')
            
            # Store original feature values before creating new features
            self.original_features = {col: X_clean[col].copy() for col in self.feature_names_}
            
            # Create medical risk scores
            print("\nCreating medical risk scores...")
            X_clean = self._create_medical_scores(X_clean)
            
            # Create clinical features
            print("\nCreating clinical features...")
            X_clean = self._create_clinical_interactions(X_clean)
            
            # Create a new DataFrame with original features
            result = pd.DataFrame(0, index=X_clean.index, columns=self.feature_names_)
            for col in self.feature_names_:
                if col in X_clean.columns:
                    result[col] = X_clean[col]
                elif col in self.original_features:
                    result[col] = self.original_features[col]
        
            return result, y
        except Exception as e:
            print(f"Error in fit_transform: {str(e)}")
            raise

    def transform(self, X):
        """Transform the input data using the fitted preprocessor."""
        try:
            # Store original data
            X_clean = X.copy()
            
            # Ensure all required columns are present
            missing_cols = set(self.required_columns) - set(X_clean.columns)
            if missing_cols:
                print(f"Warning: Missing required columns: {missing_cols}")
                for col in missing_cols:
                    print(f"Adding {col} with default value 0")
                    X_clean[col] = 0
            
            # Handle missing values
            print("Handling missing values...")
            X_clean = self._handle_missing_values(X_clean)
            
            # Create medical risk scores
            print("\nCreating medical risk scores...")
            X_clean = self._create_medical_scores(X_clean)
            
            # Create clinical features
            print("\nCreating clinical features...")
            X_clean = self._create_clinical_interactions(X_clean)
            
            # Scale features
            print("\nScaling features...")
            continuous_features_present = [col for col in self.continuous_features_ if col in X_clean.columns]
            if continuous_features_present:
                X_clean[continuous_features_present] = self.scaler.transform(X_clean[continuous_features_present]).astype('float32')
            
            # Create a new DataFrame with original features
            result = pd.DataFrame(0, index=X_clean.index, columns=self.feature_names_)
            
            # First, copy over all original features from X_clean
            for col in self.feature_names_:
                if col in X_clean.columns:
                    result[col] = X_clean[col]
                elif col in self.original_features:
                    result[col] = self.original_features[col].iloc[0]  # Use first value as default
            
            # Ensure all features are float32
            for col in result.columns:
                result[col] = result[col].astype('float32')
            
            return result
        
        except Exception as e:
            print(f"Error in transform: {str(e)}")
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
    
    def _handle_missing_values(self, X):
        """Handle missing values in the data."""
        try:
            X_clean = X.copy()
            
            # Fill missing values in continuous features with median
            continuous_features = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth']
            for col in continuous_features:
                if col in X_clean.columns:
                    median_val = X_clean[col].median()
                    X_clean[col] = X_clean[col].fillna(median_val)
            
            # Fill missing values in categorical features with mode
            categorical_features = [col for col in X_clean.columns if col not in continuous_features]
            for col in categorical_features:
                if col in X_clean.columns:
                    mode_val = X_clean[col].mode().iloc[0]
                    X_clean[col] = X_clean[col].fillna(mode_val)
            
            # Ensure all required features are present
            required_features = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth']
            for feature in required_features:
                if feature not in X_clean.columns:
                    print(f"Warning: Required feature {feature} not found in data. Adding with default value.")
                    X_clean[feature] = 0
            
            # Ensure all features are float32
            for col in X_clean.columns:
                X_clean[col] = X_clean[col].astype('float32')
            
            return X_clean
        
        except Exception as e:
            print(f"Error in handling missing values: {str(e)}")
            raise

    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using LOF."""
        try:
            # Only detect outliers in continuous features
            outlier_scores = self.lof.predict(X[self.continuous_features])
            outlier_mask = outlier_scores == -1
            
            if outlier_mask.any():
                print(f"Found {outlier_mask.sum()} outliers")
                # For outliers, replace with median of non-outlier values
                for col in self.continuous_features:
                    non_outlier_median = X.loc[~outlier_mask, col].median()
                    X.loc[outlier_mask, col] = non_outlier_median
                    
            return X
            
        except Exception as e:
            print(f"Error in outlier detection: {str(e)}")
            return X  # Return original data if outlier detection fails

    def _create_medical_scores(self, X):
        """Create medical risk scores with stratification."""
        try:
            # Basic medical risk score (weighted sum of key risk factors)
            risk_weights = {
                'BMI': 0.3,
                'Age': 0.2,
                'HighBP': 0.15,
                'HighChol': 0.15,
                'HeartDiseaseorAttack': 0.2
            }
            
            # Create age groups for stratification
            X['AgeGroup'] = pd.cut(X['Age'], 
                                 bins=[0, 35, 45, 55, 65, 100],
                                 labels=['<35', '35-45', '45-55', '55-65', '>65'])
            
            # Calculate BMI categories
            X['BMICategory'] = pd.cut(X['BMI'],
                                    bins=[0, 18.5, 25, 30, 35, 100],
                                    labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'SeverelyObese'])
            
            # Calculate basic risk score
            risk_score = sum(X[col] * weight for col, weight in risk_weights.items())
            X['BasicRiskScore'] = risk_score
            
            # Create comorbidity score
            comorbidity_factors = ['HighBP', 'HighChol', 'HeartDiseaseorAttack', 'Stroke']
            X['ComorbidityScore'] = X[comorbidity_factors].sum(axis=1)
            
            # Create lifestyle score (inverse scoring for positive factors)
            X['LifestyleScore'] = (
                (1 - X['PhysActivity']) * 0.4 +
                (1 - X['Fruits']) * 0.2 +
                (1 - X['Veggies']) * 0.2 +
                X['Smoker'] * 0.2
            )
            
            # Create healthcare access score
            X['HealthcareAccessScore'] = (
                X['AnyHealthcare'] * 0.6 +
                (1 - X['NoDocbcCost']) * 0.4
            )
            
            # Create composite risk categories
            X['RiskCategory'] = pd.qcut(X['BasicRiskScore'], q=5, 
                                      labels=['VeryLow', 'Low', 'Moderate', 'High', 'VeryHigh'])
            
            # Create interaction features for high-risk groups
            high_risk_mask = X['RiskCategory'].isin(['High', 'VeryHigh'])
            X.loc[high_risk_mask, 'HighRiskComorbidity'] = (
                X.loc[high_risk_mask, 'ComorbidityScore'] * 
                X.loc[high_risk_mask, 'BasicRiskScore']
            )
            
            return X
            
        except Exception as e:
            print(f"Error in creating medical scores: {str(e)}")
            raise

    def _create_clinical_interactions(self, X):
        """Create clinical interaction features."""
        X_new = X.copy()
        
        # Store original values for required features
        original_values = {col: X_new[col].copy() for col in self.feature_names_ if col in X_new.columns}
        
        # Create interaction terms if features are present
        if all(col in X_new.columns for col in ['HighBP', 'HighChol']):
            X_new['BP_Chol'] = X_new['HighBP'] * X_new['HighChol']
        
        if all(col in X_new.columns for col in ['HeartDiseaseorAttack', 'HighBP']):
            X_new['Heart_BP'] = X_new['HeartDiseaseorAttack'] * X_new['HighBP']
        
        if all(col in X_new.columns for col in ['BMI', 'HighBP']):
            X_new['BMI_BP'] = X_new['BMI'] * X_new['HighBP']
        
        # Restore original values
        for col in self.feature_names_:
            if col in original_values:
                X_new[col] = original_values[col]
        
        return X_new
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features efficiently."""
        # Calculate correlation matrix only for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        corr_matrix = X[numeric_cols].corr().abs()
        
        # Find features to drop
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column for column in upper.columns 
            if any(upper[column] > 0.8)
        ]
        
        # Store and return selected features
        self.selected_features = [col for col in X.columns if col not in to_drop]
        return X[self.selected_features]
    
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
            continuous_data = X[self.continuous_features].copy()
            
            # Handle any remaining missing values
            for col in continuous_data.columns:
                if continuous_data[col].isna().any():
                    continuous_data[col] = continuous_data[col].fillna(continuous_data[col].median())
            
            # Fit and predict outliers
            outlier_labels = self.lof.fit_predict(continuous_data)
            return outlier_labels == -1
            
        except Exception as e:
            print(f"Error in outlier detection: {str(e)}")
            return np.zeros(len(X), dtype=bool)  # Return no outliers if detection fails

    def _handle_class_imbalance(self, X, y):
        """Handle class imbalance with risk-stratified sampling."""
        try:
            # Calculate risk scores if not already present
            if 'BasicRiskScore' not in X.columns:
                X = self._create_medical_scores(X)
            
            # Create risk strata
            risk_strata = pd.qcut(X['BasicRiskScore'], q=5, labels=['VL', 'L', 'M', 'H', 'VH'])
            
            # Initialize resampled data containers
            X_resampled_parts = []
            y_resampled_parts = []
            
            # Process each risk stratum separately
            for stratum in risk_strata.unique():
                stratum_mask = risk_strata == stratum
                X_stratum = X[stratum_mask]
                y_stratum = y[stratum_mask]
                
                # Skip if no minority class samples in stratum
                if len(y_stratum.unique()) < 2:
                    X_resampled_parts.append(X_stratum)
                    y_resampled_parts.append(y_stratum)
                    continue
                
                # Adjust sampling strategy based on risk level
                if stratum in ['H', 'VH']:
                    sampling_ratio = 0.8  # More aggressive sampling for high-risk groups
                elif stratum == 'M':
                    sampling_ratio = 0.6  # Moderate sampling for medium risk
                else:
                    sampling_ratio = 0.4  # Conservative sampling for low-risk groups
                
                # Apply ADASYN with stratum-specific sampling
                self.adasyn.sampling_strategy = sampling_ratio
                X_res, y_res = self.adasyn.fit_resample(X_stratum, y_stratum)
                
                X_resampled_parts.append(X_res)
                y_resampled_parts.append(y_res)
            
            # Combine resampled data
            X_resampled = pd.concat(X_resampled_parts, axis=0)
            y_resampled = pd.concat(y_resampled_parts, axis=0)
            
            # Print resampling statistics
            print("\nResampling Statistics:")
            print(f"Original class distribution: {pd.Series(y).value_counts().to_dict()}")
            print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"Error in class imbalance handling: {str(e)}")
            # Fallback to original data if resampling fails
            return X, y
