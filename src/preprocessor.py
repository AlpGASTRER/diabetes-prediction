import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN, RandomOverSampler
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
            sampling_strategy=0.75,  # Target 75% of majority class
            random_state=42,
            n_neighbors=5
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
        """Create medical risk scores."""
        X_new = X.copy()
        
        # Store original values for required features
        original_values = {col: X_new[col].copy() for col in self.feature_names_ if col in X_new.columns}
        
        # Create BMI categories
        if 'BMI' in X_new.columns:
            X_new['BMI_Category'] = pd.cut(
                X_new['BMI'],
                bins=[-float('inf'), 18.5, 25, 30, float('inf')],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese']
            )
        
        # Create Age groups
        if 'Age' in X_new.columns:
            X_new['Age_Group'] = pd.cut(
                X_new['Age'],
                bins=[-float('inf'), 35, 50, 65, float('inf')],
                labels=['Young', 'Middle', 'Senior', 'Elderly']
            )
        
        # Convert categories to dummy variables
        X_new = pd.get_dummies(X_new, columns=['BMI_Category', 'Age_Group'])
        
        # Restore original values
        for col in self.feature_names_:
            if col in original_values:
                X_new[col] = original_values[col]
        
        return X_new

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
