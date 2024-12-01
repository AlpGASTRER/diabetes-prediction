import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import Tuple

class DataLoader:
    """
    Class for loading and splitting the diabetes dataset.
    
    This class handles loading the diabetes dataset from a CSV file and splitting it
    into training, validation, and test sets while maintaining class distribution.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader with path validation.
        
        Args:
            data_path (str): Path to the dataset CSV file
            
        Raises:
            FileNotFoundError: If the data_path does not exist
            ValueError: If the data_path is not a CSV file
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        if not data_path.endswith('.csv'):
            raise ValueError("Data file must be a CSV file")
        
        self.data_path = data_path
        self.target_column = 'Diabetes_binary'
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the diabetes dataset with error handling.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            pd.errors.EmptyDataError: If the CSV file is empty
            Exception: For other reading errors
        """
        try:
            data = pd.read_csv(self.data_path)
            if data.empty:
                raise pd.errors.EmptyDataError("The CSV file is empty")
            if self.target_column not in data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in dataset")
            return data
        except pd.errors.EmptyDataError:
            raise
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def split_data(self, data: pd.DataFrame, test_size: float = 0.2, 
                   val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                         pd.DataFrame, pd.Series, 
                                                                         pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets while preserving class distribution.
        
        Args:
            data (pd.DataFrame): Input dataset
            test_size (float): Proportion of dataset to include in the test split (0 to 1)
            val_size (float): Proportion of training dataset to include in the validation split (0 to 1)
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test) where:
                - X_train, X_val, X_test are DataFrames containing feature variables
                - y_train, y_val, y_test are Series containing target variables
                
        Raises:
            ValueError: If split proportions are invalid or if data is empty
        """
        if not 0 < test_size < 1 or not 0 < val_size < 1:
            raise ValueError("Split sizes must be between 0 and 1")
        if test_size + val_size >= 1:
            raise ValueError("Sum of test_size and val_size must be less than 1")
        
        if data.empty:
            raise ValueError("Cannot split empty dataset")
            
        # First split: training + validation and test
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_size,
            random_state=random_state,
            stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_names(self) -> list:
        """
        Get the list of feature names from the dataset.
        
        Returns:
            list: List of feature column names excluding the target column
        """
        data = self.load_data()
        return list(data.drop(self.target_column, axis=1).columns)
