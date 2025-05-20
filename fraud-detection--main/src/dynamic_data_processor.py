import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional, Union

class DynamicDataProcessor:
    """
    Dynamic data processor that automatically detects and handles different data types,
    applies appropriate preprocessing, and prepares data for Bayesian network modeling.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the data processor with configuration from JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.data_config = self.config["data_processing"]
        self.target_column = self.data_config["target_column"]
        self.categorical_columns = []
        self.numerical_columns = []
        self.binary_columns = []
        self.preprocessed = False
        
        # Initialize transformers
        self.num_scaler = None
        self.cat_encoder = None
        self.pca_transformer = None
    
    def detect_column_types(self, df: pd.DataFrame) -> None:
        """
        Automatically detect column types in the DataFrame.
        
        Args:
            df: Input DataFrame
        """
        self.categorical_columns = []
        self.numerical_columns = []
        self.binary_columns = []
        
        for col in df.columns:
            if col == self.target_column:
                continue
                
            # Check if column is categorical
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                self.categorical_columns.append(col)
            # Check if column is binary (only has 2 unique values)
            elif len(df[col].unique()) == 2:
                self.binary_columns.append(col)
            # Otherwise, treat as numerical
            else:
                self.numerical_columns.append(col)
        
        print(f"Detected {len(self.numerical_columns)} numerical features")
        print(f"Detected {len(self.categorical_columns)} categorical features")
        print(f"Detected {len(self.binary_columns)} binary features")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data: scale numerical features, encode categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Detect column types if not already done
        if not self.categorical_columns and not self.numerical_columns:
            self.detect_column_types(df)
        
        # Scale numerical features
        if self.numerical_columns:
            self.num_scaler = MinMaxScaler()
            processed_df[self.numerical_columns] = self.num_scaler.fit_transform(
                processed_df[self.numerical_columns]
            )
        
        # Handle categorical features
        if self.categorical_columns:
            self.cat_encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_cats = self.cat_encoder.fit_transform(processed_df[self.categorical_columns])
            
            # Create DataFrame with encoded categorical features
            encoded_df = pd.DataFrame(
                encoded_cats,
                columns=self.cat_encoder.get_feature_names_out(self.categorical_columns),
                index=processed_df.index
            )
            
            # Drop original categorical columns and add encoded ones
            processed_df = processed_df.drop(columns=self.categorical_columns)
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
        
        # Ensure all columns are numerical for further processing
        self.preprocessed = True
        return processed_df.select_dtypes(include=[np.number])
    
    def downsample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset through downsampling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Balanced DataFrame
        """
        # Skip if downsampling is disabled
        if not self.data_config["downsampling"]["enabled"]:
            return df
            
        ratio = self.data_config["downsampling"]["non_fraud_to_fraud_ratio"]
        
        # Separate fraud and non-fraud
        fraud = df[df[self.target_column] == 1]
        non_fraud = df[df[self.target_column] == 0]
        
        # Calculate target sample size
        target_non_fraud_count = int(len(fraud) * ratio)
        
        # Check if we have enough non-fraud samples
        if len(non_fraud) <= target_non_fraud_count:
            print(f"Warning: Not enough non-fraud samples ({len(non_fraud)}) to achieve {ratio}:1 ratio. Using all available.")
            non_fraud_downsampled = non_fraud
        else:
            # Downsample non-fraud to achieve the specified ratio
            non_fraud_downsampled = non_fraud.sample(
                n=target_non_fraud_count,
                random_state=self.data_config["random_state"]
            )
        
        # Combine and shuffle
        balanced_df = pd.concat([fraud, non_fraud_downsampled])
        return balanced_df.sample(
            frac=1,
            random_state=self.data_config["random_state"]
        ).reset_index(drop=True)
    
    def apply_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply PCA transformation to the features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with PCA components and target column
        """
        # Skip if PCA is disabled
        if not self.data_config["pca"]["enabled"]:
            return df
            
        n_components = self.data_config["pca"]["n_components"]
        features = df.drop(columns=[self.target_column])
        
        # Initialize and fit PCA
        self.pca_transformer = PCA(
            n_components=n_components,
            whiten=self.data_config["pca"]["whiten"],
            svd_solver=self.data_config["pca"]["svd_solver"],
            random_state=self.data_config["random_state"]
        )
        pca_result = self.pca_transformer.fit_transform(features)
        
        # Create DataFrame with PCA components
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'PCA_{i+1}' for i in range(n_components)],
            index=df.index
        )
        
        # Add target column
        pca_df[self.target_column] = df[self.target_column]
        
        # Store the explained variance for later use
        self.explained_variance_ratio = self.pca_transformer.explained_variance_ratio_
        
        return pca_df
    
    def discretize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Discretize continuous features into bins.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with discretized features
        """
        # Skip if discretization is disabled
        if not self.data_config["discretization"]["enabled"]:
            return df
            
        discretized = df.copy()
        bins = self.data_config["discretization"]["bins"]
        labels = self.data_config["discretization"]["labels"]
        
        # Get columns to discretize (all except target)
        columns_to_discretize = [col for col in df.columns if col != self.target_column]
        
        # Apply discretization
        for col in columns_to_discretize:
            discretized[col] = pd.qcut(
                pd.to_numeric(discretized[col], errors='coerce').fillna(discretized[col].median()),
                q=bins,
                labels=labels,
                duplicates='drop'
            ).astype(str)
        
        return discretized
    
    def process_pipeline(self, df: pd.DataFrame, return_test_data: bool = True) -> tuple:
        """
        Run the complete data processing pipeline on a DataFrame.
        
        Args:
            df: Input DataFrame
            return_test_data: Whether to split and return test data
            
        Returns:
            Prepared data as (X_train, X_test, y_train, y_test) if return_test_data=True
            otherwise returns (X, y)
        """
        print("Starting data processing pipeline...")
        
        # Step 1: Preprocess the data
        print("Preprocessing data...")
        processed_df = self.preprocess_data(df)
        
        # Step 2: Balance the dataset through downsampling
        print("Balancing dataset...")
        balanced_df = self.downsample(processed_df)
        
        # Step 3: Apply PCA if enabled
        print("Applying dimensionality reduction...")
        df_pca = self.apply_pca(balanced_df)
        
        # Step 4: Discretize continuous features
        print("Discretizing features...")
        df_discretized = self.discretize_data(df_pca)
        
        # Step 5: Split into features and target
        X = df_discretized.drop(columns=[self.target_column])
        y = df_discretized[self.target_column]
        
        if return_test_data:
            # Split into train and test sets
            return train_test_split(
                X, y,
                test_size=self.data_config["test_size"],
                random_state=self.data_config["random_state"]
            )
        else:
            return X, y
            
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted transformers.
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.preprocessed:
            raise ValueError("Must call preprocess_data first before transforming new data.")
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Get overlapping columns with training data
        num_cols = [col for col in self.numerical_columns if col in df.columns]
        cat_cols = [col for col in self.categorical_columns if col in df.columns]
        
        # Transform numerical features
        if num_cols and self.num_scaler:
            processed_df[num_cols] = self.num_scaler.transform(processed_df[num_cols])
        
        # Transform categorical features
        if cat_cols and self.cat_encoder:
            encoded_cats = self.cat_encoder.transform(processed_df[cat_cols])
            
            # Create DataFrame with encoded categorical features
            encoded_df = pd.DataFrame(
                encoded_cats,
                columns=self.cat_encoder.get_feature_names_out(cat_cols),
                index=processed_df.index
            )
            
            # Drop original categorical columns and add encoded ones
            processed_df = processed_df.drop(columns=cat_cols)
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
        
        # Apply PCA if it was used during training
        if self.data_config["pca"]["enabled"] and self.pca_transformer:
            # Get features (all columns except target if present)
            features = processed_df.drop(columns=[self.target_column]) if self.target_column in processed_df.columns else processed_df
            pca_result = self.pca_transformer.transform(features)
            
            # Create PCA DataFrame
            pca_df = pd.DataFrame(
                pca_result,
                columns=[f'PCA_{i+1}' for i in range(self.data_config["pca"]["n_components"])],
                index=processed_df.index
            )
            
            # Add target column if present
            if self.target_column in processed_df.columns:
                pca_df[self.target_column] = processed_df[self.target_column]
            
            processed_df = pca_df
        
        # Discretize if enabled
        if self.data_config["discretization"]["enabled"]:
            discretized = processed_df.copy()
            bins = self.data_config["discretization"]["bins"]
            labels = self.data_config["discretization"]["labels"]
            
            # Get columns to discretize (all except target if present)
            columns_to_discretize = [col for col in processed_df.columns if col != self.target_column or col not in processed_df.columns]
            
            # Apply discretization
            for col in columns_to_discretize:
                discretized[col] = pd.qcut(
                    pd.to_numeric(discretized[col], errors='coerce').fillna(discretized[col].median()),
                    q=bins,
                    labels=labels,
                    duplicates='drop'
                ).astype(str)
            
            processed_df = discretized
        
        return processed_df
    
    def get_pca_stats(self) -> Dict:
        """
        Get PCA statistics.
        
        Returns:
            Dictionary with PCA statistics
        """
        if not self.data_config["pca"]["enabled"] or not hasattr(self, 'pca_transformer'):
            return None
            
        return {
            'explained_variance_ratio': self.explained_variance_ratio,
            'n_components': self.data_config["pca"]["n_components"],
            'total_variance_explained': np.sum(self.explained_variance_ratio)
        }


# Example usage functions

def process_dataframe(df: pd.DataFrame, config_path: str = "config.json") -> Tuple:
    """
    Process a given DataFrame using configurations from config file.
    
    Args:
        df: Input DataFrame
        config_path: Path to the configuration file
        
    Returns:
        Processed data split into train/test sets
    """
    processor = DynamicDataProcessor(config_path)
    return processor.process_pipeline(df)

def process_new_data(df: pd.DataFrame, processor: DynamicDataProcessor) -> pd.DataFrame:
    """
    Process new data using an existing processor.
    
    Args:
        df: New data to process
        processor: Fitted DynamicDataProcessor
        
    Returns:
        Processed DataFrame
    """
    return processor.transform_new_data(df) 