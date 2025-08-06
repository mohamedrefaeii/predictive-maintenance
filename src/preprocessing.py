"""
Data preprocessing module for predictive maintenance system.
Handles data cleaning, feature engineering, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles all data preprocessing tasks for predictive maintenance."""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if strategy == 'median':
            return df.fillna(df.median())
        elif strategy == 'mean':
            return df.fillna(df.mean())
        elif strategy == 'forward_fill':
            return df.fillna(method='ffill')
        else:
            return df.fillna(0)
    
    def detect_outliers(self, df: pd.DataFrame, 
                       columns: List[str], 
                       method: str = 'iqr') -> pd.DataFrame:
        """Detect and handle outliers using IQR method."""
        df_clean = df.copy()
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    def create_time_features(self, df: pd.DataFrame, 
                           time_col: str = 'cycle') -> pd.DataFrame:
        """Create time-series features from sensor data."""
        df_features = df.copy()
        
        # Rolling statistics
        sensor_cols = [col for col in df.columns if col not in ['engine_id', 'RUL', time_col]]
        
        for col in sensor_cols:
            # Rolling mean and std
            df_features[f'{col}_mean_3'] = df.groupby('engine_id')[col].rolling(window=3).mean().reset_index(0, drop=True)
            df_features[f'{col}_std_3'] = df.groupby('engine_id')[col].rolling(window=3).std().reset_index(0, drop=True)
            df_features[f'{col}_mean_5'] = df.groupby('engine_id')[col].rolling(window=5).mean().reset_index(0, drop=True)
            df_features[f'{col}_std_5'] = df.groupby('engine_id')[col].rolling(window=5).std().reset_index(0, drop=True)
            
            # Lag features
            df_features[f'{col}_lag_1'] = df.groupby('engine_id')[col].shift(1)
            df_features[f'{col}_lag_2'] = df.groupby('engine_id')[col].shift(2)
        
        # Drop NaN values created by rolling features
        df_features = df_features.dropna()
        
        return df_features
    
    def normalize_features(self, X: pd.DataFrame, 
                          fit: bool = True) -> pd.DataFrame:
        """Normalize features using RobustScaler."""
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled
    
    def prepare_sequences(self, df: pd.DataFrame, 
                         sequence_length: int = 30,
                         target_col: str = 'RUL') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM model."""
        sequences = []
        targets = []
        
        for engine_id in df['engine_id'].unique():
            engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')
            
            if len(engine_data) >= sequence_length:
                for i in range(len(engine_data) - sequence_length + 1):
                    seq = engine_data.iloc[i:i+sequence_length].drop(['engine_id', 'RUL'], axis=1).values
                    target = engine_data.iloc[i+sequence_length-1][target_col]
                    
                    sequences.append(seq)
                    targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def split_data(self, df: pd.DataFrame, 
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data maintaining engine integrity."""
        np.random.seed(random_state)
        
        engines = df['engine_id'].unique()
        np.random.shuffle(engines)
        
        split_idx = int(len(engines) * (1 - test_size))
        train_engines = engines[:split_idx]
        test_engines = engines[split_idx:]
        
        train_df = df[df['engine_id'].isin(train_engines)]
        test_df = df[df['engine_id'].isin(test_engines)]
        
        return train_df, test_df
    
    def preprocess_pipeline(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Complete preprocessing pipeline."""
        # Load data
        df = self.load_data(filepath)
        if df.empty:
            return None, None, None, None
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Detect and handle outliers
        sensor_cols = [col for col in df.columns if col not in ['engine_id', 'RUL']]
        df = self.detect_outliers(df, sensor_cols)
        
        # Create time features
        df = self.create_time_features(df)
        
        # Split data
        train_df, test_df = self.split_data(df)
        
        # Prepare features and targets
        feature_cols = [col for col in train_df.columns if col not in ['engine_id', 'RUL']]
        
        X_train = train_df[feature_cols]
        y_train = train_df['RUL']
        X_test = test_df[feature_cols]
        y_test = test_df['RUL']
        
        # Normalize features
        X_train = self.normalize_features(X_train, fit=True)
        X_test = self.normalize_features(X_test, fit=False)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # For demonstration, create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'engine_id': np.repeat([1, 2, 3], 50),
        'cycle': list(range(50)) * 3,
        'T2': np.random.normal(518, 10, 150),
        'T24': np.random.normal(642, 15, 150),
        'T30': np.random.normal(1589, 50, 150),
        'T50': np.random.normal(1400, 40, 150),
        'RUL': np.random.randint(100, 200, 150)
    })
    
    # Save sample data
    sample_data.to_csv('data/sample_data.csv', index=False)
    
    # Run preprocessing
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline('data/sample_data.csv')
    print("Preprocessing completed successfully!")
