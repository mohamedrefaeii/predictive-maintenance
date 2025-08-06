"""
Model training module for predictive maintenance system.
Implements Random Forest, XGBoost, and LSTM models for failure prediction and RUL estimation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Handles training of different ML models for predictive maintenance."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           task_type: str = 'regression') -> dict[str, any]:
        """Train Random Forest model."""
        
        if task_type == 'regression':
            model = RandomForestRegressor(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task_type == 'regression':
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        else:
            from sklearn.metrics import accuracy_score
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred)
            }
        
        self.models['random_forest'] = model
        self.results['random_forest'] = metrics
        
        return metrics
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     task_type: str = 'regression') -> dict[str, any]:
        """Train XGBoost model."""
        
        if task_type == 'regression':
            model = xgb.XGBRegressor(random_state=42)
        else:
            model = xgb.XGBClassifier(random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task_type == 'regression':
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        else:
            from sklearn.metrics import accuracy_score
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred)
            }
        
        self.models['xgboost'] = model
        self.results['xgboost'] = metrics
        
        return metrics
    
    def build_lstm_model(self, input_shape: tuple) -> tf.keras.Model:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 50, batch_size: int = 32) -> Dict[str, any]:
        """Train LSTM model."""
        
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        y_pred = model.predict(X_val).flatten()
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred)
        }
        
        self.models['lstm'] = model
        self.results['lstm'] = metrics
        
        return metrics
    
    def get_best_model(self) -> tuple:
        """Get the best performing model based on validation metrics."""
        if not self.results:
            return None, None
        
        best_model_name = max(self.results.items(), key=lambda x: x[1]['r2'] if 'r2' in x[1] else x[1]['accuracy'])[0]
        return best_model_name, self.models[best_model_name]
    
    def save_all_models(self):
        """Save all trained models."""
        import os
        os.makedirs('models', exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'lstm':
                model.save('models/lstm_model.h5')
            else:
                joblib.dump(model, f'models/{name}_model.joblib')
