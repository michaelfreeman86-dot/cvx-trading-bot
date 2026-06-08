"""
Machine learning models module - LSTM and XGBoost
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging
from config import (
    LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LOOKBACK,
    XGB_PARAMS, TEST_SIZE, RANDOM_STATE
)

logger = logging.getLogger(__name__)


class DataSequenceGenerator:
    """
    Generate sequences for LSTM training.
    """
    
    def __init__(self, lookback=LSTM_LOOKBACK):
        self.lookback = lookback
    
    def create_sequences(self, data, target_col_idx=0):
        """
        Create sequences for LSTM.
        """
        X, y = [], []
        
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback, target_col_idx])
        
        return np.array(X), np.array(y)


class LSTMModel:
    """
    LSTM neural network for time series prediction.
    """
    
    def __init__(self, input_shape):
        self.model = None
        self.history = None
        self.input_shape = input_shape
        self.build_model()
    
    def build_model(self):
        """
        Build LSTM model architecture.
        """
        try:
            logger.info(f"Building LSTM model with input shape {self.input_shape}...")
            
            self.model = keras.Sequential([
                layers.LSTM(LSTM_UNITS, activation='relu', input_shape=self.input_shape, return_sequences=True),
                layers.Dropout(LSTM_DROPOUT),
                layers.LSTM(LSTM_UNITS // 2, activation='relu', return_sequences=False),
                layers.Dropout(LSTM_DROPOUT),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
            
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            logger.info("LSTM model built successfully")
            logger.info(self.model.summary())
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train LSTM model.
        """
        try:
            logger.info("Training LSTM model...")
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            self.history = self.model.fit(
                X_train, y_train,
                epochs=LSTM_EPOCHS,
                batch_size=LSTM_BATCH_SIZE,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                verbose=1
            )
            
            logger.info("LSTM model training completed")
            return self.history
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise
    
    def predict(self, X_test):
        """
        Make predictions with LSTM model.
        """
        try:
            predictions = self.model.predict(X_test, verbose=0)
            return predictions.flatten()
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def save(self, filepath):
        """
        Save model to file.
        """
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self, filepath):
        """
        Load model from file.
        """
        try:
            self.model = keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


class XGBoostModel:
    """
    XGBoost model for tabular data prediction.
    """
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model.
        """
        try:
            logger.info("Training XGBoost model...")
            
            self.model = xgb.XGBRegressor(
                **XGB_PARAMS,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20
            )
            
            self.model.fit(X_train, y_train, verbose=False)
            logger.info("XGBoost model training completed")
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise
    
    def predict(self, X_test):
        """
        Make predictions with XGBoost model.
        """
        try:
            predictions = self.model.predict(X_test)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance scores.
        """
        try:
            importance = self.model.get_booster().get_score(importance_type='weight')
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            return sorted_importance[:top_n]
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise
    
    def save(self, filepath):
        """
        Save model to file.
        """
        try:
            self.model.save_model(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self, filepath):
        """
        Load model from file.
        """
        try:
            self.model = xgb.XGBRegressor()
            self.model.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    logger.info(f"\n{model_name} Evaluation Metrics:")
    logger.info(f"  MSE:  {mse:.6f}")
    logger.info(f"  MAE:  {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  R²:   {r2:.6f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }
