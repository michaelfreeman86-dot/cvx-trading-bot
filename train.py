"""
Main training script for CVX trading bot
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import get_data
from feature_engineering import FeatureEngineer
from models import (
    LSTMModel, XGBoostModel, DataSequenceGenerator, evaluate_model
)
from config import TEST_SIZE, RANDOM_STATE, LSTM_LOOKBACK
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_data_for_lstm(X, y, test_size=TEST_SIZE, lookback=LSTM_LOOKBACK):
    """
    Prepare data for LSTM model.
    """
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, shuffle=False
    )
    
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, shuffle=False
    )
    
    # Create sequences
    seq_gen = DataSequenceGenerator(lookback=lookback)
    X_train_seq, y_train_seq = seq_gen.create_sequences(np.hstack([X_train, y_train.reshape(-1, 1)]), target_col_idx=-1)
    X_val_seq, y_val_seq = seq_gen.create_sequences(np.hstack([X_val, y_val.reshape(-1, 1)]), target_col_idx=-1)
    X_test_seq, y_test_seq = seq_gen.create_sequences(np.hstack([X_test, y_test.reshape(-1, 1)]), target_col_idx=-1)
    
    return X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq


def main():
    """
    Main training pipeline.
    """
    try:
        logger.info("="*80)
        logger.info("CVX POLITICAL TRADING BOT - TRAINING PIPELINE")
        logger.info("="*80)
        
        # 1. Load data
        logger.info("\n[STEP 1] Loading data...")
        df = get_data()
        logger.info(f"Loaded {len(df)} rows of data")
        
        # 2. Feature engineering
        logger.info("\n[STEP 2] Feature Engineering...")
        fe = FeatureEngineer()
        
        # Create lagged features
        df = fe.create_lagged_features(df)
        
        # Create rolling features
        df = fe.create_rolling_features(df)
        
        # Select features (remove low variance)
        df = fe.select_features(df, threshold=0.001)
        
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Data shape: {df.shape}")
        
        # 3. Prepare data
        logger.info("\n[STEP 3] Preparing data...")
        
        # Separate features and target
        y = df['Close'].values
        X = df.drop('Close', axis=1).values
        
        # Normalize data
        X_normalized = fe.normalize_features(df.drop('Close', axis=1))
        X_normalized = X_normalized.values
        
        logger.info(f"Feature matrix shape: {X_normalized.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # 4. Train LSTM Model
        logger.info("\n[STEP 4] Training LSTM Model...")
        X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = prepare_data_for_lstm(
            X_normalized, y
        )
        
        logger.info(f"LSTM sequences shape - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
        
        lstm_model = LSTMModel(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
        lstm_history = lstm_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
        
        # Evaluate LSTM
        y_pred_lstm = lstm_model.predict(X_test_seq)
        lstm_metrics = evaluate_model(y_test_seq, y_pred_lstm, "LSTM")
        
        # Save LSTM model
        lstm_model.save('models/lstm_cvx_model.h5')
        
        # 5. Train XGBoost Model
        logger.info("\n[STEP 5] Training XGBoost Model...")
        
        X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
        )
        
        X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
            X_train_xgb, y_train_xgb, test_size=0.15, random_state=RANDOM_STATE, shuffle=False
        )
        
        xgb_model = XGBoostModel()
        xgb_model.train(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb)
        
        # Evaluate XGBoost
        y_pred_xgb = xgb_model.predict(X_test_xgb)
        xgb_metrics = evaluate_model(y_test_xgb, y_pred_xgb, "XGBoost")
        
        # Feature importance
        logger.info("\nXGBoost Top 20 Important Features:")
        importance = xgb_model.get_feature_importance(top_n=20)
        for i, (feature, score) in enumerate(importance, 1):
            logger.info(f"  {i:2d}. {feature:40s} : {score}")
        
        # Save XGBoost model
        xgb_model.save('models/xgb_cvx_model.pkl')
        
        # 6. Model Comparison
        logger.info("\n[STEP 6] Model Comparison...")
        logger.info(f"\nLSTM R² Score:    {lstm_metrics['R2']:.6f}")
        logger.info(f"XGBoost R² Score: {xgb_metrics['R2']:.6f}")
        
        if lstm_metrics['R2'] > xgb_metrics['R2']:
            logger.info("✓ LSTM performs better")
            best_model = "LSTM"
        else:
            logger.info("✓ XGBoost performs better")
            best_model = "XGBoost"
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Best Model: {best_model}")
        logger.info("="*80)
        
        return lstm_model, xgb_model, lstm_metrics, xgb_metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    lstm_model, xgb_model, lstm_metrics, xgb_metrics = main()
