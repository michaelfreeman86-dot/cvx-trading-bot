"""
Make predictions and generate trading signals
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_loader import get_data
from feature_engineering import FeatureEngineer
from models import LSTMModel, XGBoostModel
from trading_signals import TradingSignalGenerator, RiskManager
from config import TICKER, LSTM_LOOKBACK

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Make predictions and generate actionable trading signals.
    """
    
    def __init__(self, lstm_model_path='models/lstm_cvx_model.h5',
                 xgb_model_path='models/xgb_cvx_model.pkl'):
        self.lstm_model = LSTMModel(input_shape=(LSTM_LOOKBACK, 30))  # Adjust as needed
        self.lstm_model.load(lstm_model_path)
        
        self.xgb_model = XGBoostModel()
        self.xgb_model.load(xgb_model_path)
        
        self.signal_generator = TradingSignalGenerator()
        self.risk_manager = RiskManager()
        self.fe = FeatureEngineer()
    
    def predict_next_day(self):
        """
        Make predictions for the next trading day.
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("GENERATING TRADING PREDICTIONS")
            logger.info("="*80)
            
            # Load latest data
            logger.info("\n[STEP 1] Loading latest data...")
            df = get_data()
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            logger.info(f"Current price: ${current_price:.2f}")
            
            # Prepare features
            logger.info("\n[STEP 2] Preparing features...")
            df_processed = self.fe.create_lagged_features(df)
            df_processed = self.fe.create_rolling_features(df_processed)
            
            # Get latest political data
            political_risk = df_processed['Political_Risk_Score'].iloc[-1] if 'Political_Risk_Score' in df_processed else 0
            political_tone = df_processed['Political_Tone'].iloc[-1] if 'Political_Tone' in df_processed else 0
            
            logger.info(f"Political Risk Score: {political_risk:.2f}")
            logger.info(f"Political Tone: {political_tone:.2f}")
            
            # Normalize features
            df_normalized = self.fe.normalize_features(df_processed.drop('Close', axis=1), fit=False)
            
            # LSTM Prediction
            logger.info("\n[STEP 3] Making LSTM prediction...")
            X_lstm = df_normalized.values[-LSTM_LOOKBACK:].reshape(1, LSTM_LOOKBACK, df_normalized.shape[1])
            lstm_pred = self.lstm_model.predict(X_lstm)[0]
            logger.info(f"LSTM predicted price: ${lstm_pred:.2f}")
            
            # XGBoost Prediction
            logger.info("\n[STEP 4] Making XGBoost prediction...")
            X_xgb = df_processed.drop('Close', axis=1).iloc[-1:]
            xgb_pred = self.xgb_model.predict(X_xgb)[0]
            logger.info(f"XGBoost predicted price: ${xgb_pred:.2f}")
            
            # Ensemble prediction (average)
            ensemble_pred = (lstm_pred + xgb_pred) / 2
            logger.info(f"\nEnsemble predicted price: ${ensemble_pred:.2f}")
            
            # Calculate confidence (agreement between models)
            pred_diff = abs(lstm_pred - xgb_pred) / current_price
            confidence = max(0, 1 - pred_diff)  # Higher agreement = higher confidence
            logger.info(f"Model agreement confidence: {confidence:.2f}")
            
            # Generate trading signals
            logger.info("\n[STEP 5] Generating trading signals...")
            base_signal = self.signal_generator.generate_signals(
                current_price, ensemble_pred, confidence
            )
            
            # Adjust for political impact
            logger.info("\n[STEP 6] Analyzing political impact...")
            political_impact = self.signal_generator.analyze_political_impact(
                political_risk, political_tone
            )
            
            # Apply adjustments
            base_signal['position_size'] *= political_impact['position_size_adjustment']
            base_signal['confidence'] *= political_impact['confidence_adjustment']
            
            # Final recommendation
            logger.info("\n" + "="*80)
            logger.info("FINAL RECOMMENDATION")
            logger.info("="*80)
            logger.info(f"Signal: {base_signal['signal']}")
            logger.info(f"Current Price: ${current_price:.2f}")
            logger.info(f"Target Price: ${ensemble_pred:.2f}")
            logger.info(f"Predicted Change: {base_signal['price_change_pct']:.2f}%")
            logger.info(f"Confidence: {base_signal['confidence']:.2f}")
            logger.info(f"Signal Strength: {base_signal['strength']:.2f}")
            logger.info(f"Recommended Position Size: {base_signal['position_size']:.4f}")
            logger.info(f"Stop Loss: ${base_signal['stop_loss']:.2f}")
            logger.info(f"Take Profit: ${base_signal['take_profit']:.2f}")
            logger.info(f"Political Risk Level: {political_impact['risk_level']}")
            logger.info("="*80)
            
            return {
                'current_price': current_price,
                'lstm_prediction': lstm_pred,
                'xgb_prediction': xgb_pred,
                'ensemble_prediction': ensemble_pred,
                'signal': base_signal,
                'political_impact': political_impact,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise


def main():
    """
    Main prediction pipeline.
    """
    try:
        engine = PredictionEngine()
        results = engine.predict_next_day()
        return results
    except Exception as e:
        logger.error(f"Prediction engine failed: {e}")
        raise


if __name__ == "__main__":
    results = main()
