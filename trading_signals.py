"""
Generate trading signals and recommendations
"""
import numpy as np
import pandas as pd
import logging
from config import (
    MAX_POSITION_SIZE, STOP_LOSS_PCT, TAKE_PROFIT_PCT
)

logger = logging.getLogger(__name__)


class TradingSignalGenerator:
    """
    Generate buy/sell signals based on model predictions.
    """
    
    def __init__(self, prediction_model='ensemble'):
        self.prediction_model = prediction_model
        self.signals = {}
    
    def generate_signals(self, current_price, predicted_price, confidence=None):
        """
        Generate trading signals based on predictions.
        """
        try:
            price_change_pct = (predicted_price - current_price) / current_price * 100
            
            signal = {
                'timestamp': pd.Timestamp.now(),
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change_pct,
                'confidence': confidence or 0.5,
                'signal': 'HOLD',
                'strength': 0,
                'position_size': 0,
                'stop_loss': current_price * (1 - STOP_LOSS_PCT),
                'take_profit': current_price * (1 + TAKE_PROFIT_PCT),
            }
            
            # Strong buy signal
            if price_change_pct > 3 and confidence > 0.7:
                signal['signal'] = 'BUY'
                signal['strength'] = min(confidence * abs(price_change_pct) / 10, 1.0)
                signal['position_size'] = signal['strength'] * MAX_POSITION_SIZE
            
            # Buy signal
            elif price_change_pct > 1.5 and confidence > 0.6:
                signal['signal'] = 'BUY'
                signal['strength'] = confidence * 0.7
                signal['position_size'] = signal['strength'] * MAX_POSITION_SIZE * 0.7
            
            # Strong sell signal
            elif price_change_pct < -3 and confidence > 0.7:
                signal['signal'] = 'SELL'
                signal['strength'] = min(confidence * abs(price_change_pct) / 10, 1.0)
                signal['position_size'] = signal['strength'] * MAX_POSITION_SIZE
            
            # Sell signal
            elif price_change_pct < -1.5 and confidence > 0.6:
                signal['signal'] = 'SELL'
                signal['strength'] = confidence * 0.7
                signal['position_size'] = signal['strength'] * MAX_POSITION_SIZE * 0.7
            
            logger.info(f"Signal generated: {signal['signal']} | "
                       f"Strength: {signal['strength']:.2f} | "
                       f"Position: {signal['position_size']:.4f}")
            
            return signal
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            raise
    
    def analyze_political_impact(self, political_risk_score, political_tone):
        """
        Analyze impact of political events on trading signals.
        """
        try:
            impact = {
                'risk_level': 'LOW',
                'volatility_adjustment': 1.0,
                'position_size_adjustment': 1.0,
                'confidence_adjustment': 1.0,
            }
            
            # High political risk
            if political_risk_score > 5:
                impact['risk_level'] = 'HIGH'
                impact['volatility_adjustment'] = 1.5
                impact['position_size_adjustment'] = 0.7  # Reduce position size
                impact['confidence_adjustment'] = 0.8  # Reduce confidence
            
            # Medium political risk
            elif political_risk_score > 2:
                impact['risk_level'] = 'MEDIUM'
                impact['volatility_adjustment'] = 1.2
                impact['position_size_adjustment'] = 0.85
                impact['confidence_adjustment'] = 0.9
            
            # Negative sentiment
            if political_tone < -3:
                impact['confidence_adjustment'] *= 0.9
                impact['position_size_adjustment'] *= 0.9
            
            logger.info(f"Political Impact Analysis: Risk={impact['risk_level']}, "
                       f"Volatility_Adj={impact['volatility_adjustment']:.2f}")
            
            return impact
        except Exception as e:
            logger.error(f"Error analyzing political impact: {e}")
            raise
    
    def adjust_for_portfolio_risk(self, signal, portfolio_exposure):
        """
        Adjust signal based on current portfolio risk exposure.
        """
        try:
            adjusted_signal = signal.copy()
            
            # Reduce position size if portfolio is already exposed
            if portfolio_exposure > 0.3:  # >30% of portfolio
                adjusted_signal['position_size'] *= (1 - portfolio_exposure)
                adjusted_signal['signal'] = 'HOLD' if adjusted_signal['signal'] == 'BUY' else adjusted_signal['signal']
            
            logger.info(f"Adjusted signal for portfolio exposure: {adjusted_signal['signal']} | "
                       f"New position size: {adjusted_signal['position_size']:.4f}")
            
            return adjusted_signal
        except Exception as e:
            logger.error(f"Error adjusting for portfolio risk: {e}")
            raise


class RiskManager:
    """
    Manage trading risk and position sizing.
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
    
    def calculate_position_size(self, entry_price, stop_loss, account_risk_pct=0.02):
        """
        Calculate position size based on risk-reward.
        """
        try:
            risk_per_trade = self.current_capital * account_risk_pct
            price_risk = abs(entry_price - stop_loss)
            
            if price_risk == 0:
                position_size = 0
            else:
                position_size = risk_per_trade / price_risk
            
            # Cap position size
            max_position = self.current_capital * MAX_POSITION_SIZE / entry_price
            position_size = min(position_size, max_position)
            
            logger.info(f"Position size calculated: {position_size:.4f} shares (Risk: {risk_per_trade:.2f})")
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            raise
    
    def record_trade(self, trade_info):
        """
        Record completed trade.
        """
        self.trade_history.append(trade_info)
        
        # Update capital
        if trade_info['type'] == 'exit':
            pnl = trade_info['shares'] * (trade_info['exit_price'] - trade_info['entry_price'])
            self.current_capital += pnl
            logger.info(f"Trade closed. P&L: {pnl:.2f}, Updated capital: {self.current_capital:.2f}")
    
    def get_portfolio_metrics(self):
        """
        Calculate current portfolio metrics.
        """
        total_open_pnl = 0
        total_position_value = 0
        
        for symbol, position in self.positions.items():
            position_value = position['shares'] * position['current_price']
            pnl = position['shares'] * (position['current_price'] - position['entry_price'])
            total_open_pnl += pnl
            total_position_value += position_value
        
        metrics = {
            'current_capital': self.current_capital,
            'total_portfolio_value': self.current_capital + total_open_pnl,
            'open_pnl': total_open_pnl,
            'position_value': total_position_value,
            'total_return_pct': (total_open_pnl / self.initial_capital) * 100,
            'num_trades': len(self.trade_history),
            'win_rate': self._calculate_win_rate(),
        }
        
        return metrics
    
    def _calculate_win_rate(self):
        """
        Calculate win rate from trade history.
        """
        closed_trades = [t for t in self.trade_history if t['type'] == 'exit']
        if not closed_trades:
            return 0
        
        wins = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
        return (wins / len(closed_trades)) * 100 if closed_trades else 0
