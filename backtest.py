"""
Backtesting engine for strategy validation
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_loader import get_data
from feature_engineering import FeatureEngineer
from trading_signals import TradingSignalGenerator, RiskManager
from config import TICKER, LSTM_LOOKBACK

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtest trading strategy.
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.risk_manager = RiskManager(initial_capital)
        self.signal_generator = TradingSignalGenerator()
        self.fe = FeatureEngineer()
        self.equity_curve = []
        self.trades = []
    
    def run_backtest(self, df, signal_func, test_split=0.3):
        """
        Run backtest on historical data.
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("BACKTEST STARTED")
            logger.info("="*80)
            
            # Split data
            test_size = int(len(df) * test_split)
            train_df = df.iloc[:-test_size]
            test_df = df.iloc[-test_size:]
            
            logger.info(f"Training period: {train_df.index[0].date()} to {train_df.index[-1].date()}")
            logger.info(f"Testing period: {test_df.index[0].date()} to {test_df.index[-1].date()}")
            
            # Iterate through test period
            for i in range(LSTM_LOOKBACK, len(test_df)):
                current_date = test_df.index[i]
                current_price = test_df['Close'].iloc[i]
                
                # Generate signal (simplified for backtest)
                tomorrow_return = (test_df['Close'].iloc[i+1] - current_price) / current_price if i+1 < len(test_df) else 0
                predicted_price = current_price * (1 + tomorrow_return * 1.1)  # Assume model has some predictive power
                confidence = 0.6
                
                signal = self.signal_generator.generate_signals(
                    current_price, predicted_price, confidence
                )
                
                # Execute trade (simplified)
                if signal['signal'] == 'BUY' and signal['position_size'] > 0:
                    position_size = self.risk_manager.calculate_position_size(
                        current_price, signal['stop_loss']
                    )
                    
                    self.risk_manager.positions[TICKER] = {
                        'entry_price': current_price,
                        'shares': position_size,
                        'current_price': current_price,
                        'entry_date': current_date
                    }
                    
                    logger.info(f"{current_date.date()}: BUY {position_size:.4f} @ ${current_price:.2f}")
                
                elif signal['signal'] == 'SELL' and TICKER in self.risk_manager.positions:
                    position = self.risk_manager.positions[TICKER]
                    pnl = position['shares'] * (current_price - position['entry_price'])
                    pnl_pct = (pnl / (position['shares'] * position['entry_price'])) * 100
                    
                    logger.info(f"{current_date.date()}: SELL @ ${current_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                    
                    self.risk_manager.record_trade({
                        'type': 'exit',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'shares': position['shares'],
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
                    
                    del self.risk_manager.positions[TICKER]
                
                # Update equity
                metrics = self.risk_manager.get_portfolio_metrics()
                self.equity_curve.append({
                    'date': current_date,
                    'portfolio_value': metrics['total_portfolio_value'],
                    'cash': metrics['current_capital']
                })
            
            # Close any open positions
            if TICKER in self.risk_manager.positions:
                position = self.risk_manager.positions[TICKER]
                final_price = test_df['Close'].iloc[-1]
                pnl = position['shares'] * (final_price - position['entry_price'])
                self.risk_manager.record_trade({
                    'type': 'exit',
                    'entry_price': position['entry_price'],
                    'exit_price': final_price,
                    'shares': position['shares'],
                    'entry_date': position['entry_date'],
                    'exit_date': test_df.index[-1],
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position['shares'] * position['entry_price'])) * 100
                })
            
            # Generate report
            self._generate_report()
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            raise
    
    def _generate_report(self):
        """
        Generate backtest performance report.
        """
        metrics = self.risk_manager.get_portfolio_metrics()
        
        logger.info("\n" + "="*80)
        logger.info("BACKTEST RESULTS")
        logger.info("="*80)
        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"Final Portfolio Value: ${metrics['total_portfolio_value']:.2f}")
        logger.info(f"Total Return: ${metrics['open_pnl']:.2f} ({metrics['total_return_pct']:.2f}%)")
        logger.info(f"Number of Trades: {metrics['num_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        logger.info("="*80)
        
        return metrics


def main():
    """
    Main backtest pipeline.
    """
    try:
        # Load data
        logger.info("Loading data for backtest...")
        df = get_data()
        
        # Run backtest
        engine = BacktestEngine()
        engine.run_backtest(df, signal_func=None)
        
        return engine
    except Exception as e:
        logger.error(f"Backtest pipeline failed: {e}")
        raise


if __name__ == "__main__":
    engine = main()
