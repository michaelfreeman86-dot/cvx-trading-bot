"""
Configuration settings for CVX Trading Bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Trading Parameters
TICKER = 'CVX'
PERIOD = '5y'
INTERVAL = '1d'

# Model Parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# LSTM Parameters
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_LOOKBACK = 60

# XGBoost Parameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'random_state': RANDOM_STATE,
    'verbose': 0
}

# Political Events
POLITICAL_EVENTS = {
    'Anti_Weaponization_Fund': '2026-05-18',
    'Trump_Tariff_Announcement': '2025-01-20',
    'Energy_Policy_Shift': '2025-02-15',
}

# GDELT Parameters
GDELT_KEYWORDS = [
    'Trump weaponization fund',
    'oil profiteering',
    'energy policy',
    'Middle East crisis',
    'Hormuz strait',
    'sanctions oil',
]

# Risk Management
MAX_POSITION_SIZE = 0.1  # 10% of portfolio
STOP_LOSS_PCT = 0.05  # 5%
TAKE_PROFIT_PCT = 0.15  # 15%

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_bot.log'
