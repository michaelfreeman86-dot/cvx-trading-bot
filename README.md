# CVX Political Trading Bot

A sophisticated algorithmic trading bot for Chevron (CVX) stock that integrates political event analysis, GDELT news sentiment, and machine learning models to generate actionable trading signals.

## Features

### Data Sources
- **Yahoo Finance**: Real-time and historical price data
- **GDELT**: Global Event, Language, and Tone database for news sentiment analysis
- **Political Events**: Curated database of major political announcements affecting energy sector

### Technical Analysis
- SMA, EMA, MACD, RSI, Bollinger Bands
- ATR, Stochastic, CCI, ADX
- Candlestick patterns (Hammer, Doji, Engulfing, Morning Star)
- Volume indicators (OBV, Accumulation/Distribution)

### Machine Learning Models
- **LSTM Neural Network**: Deep learning time series prediction
- **XGBoost**: Gradient boosting for tabular data
- **Ensemble Approach**: Combines both models for robust predictions

### Political Event Integration
- Event flags for major announcements (Anti-Weaponization Fund, Tariffs, etc.)
- Exponential decay functions to model event impact
- Political tone and news volume analysis
- Risk scoring system

### Risk Management
- Position sizing based on Kelly Criterion
- Stop-loss and take-profit levels
- Portfolio exposure limits
- Political risk adjustments

## Installation

```bash
# Clone repository
git clone https://github.com/michaelfreeman86-dot/cvx-trading-bot.git
cd cvx-trading-bot

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train Models

Train the LSTM and XGBoost models on historical data:

```bash
python train.py
```

This will:
- Download 5 years of CVX price data
- Create technical and political features
- Train LSTM and XGBoost models
- Save models to `models/` directory
- Generate performance metrics

### 2. Make Predictions

Generate trading signals for the next trading day:

```bash
python predict.py
```

Output includes:
- Current price
- LSTM and XGBoost predictions
- Ensemble prediction
- Trading signal (BUY/SELL/HOLD)
- Confidence level
- Position size recommendation
- Political risk analysis

### 3. Backtest Strategy

Validate the strategy on historical data:

```bash
python backtest.py
```

Generates performance metrics:
- Total return
- Number of trades
- Win rate
- Equity curve

## Configuration

Edit `config.py` to customize:

```python
# Trading Parameters
TICKER = 'CVX'           # Stock ticker
PERIOD = '5y'            # Historical period

# Model Parameters
LSTM_UNITS = 64          # LSTM layer size
LSTM_EPOCHS = 50         # Training epochs
LSTM_LOOKBACK = 60       # Number of past days to use

# Political Events
POLITICAL_EVENTS = {
    'Anti_Weaponization_Fund': '2026-05-18',
    'Trump_Tariff_Announcement': '2025-01-20',
    # ... add more events
}

# Risk Management
MAX_POSITION_SIZE = 0.1      # 10% of portfolio
STOP_LOSS_PCT = 0.05         # 5% stop loss
TAKE_PROFIT_PCT = 0.15       # 15% take profit
```

## Project Structure

```
cvx-trading-bot/
├── config.py                # Configuration settings
├── data_loader.py           # Data fetching and preprocessing
├── feature_engineering.py   # Feature creation and scaling
├── models.py                # LSTM and XGBoost models
├── train.py                 # Model training pipeline
├── predict.py               # Prediction and signal generation
├── trading_signals.py       # Signal generation and risk management
├── backtest.py              # Backtesting engine
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── models/                  # Trained model files
│   ├── lstm_cvx_model.h5
│   └── xgb_cvx_model.pkl
└── logs/                    # Log files
    ├── training.log
    ├── prediction.log
    └── backtest.log
```

## Key Innovations

1. **Political Risk Modeling**: Integrates major political events that impact energy stocks
2. **Multi-Source Sentiment Analysis**: Combines GDELT news tone with specific event dates
3. **Ensemble Predictions**: Leverages both neural networks and tree-based models
4. **Intelligent Position Sizing**: Adjusts for political risk and market conditions
5. **Real-time Signals**: Generates actionable buy/sell signals with confidence levels

## Trading Signal Explanation

### Signal Strength
- **BUY**: Model predicts price increase > 1.5% with confidence > 60%
- **SELL**: Model predicts price decrease > 1.5% with confidence > 60%
- **HOLD**: Insufficient signal strength or conflicting indicators

### Confidence Levels
- High confidence (>0.7): Strong agreement between LSTM and XGBoost
- Medium confidence (0.5-0.7): Moderate agreement
- Low confidence (<0.5): Divergent predictions

### Political Adjustments
- HIGH risk: Position size reduced by 30%, confidence reduced by 20%
- MEDIUM risk: Position size reduced by 15%, confidence reduced by 10%
- LOW risk: No adjustment

## Performance Metrics

The bot evaluates predictions using:
- **MSE**: Mean squared error
- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error
- **R²**: Coefficient of determination
- **MAPE**: Mean absolute percentage error

## Risk Disclaimer

⚠️ **WARNING**: This trading bot is for educational purposes only. Past performance does not guarantee future results. Always:

1. **Backtest thoroughly** before live trading
2. **Start with small position sizes**
3. **Monitor closely** during market events
4. **Use stop losses** to limit downside
5. **Never risk capital** you cannot afford to lose
6. **Consult with a financial advisor** before trading

## Future Enhancements

- [ ] Real-time portfolio tracking dashboard
- [ ] Integration with brokerage APIs (Alpaca, Interactive Brokers)
- [ ] Live GDELT API integration (replace simulated data)
- [ ] Multi-asset portfolio optimization
- [ ] Options strategies integration
- [ ] Machine learning hyperparameter optimization
- [ ] Reinforcement learning for adaptive strategies

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

MIT License - See LICENSE file for details

## Contact

For questions or feedback: michaelfreeman86-dot@github.com

## Acknowledgments

- TA-Lib for technical analysis
- TensorFlow/Keras for LSTM implementation
- XGBoost team for gradient boosting
- Yahoo Finance and GDELT for data

---

**Last Updated**: June 8, 2026
**Version**: 1.0.0
