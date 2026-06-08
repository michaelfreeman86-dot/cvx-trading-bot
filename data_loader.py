"""
Data loading and preprocessing module
"""
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import logging
from config import TICKER, PERIOD, GDELT_KEYWORDS, POLITICAL_EVENTS

logger = logging.getLogger(__name__)


def get_price_data(ticker=TICKER, period=PERIOD):
    """
    Download historical price data from Yahoo Finance.
    """
    try:
        logger.info(f"Downloading {ticker} data for period {period}...")
        df = yf.download(ticker, period=period, progress=False)
        logger.info(f"Downloaded {len(df)} rows of data")
        return df
    except Exception as e:
        logger.error(f"Error downloading price data: {e}")
        raise


def add_technical_features(df):
    """
    Add technical analysis features using TA-Lib.
    """
    try:
        logger.info("Adding technical features...")
        
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Trend Indicators
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['SMA_200'] = talib.SMA(close, timeperiod=200)
        df['EMA_12'] = talib.EMA(close, timeperiod=12)
        df['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # RSI
        df['RSI_14'] = talib.RSI(close, timeperiod=14)
        df['RSI_7'] = talib.RSI(close, timeperiod=7)
        
        # Bollinger Bands
        df['BBands_Upper'], df['BBands_Mid'], df['BBands_Lower'] = talib.BBANDS(close, timeperiod=20)
        df['BBands_Width'] = df['BBands_Upper'] - df['BBands_Lower']
        df['BBands_Position'] = (close - df['BBands_Lower']) / (df['BBands_Upper'] - df['BBands_Lower'])
        
        # ATR (Volatility)
        df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Volume Indicators
        df['OBV'] = talib.OBV(close, volume)
        df['AD'] = talib.AD(high, low, close, volume)
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        
        # CCI
        df['CCI_14'] = talib.CCI(high, low, close, timeperiod=14)
        
        # ADX (Trend Strength)
        df['ADX_14'] = talib.ADX(high, low, close, timeperiod=14)
        
        # Price patterns
        df['CDLHAMMER'] = talib.CDLHAMMER(df['Open'], high, low, close)
        df['CDLDOJI'] = talib.CDLDOJI(df['Open'], high, low, close)
        df['CDLENGULFING'] = talib.CDLENGULFING(df['Open'], high, low, close)
        df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df['Open'], high, low, close)
        
        # Returns
        df['Daily_Return'] = close / np.roll(close, 1) - 1
        df['Log_Return'] = np.log(close / np.roll(close, 1))
        df['Volatility'] = df['Log_Return'].rolling(window=20).std()
        
        logger.info("Technical features added successfully")
        return df
    except Exception as e:
        logger.error(f"Error adding technical features: {e}")
        raise


def get_political_events_features(start_date, end_date):
    """
    Political event flags + sentiment analysis.
    """
    try:
        logger.info("Creating political event features...")
        
        df_events = pd.DataFrame(index=pd.date_range(start_date, end_date))
        
        # Add event flags and decay
        for event_name, event_date in POLITICAL_EVENTS.items():
            event_dt = pd.to_datetime(event_date)
            df_events[event_name] = 0
            
            # Spike window (announcement + impact period)
            window = pd.date_range(event_dt - timedelta(days=3), event_dt + timedelta(days=30))
            mask = df_events.index.isin(window)
            df_events.loc[mask, event_name] = 1
            
            # Exponential decay
            days_diff = (df_events.index - event_dt).days
            decay_vals = np.exp(-0.05 * np.maximum(days_diff, 0))
            df_events[f'{event_name}_decay'] = df_events[event_name] * decay_vals
        
        # Add GDELT sentiment data (simulated for demo)
        df_events['Political_News_Volume'] = simulate_gdelt_volume(start_date, end_date)
        df_events['Political_Tone'] = simulate_gdelt_tone(start_date, end_date)
        
        # Calculate political risk score
        df_events['Political_Risk_Score'] = (
            df_events['Political_News_Volume'] * (df_events['Political_Tone'] < -1).astype(int) +
            df_events.get('Anti_Weaponization_Fund', 0) * 0.5
        )
        
        # Forward fill and fill NA values
        df_events['Political_Risk_Score'] = df_events['Political_Risk_Score'].ffill().fillna(0)
        df_events['Political_Tone'] = df_events['Political_Tone'].ffill().fillna(0)
        df_events['Political_News_Volume'] = df_events['Political_News_Volume'].ffill().fillna(0)
        
        logger.info("Political event features created successfully")
        return df_events
    except Exception as e:
        logger.error(f"Error creating political features: {e}")
        raise


def simulate_gdelt_volume(start_date, end_date):
    """
    Simulate GDELT news volume (replace with real API call).
    """
    date_range = pd.date_range(start_date, end_date)
    volume = np.random.poisson(lam=5, size=len(date_range))
    return pd.Series(volume, index=date_range)


def simulate_gdelt_tone(start_date, end_date):
    """
    Simulate GDELT sentiment tone (replace with real API call).
    """
    date_range = pd.date_range(start_date, end_date)
    tone = np.random.uniform(-10, 10, size=len(date_range))
    return pd.Series(tone, index=date_range)


def get_data(ticker=TICKER, period=PERIOD):
    """
    Main data loading function combining price and political features.
    """
    try:
        logger.info(f"Loading complete dataset for {ticker}...")
        
        # Get price data
        df = get_price_data(ticker, period)
        
        # Add technical features
        df = add_technical_features(df)
        
        # Add political features
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        pol_df = get_political_events_features(start, end)
        
        # Join political features
        if not pol_df.empty:
            df = df.join(pol_df, how='left')
            
            # Create interaction features
            df['Pol_Volatility_Interaction'] = df['Political_Risk_Score'] * df['Volatility']
            df['Pol_Volume_Interaction'] = df['Political_Risk_Score'] * (df['Volume'] / df['Volume'].mean())
            
            # Fill NA values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
        
        # Drop remaining NaN values
        df = df.dropna()
        
        logger.info(f"Dataset prepared: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading complete dataset: {e}")
        raise
