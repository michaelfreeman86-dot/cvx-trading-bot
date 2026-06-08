"""
Advanced feature engineering module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handle feature scaling, normalization, and dimensionality reduction.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.pca = None
        self.feature_names = None
        self.scaled_feature_names = None
    
    def create_lagged_features(self, df, lags=[1, 2, 3, 5, 10]):
        """
        Create lagged features for time series.
        """
        try:
            logger.info(f"Creating lagged features with lags: {lags}")
            
            df_lagged = df.copy()
            
            for col in ['Close', 'Volume', 'RSI_14', 'MACD']:
                if col in df.columns:
                    for lag in lags:
                        df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            df_lagged = df_lagged.dropna()
            logger.info(f"Lagged features created: {len(df_lagged.columns)} total columns")
            return df_lagged
        except Exception as e:
            logger.error(f"Error creating lagged features: {e}")
            raise
    
    def create_rolling_features(self, df, windows=[5, 10, 20]):
        """
        Create rolling statistics features.
        """
        try:
            logger.info(f"Creating rolling features with windows: {windows}")
            
            df_rolling = df.copy()
            
            for window in windows:
                df_rolling[f'Close_rolling_mean_{window}'] = df['Close'].rolling(window).mean()
                df_rolling[f'Close_rolling_std_{window}'] = df['Close'].rolling(window).std()
                df_rolling[f'Volume_rolling_mean_{window}'] = df['Volume'].rolling(window).mean()
                df_rolling[f'Returns_rolling_mean_{window}'] = df['Daily_Return'].rolling(window).mean()
            
            df_rolling = df_rolling.dropna()
            logger.info(f"Rolling features created: {len(df_rolling.columns)} total columns")
            return df_rolling
        except Exception as e:
            logger.error(f"Error creating rolling features: {e}")
            raise
    
    def scale_features(self, df, fit=True):
        """
        Scale features using StandardScaler.
        """
        try:
            logger.info("Scaling features...")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fit:
                df_scaled = pd.DataFrame(
                    self.scaler.fit_transform(df[numeric_cols]),
                    columns=numeric_cols,
                    index=df.index
                )
            else:
                df_scaled = pd.DataFrame(
                    self.scaler.transform(df[numeric_cols]),
                    columns=numeric_cols,
                    index=df.index
                )
            
            logger.info("Features scaled successfully")
            return df_scaled
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            raise
    
    def normalize_features(self, df, fit=True):
        """
        Normalize features using MinMaxScaler (0-1 range).
        """
        try:
            logger.info("Normalizing features...")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fit:
                df_normalized = pd.DataFrame(
                    self.minmax_scaler.fit_transform(df[numeric_cols]),
                    columns=numeric_cols,
                    index=df.index
                )
            else:
                df_normalized = pd.DataFrame(
                    self.minmax_scaler.transform(df[numeric_cols]),
                    columns=numeric_cols,
                    index=df.index
                )
            
            logger.info("Features normalized successfully")
            return df_normalized
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            raise
    
    def apply_pca(self, df, n_components=0.95, fit=True):
        """
        Apply PCA for dimensionality reduction.
        """
        try:
            logger.info(f"Applying PCA with {n_components} variance explained...")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fit:
                self.pca = PCA(n_components=n_components)
                pca_data = self.pca.fit_transform(df[numeric_cols])
            else:
                pca_data = self.pca.transform(df[numeric_cols])
            
            n_components_actual = self.pca.n_components_
            logger.info(f"PCA reduced to {n_components_actual} components")
            
            df_pca = pd.DataFrame(
                pca_data,
                columns=[f'PC_{i+1}' for i in range(n_components_actual)],
                index=df.index
            )
            
            return df_pca
        except Exception as e:
            logger.error(f"Error applying PCA: {e}")
            raise
    
    def select_features(self, df, method='variance_threshold', threshold=0.01):
        """
        Select important features based on variance.
        """
        try:
            logger.info(f"Selecting features using {method} method...")
            
            from sklearn.feature_selection import VarianceThreshold
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if method == 'variance_threshold':
                selector = VarianceThreshold(threshold=threshold)
                df_selected = pd.DataFrame(
                    selector.fit_transform(df[numeric_cols]),
                    columns=df[numeric_cols].columns[selector.get_support()],
                    index=df.index
                )
            
            logger.info(f"Selected {len(df_selected.columns)} features")
            return df_selected
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            raise
