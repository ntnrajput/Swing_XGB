# features/advanced_features.py
import pandas as pd
import numpy as np

def calculate_advanced_features(df):
    """
    Simplified advanced features using only stock OHLCV data.
    df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
    """
    eps = 1e-9  # very small constant
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    # ------------------------
    # 1. Time Features
    # ------------------------
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month_of_year'] = df['date'].dt.month

    # ------------------------
    # 2. Price Action & Candlestick Patterns
    # ------------------------
    df['large_candle'] = (df['high'] - df['low']) / df['close']
    df['opening_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['end_day_strength'] = (df['close'] - df['low']) / (df['high'] - df['low']+eps)
    df['overnight_gap'] = df['open'].pct_change()

    # Swing & Trend Patterns
    df['higher_high_seq'] = (df['high'] > df['high'].shift(1)).astype(int).groupby((df['high'] <= df['high'].shift(1)).cumsum()).cumsum()
    df['higher_low_seq'] = (df['low'] > df['low'].shift(1)).astype(int).groupby((df['low'] <= df['low'].shift(1)).cumsum()).cumsum()

    # ------------------------
    # 3. Momentum Features
    # ------------------------
    df['momentum_3'] = df['close'].pct_change(3)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['roc_acceleration'] = df['close'].pct_change() - df['close'].pct_change(3)

    # ------------------------
    # 4. Volatility Features
    # ------------------------
    df['true_range'] = df['high'] - df['low']
    df['volatility_14'] = df['true_range'].rolling(14).std()
    
    df['range_expansion'] = (df['high'] - df['low']) / (df['volatility_14'] + eps)

    # ------------------------
    # 5. Volume Features
    # ------------------------
    df['volume_sma_10'] = df['volume'] / df['volume'].rolling(10).mean()
    df['volume_sma_20'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_price_trend'] = (df['close'].diff() * df['volume']).cumsum()
    df['unusual_volume'] = (df['volume'] > 2 * df['volume'].rolling(20).mean()).astype(int)

    # ------------------------
    # 6. Price Acceleration
    # ------------------------
    df['price_acceleration'] = df['close'].diff().diff()

    return df
