# features/enhanced_features.py

import pandas as pd
import numpy as np
from numba import jit
import warnings
warnings.filterwarnings('ignore')

def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add enhanced features based on feature importance analysis.
    Focus on volatility, trend, and momentum features that complement existing top performers.
    All features are normalized/scaled to be comparable across different stocks.
    """
    df = df.copy()
    
    # Volatility-based features (since atr_pct and volatility_14 are top features)
    df = add_advanced_volatility_features(df)
    
    # Enhanced trend features (building on SMA importance)
    df = add_advanced_trend_features(df)
    
    # Enhanced MACD derivatives (since MACD features dominate)
    df = add_macd_derivatives(df)
    
    # Market microstructure features
    df = add_microstructure_features(df)
    
    # Regime-based features
    df = add_regime_features(df)
    
    # Enhanced volume features (building on volume_price_trend)
    df = add_advanced_volume_features(df)
    
    # Price action quality features
    df = add_price_action_quality(df)
    
    return df

def add_advanced_volatility_features(df):
    """Enhanced volatility features based on ATR and volatility importance."""
    # Volatility percentiles (market context) - already normalized
    df['vol_14_percentile'] = df['volatility_14'].rolling(252).rank(pct=True)
    df['atr_pct_percentile'] = df['atr_pct'].rolling(252).rank(pct=True)
    
    # Volatility regime changes - binary features
    df['vol_regime_high'] = (df['vol_14_percentile'] > 0.8).astype(int)
    df['vol_regime_low'] = (df['vol_14_percentile'] < 0.2).astype(int)
    
    # Volatility expansion/contraction - ratio form
    df['vol_expansion'] = df['volatility_14'] / df['volatility_14'].rolling(20).mean()
    df['vol_breakout'] = (df['vol_expansion'] > 1.5).astype(int)
    
    # ATR-based features - all in ratio/percentage form
    df['atr_multiple'] = (df['high'] - df['low']) / df['atr']  # How many ATRs is current range
    df['price_atr_position'] = (df['close'] - df['low']) / df['atr']  # Position within ATR from low
    
    # Volatility clustering - coefficient of variation
    vol_mean = df['volatility_14'].rolling(5).mean()
    df['vol_cluster'] = df['volatility_14'].rolling(5).std() / (vol_mean + 1e-8)
    
    # Gap volatility - ratio form
    if 'gap_pct' in df.columns:
        df['gap_vol_ratio'] = np.abs(df['gap_pct']) / (df['atr_pct'] + 1e-8)
    
    return df

def add_advanced_trend_features(df):
    """Enhanced trend features building on SMA importance."""
    # Multi-timeframe trend alignment - check available SMAs
    available_sma_periods = []
    for period in [20, 50, 200]:  # Common periods
        sma_col = f'sma{period}'
        if sma_col in df.columns:
            available_sma_periods.append(period)
            df[f'price_above_sma{period}'] = (df['close'] > df[sma_col]).astype(int)
    
    # Calculate trend alignment score based on available SMAs
    if available_sma_periods:
        trend_score = sum(df[f'price_above_sma{period}'] for period in available_sma_periods)
        df['trend_alignment_score'] = trend_score / len(available_sma_periods)
    else:
        df['trend_alignment_score'] = 0.5
    
    # Trend strength using multiple SMAs - percentage form
    if 'sma20' in df.columns and 'sma200' in df.columns:
        df['trend_strength'] = (df['sma20'] - df['sma200']) / df['sma200']  # Already percentage
        df['trend_acceleration'] = df['trend_strength'].diff()
    
    # SMA slope (trend momentum) - percentage change form
    for period in available_sma_periods:
        sma_col = f'sma{period}'
        df[f'sma{period}_slope_3'] = df[sma_col].pct_change(3)  # Percentage change
        df[f'sma{period}_slope_5'] = df[sma_col].pct_change(5)  # Percentage change
    
    # Trend consistency - rolling proportion
    if 'sma50' in df.columns:
        df['trend_consistency'] = (df['sma50'] > df['sma50'].shift(5)).rolling(20).sum() / 20
    
    # Distance from trend lines as percentage of price (NOT ATR)
    if 'sma20' in df.columns:
        df['pullback_to_sma20'] = (df['close'] - df['sma20']) / df['close']  # Percentage distance
    if 'sma50' in df.columns:
        df['pullback_to_sma50'] = (df['close'] - df['sma50']) / df['close']  # Percentage distance
    
    return df

def add_macd_derivatives(df):
    """Enhanced MACD features building on existing MACD importance."""
    # MACD momentum features - percentage of price
    if 'macd_line' in df.columns:
        df['macd_momentum'] = df['macd_line'].diff() / df['close']  # As percentage of price
        df['macd_acceleration'] = df['macd_momentum'].diff()
        
        # MACD divergence with price - standardized
        df['macd_price_divergence'] = calculate_divergence(df['close'], df['macd_line'])
    
    if 'macd_histogram' in df.columns:
        # Histogram momentum as percentage of price
        df['macd_hist_momentum'] = df['macd_histogram'].diff() / df['close']
        
        # Histogram mean reversion - z-score form
        hist_std = df['macd_histogram'].rolling(20).std()
        df['macd_hist_mean_reversion'] = df['macd_histogram'] / (hist_std + 1e-8)
        
        # Histogram zero-line interactions - binary
        df['macd_hist_above_zero'] = (df['macd_histogram'] > 0).astype(int)
        df['macd_hist_zero_cross'] = (
            (df['macd_histogram'] > 0) & (df['macd_histogram'].shift(1) <= 0)
        ).astype(int)
    
    # MACD signal quality - as percentage of price
    if 'macd_signal' in df.columns and 'macd_line' in df.columns:
        df['macd_signal_strength'] = np.abs(df['macd_line'] - df['macd_signal']) / df['close']
        df['macd_convergence'] = (df['macd_signal_strength'] < df['macd_signal_strength'].shift(1)).astype(int)
    
    return df

def add_microstructure_features(df):
    """Market microstructure features for swing trading."""
    # Intraday strength - ratio form
    df['intraday_strength'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Price efficiency - ratio form
    df['price_efficiency'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Shadow analysis - percentage of price
    df['upper_shadow_pct'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
    df['lower_shadow_pct'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
    df['shadow_ratio'] = df['upper_shadow_pct'] / (df['lower_shadow_pct'] + 1e-8)
    
    # Trading range analysis - normalized forms
    df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Range expansion relative to recent volatility
    recent_vol = df['close'].pct_change().rolling(20).std()
    df['range_expansion'] = (df['high'] - df['low']) / df['close'] / (recent_vol + 1e-8)
    
    # Close-to-close vs range relationship - already normalized
    df['efficiency_ratio'] = np.abs(df['close'].pct_change()) / (df['atr_pct'] + 1e-8)
    
    return df

def add_regime_features(df):
    """Market regime identification features."""
    # Volatility regime - percentile based
    vol_threshold_high = df['volatility_14'].rolling(252).quantile(0.75)
    vol_threshold_low = df['volatility_14'].rolling(252).quantile(0.25)
    
    df['high_vol_regime'] = (df['volatility_14'] > vol_threshold_high).astype(int)
    df['low_vol_regime'] = (df['volatility_14'] < vol_threshold_low).astype(int)
    
    # Trend regime - binary based on SMA relationship
    if 'sma50' in df.columns and 'sma200' in df.columns:
        df['bull_regime'] = (df['sma50'] > df['sma200']).astype(int)
        df['bear_regime'] = (df['sma50'] < df['sma200']).astype(int)
        
        # Regime persistence - rolling proportion
        df['regime_persistence'] = df['bull_regime'].rolling(20).mean()
    
    # Mean reversion vs momentum regime - based on return correlation
    returns_5d = df['close'].pct_change(5)
    returns_20d = df['close'].pct_change(20)
    
    df['momentum_regime'] = (returns_5d * returns_20d > 0).astype(int)
    df['mean_reversion_regime'] = (returns_5d * returns_20d < 0).astype(int)
    
    return df

def add_advanced_volume_features(df):
    """Enhanced volume features building on volume_price_trend importance."""
    if 'volume' not in df.columns:
        # Create dummy features if volume not available
        df['vpt_momentum'] = 0
        df['vpt_divergence'] = 0
        df['volume_surge'] = 0
        df['volume_dry_up'] = 0
        df['price_vwap_ratio'] = 1.0
        df['obv_momentum'] = 0
        df['obv_divergence'] = 0
        df['volume_roc_5'] = 0
        df['volume_roc_10'] = 0
        return df
    
    # Volume-Price Trend (VPT) derivatives - percentage change
    if 'volume_price_trend' in df.columns:
        df['vpt_momentum'] = df['volume_price_trend'].pct_change(5)
        df['vpt_divergence'] = calculate_divergence(df['close'], df['volume_price_trend'])
    
    # Volume patterns - ratio based
    vol_avg_20 = df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > 2 * vol_avg_20).astype(int)
    df['volume_dry_up'] = (df['volume'] < 0.5 * vol_avg_20).astype(int)
    
    # Volume-weighted features - only keep ratios, not absolute values
    vwap_20 = calculate_vwap(df, 20)
    df['price_vwap_ratio'] = df['close'] / vwap_20  # Ratio form only
    
    # Accumulation/Distribution derivatives - percentage change
    if 'obv' in df.columns:
        df['obv_momentum'] = df['obv'].pct_change(5)
        df['obv_divergence'] = calculate_divergence(df['close'], df['obv'])
    
    # Volume rate of change - already percentage
    df['volume_roc_5'] = df['volume'].pct_change(5)
    df['volume_roc_10'] = df['volume'].pct_change(10)
    
    return df

def add_price_action_quality(df):
    """Price action quality and setup identification."""
    # Candle quality scores - normalized 0-1 scale
    df['bullish_candle_quality'] = calculate_bullish_quality(df)
    df['bearish_candle_quality'] = calculate_bearish_quality(df)
    
    # Setup quality (multiple confirmations) - weighted score
    setup_score = np.zeros(len(df))
    
    # Trend alignment (0-1)
    if 'trend_alignment_score' in df.columns:
        setup_score += df['trend_alignment_score'] * 0.4
    
    # Volume confirmation (0-0.3)
    if 'volume_surge' in df.columns:
        setup_score += df['volume_surge'] * 0.3
    
    # MACD confirmation (0-0.3)
    if 'macd_hist_above_zero' in df.columns:
        setup_score += df['macd_hist_above_zero'] * 0.3
    
    df['setup_quality_score'] = setup_score
    
    # Breakout quality - normalized score
    df['breakout_quality'] = calculate_breakout_quality(df)
    
    # Pullback quality - normalized score
    df['pullback_quality'] = calculate_pullback_quality(df)
    
    return df

@jit(nopython=True)
def calculate_divergence_numba(price_arr, indicator_arr, period=10):
    """Fast divergence calculation using numba."""
    n = len(price_arr)
    divergence = np.full(n, 0.0)
    
    for i in range(period, n):
        price_slope = (price_arr[i] - price_arr[i-period]) / period
        indicator_slope = (indicator_arr[i] - indicator_arr[i-period]) / period
        
        # Bearish divergence: price up, indicator down
        if price_slope > 0 and indicator_slope < 0:
            divergence[i] = -1
        # Bullish divergence: price down, indicator up
        elif price_slope < 0 and indicator_slope > 0:
            divergence[i] = 1
            
    return divergence

def calculate_divergence(price_series, indicator_series, period=10):
    """Calculate price-indicator divergence."""
    price_arr = price_series.fillna(0).values
    indicator_arr = indicator_series.fillna(0).values
    
    divergence_arr = calculate_divergence_numba(price_arr, indicator_arr, period)
    return pd.Series(divergence_arr, index=price_series.index)

def calculate_vwap(df, period):
    """Calculate Volume Weighted Average Price."""
    if 'volume' not in df.columns:
        return df['close'].rolling(period).mean()
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    volume_price = typical_price * df['volume']
    
    return volume_price.rolling(period).sum() / df['volume'].rolling(period).sum()

def calculate_bullish_quality(df):
    """Calculate bullish candle quality score (0-1 normalized)."""
    quality = np.zeros(len(df))
    
    # Strong close relative to range (0-0.5)
    if 'range_position' in df.columns:
        quality += df['range_position'] * 0.5
    
    # Volume confirmation (0-0.3)
    if 'vol_by_avg_vol' in df.columns:
        vol_score = np.minimum(df['vol_by_avg_vol'] - 1, 1)  # Above average gets score
        vol_score = np.maximum(vol_score, 0)  # No negative scores
        quality += vol_score * 0.3
    
    # Small upper shadow (0-0.2)
    if 'upper_shadow_pct' in df.columns:
        shadow_score = 1 - np.minimum(df['upper_shadow_pct'] * 10, 1)  # Penalize large shadows
        quality += shadow_score * 0.2
    
    return np.minimum(quality, 1.0)  # Cap at 1.0

def calculate_bearish_quality(df):
    """Calculate bearish candle quality score (0-1 normalized)."""
    quality = np.zeros(len(df))
    
    # Weak close relative to range (0-0.5)
    if 'range_position' in df.columns:
        quality += (1 - df['range_position']) * 0.5
    
    # Volume confirmation (0-0.3)
    if 'vol_by_avg_vol' in df.columns:
        vol_score = np.minimum(df['vol_by_avg_vol'] - 1, 1)  # Above average gets score
        vol_score = np.maximum(vol_score, 0)  # No negative scores
        quality += vol_score * 0.3
    
    # Small lower shadow (0-0.2)
    if 'lower_shadow_pct' in df.columns:
        shadow_score = 1 - np.minimum(df['lower_shadow_pct'] * 10, 1)  # Penalize large shadows
        quality += shadow_score * 0.2
    
    return np.minimum(quality, 1.0)  # Cap at 1.0

def calculate_breakout_quality(df):
    """Calculate breakout quality score (0-1 normalized)."""
    quality = np.zeros(len(df))
    
    # Volume expansion (0-0.5)
    if 'volume_surge' in df.columns:
        quality += df['volume_surge'] * 0.5
    
    # Volatility expansion (0-0.3)
    if 'vol_breakout' in df.columns:
        quality += df['vol_breakout'] * 0.3
    
    # Range expansion (0-0.2)
    if 'range_expansion' in df.columns:
        range_score = np.minimum(df['range_expansion'] / 2, 1)  # Normalize by dividing by 2
        quality += range_score * 0.2
    
    return np.minimum(quality, 1.0)  # Cap at 1.0

def calculate_pullback_quality(df):
    """Calculate pullback/retracement quality score (0-1 normalized)."""
    quality = np.zeros(len(df))
    
    # Trend context (0-0.4)
    if 'trend_alignment_score' in df.columns:
        quality += df['trend_alignment_score'] * 0.4
    
    # Support at moving average (0-0.4)
    if 'pullback_to_sma20' in df.columns:
        # Good pullback if close to SMA20 (within 2%)
        pullback_score = np.exp(-np.abs(df['pullback_to_sma20']) * 50)  # Exponential decay
        quality += pullback_score * 0.4
    
    # Volume dry-up during pullback (0-0.2)
    if 'volume_dry_up' in df.columns:
        quality += df['volume_dry_up'] * 0.2
    
    return np.minimum(quality, 1.0)  # Cap at 1.0