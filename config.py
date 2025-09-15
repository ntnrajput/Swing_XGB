# refactored_fyers_swing/config.py

import os
from pathlib import Path

# === FYERS API ===
FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "VE3CCLJZWA-100" )
FYERS_SECRET_ID = os.getenv("FYERS_SECRET_ID", "QEGA69PVUL")
FYERS_REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "https://www.google.com" )
FYERS_APP_ID_HASH = os.getenv("FYERS_APP_ID_HASH", "b209632623b60de416ea3bcbd2b780ef11ebdbb652b3f06f63ffdd34366faa18")

TOKEN_PATH = Path("outputs/fyers_access_token.txt")


#=====Good Stock Filter====
AVG_VOL = 3
AVG_PRICE = 100

# === DATA SETTINGS ===
SYMBOLS_FILE = Path("symbols.csv")
HISTORICAL_DATA_FILE = Path("outputs/all_symbols_history.parquet")
HISTORICAL_DATA_FILE_csv = Path("outputs/all_symbols_history.csv")
LATEST_DATA_FILE = Path("outputs/latest_full_data.parquet")
DAILY_DATA_FILE = Path("outputs/today_data.csv")


# === INDICATOR SETTINGS ===
EMA_PERIODS = [20, 50, 200]
RSI_PERIOD = 14
ATR_Period = 14
Vol_Avg_Period = 50
Fib_Pivot_Window = 21
VOLUME_LOOKBACK = 20

Strong_Low_Close = -0.035
Strong_High_Close = 0.07

# === MODEL ===
version = input("Please enter model verison")
MODEL_FILE = Path(f"models/{version}/enhanced_model_pipeline.pkl")

# === LOGGING ===
LOG_FILE = Path("outputs/logs/system.log")

#====Feature Columns====
# FEATURE_COLUMNS = [
#     'ema20_ema50',	'sma20_sma50','ema50_ema200', 'sma50_sma200',	'ema20_price',	'ema50_price',	'ema200_price',
#     'sma20_price','sma50_price','sma200_price','rsi',	'atr_pct',
#     'obv',	'vol_change_5d', 'ret_1d',	'ret_3d',	'ret_5d',	'ret_10d', 'vol_5d',	
#     'vol_10d',	'rsi_3_slope',	'rsi_5_slope',	'ema20_above_ema50', 'ema20_50_cross_up', 'ema20_50_cross_down',
#     'close_position_in_range',	'gap_pct',	'fib_pivot_distance_pct',	'fib_r1_distance_pct',	'fib_r2_distance_pct',
#     'fib_s1_distance_pct',	'fib_s2_distance_pct', 'is_bullish',	'is_bearish',	'body_to_range',	
#     'upper_shadow_to_range',	'lower_shadow_to_range',	'is_doji',	'is_hammer',	'is_shooting_star',
#     'spinning_top','marubozu_bull','marubozu_bear','three_white_soldiers','three_black_crows', 
#     'morning_star', 'evening_star', 'bullish_harami','bearish_harami', 'pattern_strength',
#     'close_compared_to_previous', 'bb_position', 'support_pct', 'resistance_pct',  'sma20_50_cross_up',
#     'macd_line', 'macd_signal', 'macd_histogram', 'macd_line_roc_3', 'macd_line_roc_5', 'macd_line_roc_10', 'macd_signal_roc_3', 'macd_signal_roc_5', 'macd_hist_roc_3', 'macd_hist_roc_5', 'macd_above_signal', 'macd_cross_above', 'macd_cross_below', 'macd_above_zero', 'macd_cross_above_zero', 'macd_cross_below_zero', 'macd_hist_positive', 'macd_hist_increasing', 'macd_bullish_divergence', 'macd_bearish_divergence', 'macd_line_rank_20', 'macd_line_rank_50', 'macd_histogram_rank_20', 'macd_line_std_10', 'macd_line_std_20', 'macd_histogram_std_10', 'macd_signal_spread', 'macd_signal_spread_pct', 'macd_consecutive_above_signal', 'macd_consecutive_positive_hist', 'macd_line_acceleration', 'macd_histogram_acceleration', 'macd_line_normalized', 'macd_histogram_normalized', 'macd_line_zscore_20', 'macd_line_zscore_50', 'macd_fast_line', 'macd_fast_histogram', 'macd_fast_above_signal', 'macd_slow_line', 'macd_slow_histogram', 'macd_slow_above_signal', 'macd_timeframe_agreement',
#     'large_candle','opening_gap','end_day_strength','overnight_gap','higher_high_seq','higher_low_seq','momentum_3','momentum_5','momentum_10','roc_acceleration','true_range','volatility_14','range_expansion','volume_sma_10','volume_sma_20','volume_price_trend','unusual_volume','price_acceleration'
# ]

FEATURE_COLUMNS = [
    # Original Basic Features
    'ema20_ema50', 'sma20_sma50', 'ema50_ema200', 'sma50_sma200', 'ema20_price', 'ema50_price', 'ema200_price',
    'sma20_price', 'sma50_price', 'sma200_price', 'rsi', 'atr_pct',
    'obv', 'vol_change_5d', 'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'vol_5d',
    'vol_10d', 'rsi_3_slope', 'rsi_5_slope', 'ema20_above_ema50', 'ema20_50_cross_up', 'ema20_50_cross_down',
    'close_position_in_range', 'gap_pct', 'fib_pivot_distance_pct', 'fib_r1_distance_pct', 'fib_r2_distance_pct',
    'fib_s1_distance_pct', 'fib_s2_distance_pct', 
    
    # Original Candlestick Features
    'is_bullish', 'is_bearish', 'body_to_range',
    'upper_shadow_to_range', 'lower_shadow_to_range', 'is_doji', 'is_hammer', 'is_shooting_star',
    'spinning_top', 'marubozu_bull', 'marubozu_bear', 'three_white_soldiers', 'three_black_crows',
    'morning_star', 'evening_star', 'bullish_harami', 'bearish_harami', 'pattern_strength',
    'close_compared_to_previous', 'bb_position', 'support_pct', 'resistance_pct', 'sma20_50_cross_up',
    
    # Original MACD Features
    'macd_line', 'macd_signal', 'macd_histogram', 'macd_line_roc_3', 'macd_line_roc_5', 'macd_line_roc_10',
    'macd_signal_roc_3', 'macd_signal_roc_5', 'macd_hist_roc_3', 'macd_hist_roc_5', 'macd_above_signal',
    'macd_cross_above', 'macd_cross_below', 'macd_above_zero', 'macd_cross_above_zero', 'macd_cross_below_zero',
    'macd_hist_positive', 'macd_hist_increasing', 'macd_bullish_divergence', 'macd_bearish_divergence',
    'macd_line_rank_20', 'macd_line_rank_50', 'macd_histogram_rank_20', 'macd_line_std_10', 'macd_line_std_20',
    'macd_histogram_std_10', 'macd_signal_spread', 'macd_signal_spread_pct', 'macd_consecutive_above_signal',
    'macd_consecutive_positive_hist', 'macd_line_acceleration', 'macd_histogram_acceleration',
    'macd_line_normalized', 'macd_histogram_normalized', 'macd_line_zscore_20', 'macd_line_zscore_50',
    'macd_fast_line', 'macd_fast_histogram', 'macd_fast_above_signal', 'macd_slow_line', 'macd_slow_histogram',
    'macd_slow_above_signal', 'macd_timeframe_agreement',
    
    # Original Advanced Features
    'large_candle', 'opening_gap', 'end_day_strength', 'overnight_gap', 'higher_high_seq', 'higher_low_seq',
    'momentum_3', 'momentum_5', 'momentum_10', 'roc_acceleration', 'true_range', 'volatility_14',
    'range_expansion', 'volume_sma_10', 'volume_sma_20', 'volume_price_trend', 'unusual_volume',
    'price_acceleration',
    
    # NEW ENHANCED FEATURES - Advanced Volatility Features
    'vol_14_percentile', 'atr_pct_percentile', 'vol_regime_high', 'vol_regime_low', 'vol_expansion',
    'vol_breakout', 'atr_multiple', 'price_atr_position', 'vol_cluster', 'gap_vol_ratio',
    
    # Volatility Features
    'vol_14_percentile', 'atr_pct_percentile', 'vol_regime_high','vol_regime_low', 'vol_expansion',
    'vol_breakout', 'atr_multiple', 'price_atr_position', 'vol_cluster', 'gap_vol_ratio',
    
    # Trend Features
    'price_above_sma20', 'price_above_sma50', 'price_above_sma200','trend_alignment_score',
    'trend_strength', 'trend_acceleration', 'sma20_slope_3', 'sma20_slope_5', 'sma50_slope_3',
    'sma50_slope_5', 'sma200_slope_3', 'sma200_slope_5', 'trend_consistency', 'pullback_to_sma20', 'pullback_to_sma50',
    
    # MACD Features
    'macd_momentum', 'macd_acceleration', 'macd_price_divergence', 'macd_hist_momentum', 'macd_hist_mean_reversion',
    'macd_hist_above_zero', 'macd_hist_zero_cross', 'macd_signal_strength', 'macd_convergence',
    
    # Microstructure Features
    'intraday_strength', 'price_efficiency', 'upper_shadow_pct', 'lower_shadow_pct', 'shadow_ratio',
    'range_position', 'range_expansion', 'efficiency_ratio',
    
    # Regime Features
    'high_vol_regime', 'low_vol_regime', 'bull_regime', 'bear_regime','regime_persistence',
    'momentum_regime', 'mean_reversion_regime',
    
    # Volume Features
    'vpt_momentum', 'vpt_divergence', 'volume_surge', 'volume_dry_up','price_vwap_ratio',
    'obv_momentum', 'obv_divergence', 'volume_roc_5', 'volume_roc_10',
    
    # Price Action Quality Features
    'bullish_candle_quality', 'bearish_candle_quality','setup_quality_score', 'breakout_quality', 'pullback_quality'
]




CONFIDENCE_THRESHOLD = 0.6
# old model
# 0.7   452  30.5  -6
# 0.72    300   19.84   -5.83


# new model 
# 0.7   409  23.82  -8.27
# 0.72    303   21.98   -5.45
# 0.73    253   18.77   -1.02
# 0.75    145   15.67  -1.05
# 0.76  145   13.81  -1.07
# 0.77  75   9.14  -1.11


# Latest v1
# 0.6  327 10.16 -9.27
# 0.65 236 16.3 -8.65
# 0.7 168 -0.46 -8.14
# 0.75 97 5 -4.17
# 0.8 43 1.3 -4.32

