# features/swing_utils.py

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(prices, period=14):
    """Calculate RSI with improved smoothing."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Use Wilder's smoothing method
    gain_smooth = gain.ewm(alpha=1/period, adjust=False).mean()
    loss_smooth = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = gain_smooth / loss_smooth
    rsi = 1 - (1 / (1 + rs))
    return rsi

def calculate_bb_position(prices, period=20, std_dev=2):
    """Calculate Bollinger Band position with configurable parameters."""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # Calculate position within bands
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    
    # Add squeeze detection
    bb_width = (upper_band - lower_band) / sma
    bb_squeeze = bb_width < bb_width.rolling(20).quantile(0.1)
    
    return bb_position.clip(0, 1)

def add_candle_features(df):
    """Enhanced candlestick pattern recognition."""
    df = df.copy()
    
    # Vectorized basic candle components
    df['body'] = np.abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    
    # Vectorized max/min operations
    ohlc_max = np.maximum(df['open'], df['close'])
    ohlc_min = np.minimum(df['open'], df['close'])
    
    df['upper_shadow'] = df['high'] - ohlc_max 
    df['lower_shadow'] = ohlc_min - df['low']
    
    # Candle direction
    df['is_bullish'] = (df['close'] > df['open']).astype(np.int8)
    df['is_bearish'] = (df['open'] > df['close']).astype(np.int8)
    
    # Ratios with epsilon for numerical stability
    range_safe = df['range'] + 1e-8
    df['body_to_range'] = df['body'] / range_safe
    df['upper_shadow_to_range'] = df['upper_shadow'] / range_safe
    df['lower_shadow_to_range'] = df['lower_shadow'] / range_safe
    
    # Vectorized basic patterns
    df['is_doji'] = (df['body_to_range'] < 0.1).astype(np.int8)
    df['is_hammer'] = ((df['body_to_range'] < 0.3) & 
                       (df['lower_shadow_to_range'] > 0.6) & 
                       (df['upper_shadow_to_range'] < 0.1)).astype(np.int8)
    df['is_shooting_star'] = ((df['body_to_range'] < 0.3) & 
                             (df['upper_shadow_to_range'] > 0.6) & 
                             (df['lower_shadow_to_range'] < 0.1)).astype(np.int8)
    
    df['close_compared_to_previous'] = (df['close']-df['close'].shift(1))/df['close']
    

    # Advanced patterns
    df = add_advanced_candle_patterns(df)
    
    # Pattern strength
    df['pattern_strength'] = calculate_pattern_strength(df)
    
    return df

def add_advanced_candle_patterns(df):
    """Add advanced candlestick patterns."""
    df = df.copy()
    
    # Vectorized spinning tops
    df['spinning_top'] = ((df['body_to_range'] < 0.3) & 
                         (df['upper_shadow_to_range'] > 0.3) & 
                         (df['lower_shadow_to_range'] > 0.3)).astype(np.int8)
    
    # Vectorized Marubozu
    df['marubozu_bull'] = ((df['body_to_range'] > 0.95) & 
                          (df['close'] > df['open'])).astype(np.int8)
    df['marubozu_bear'] = ((df['body_to_range'] > 0.95) & 
                          (df['close'] < df['open'])).astype(np.int8)
    
    # Pre-compute shifts for multi-candle patterns
    close_s1 = df['close'].shift(1)
    close_s2 = df['close'].shift(2)
    is_bullish_s1 = df['is_bullish'].shift(1)
    is_bullish_s2 = df['is_bullish'].shift(2)
    is_bearish_s1 = df['is_bearish'].shift(1)
    is_bearish_s2 = df['is_bearish'].shift(2)
    body_s1 = df['body'].shift(1)
    body_s2 = df['body'].shift(2)
    open_s2 = df['open'].shift(2)
    
    # Three white soldiers / Three black crows
    df['three_white_soldiers'] = (
        (df['is_bullish'] == 1) & 
        (is_bullish_s1 == 1) & 
        (is_bullish_s2 == 1) & 
        (df['close'] > close_s1) & 
        (close_s1 > close_s2) & 
        (df['body'] > body_s1 * 0.8) & 
        (body_s1 > body_s2 * 0.8)
    ).astype(np.int8)
    
    df['three_black_crows'] = (
        (df['is_bearish'] == 1) & 
        (is_bearish_s1 == 1) & 
        (is_bearish_s2 == 1) & 
        (df['close'] < close_s1) & 
        (close_s1 < close_s2) & 
        (df['body'] > body_s1 * 0.8) & 
        (body_s1 > body_s2 * 0.8)
    ).astype(np.int8)
    
    # Morning star / Evening star
    midpoint_s2 = (open_s2 + close_s2) / 2
    df['morning_star'] = (
        (is_bearish_s2 == 1) & 
        (body_s1 < body_s2 * 0.3) & 
        (df['is_bullish'] == 1) & 
        (df['close'] > midpoint_s2)
    ).astype(np.int8)
    
    df['evening_star'] = (
        (is_bullish_s2 == 1) & 
        (body_s1 < body_s2 * 0.3) & 
        (df['is_bearish'] == 1) & 
        (df['close'] < midpoint_s2)
    ).astype(np.int8)
    
    # Harami patterns
    df['bullish_harami'] = (
        (is_bearish_s1 == 1) & 
        (df['is_bullish'] == 1) & 
        (df['high'] < close_s1) & 
        (df['low'] > open_s2)
    ).astype(np.int8)
    
    df['bearish_harami'] = (
        (is_bullish_s1 == 1) & 
        (df['is_bearish'] == 1) & 
        (df['high'] < open_s2) & 
        (df['low'] > close_s1)
    ).astype(np.int8)
    
    return df

def calculate_pattern_strength(df):
    """Calculate overall pattern strength score."""
    pattern_columns = [
        'is_doji', 'is_hammer', 'is_shooting_star', 'spinning_top',
        'marubozu_bull', 'marubozu_bear', 'three_white_soldiers',
        'three_black_crows', 'morning_star', 'evening_star',
        'bullish_harami', 'bearish_harami'
    ]
    
    # Weight patterns by reliability
    weights = np.array([
        0.5, 2.0, 2.0, 0.5, 1.5, 1.5, 3.0, 3.0, 2.5, 2.5, 1.5, 1.5
    ])
    
    # Vectorized calculation
    pattern_strength = np.zeros(len(df))
    for i, col in enumerate(pattern_columns):
        if col in df.columns:
            pattern_strength += df[col].values * weights[i]
    
    return pattern_strength


def detect_enhanced_swings(df, 
                          threshold_percent=12.0, 
                          lookback_window=3, 
                          lookahead_window=0,
                          min_time_gap=1, 
                          min_price_distance_pct=5.0):
    """
    Enhanced swing detection with advanced filtering logic.
    
    Parameters:
    - threshold_percent: Minimum percentage move required to confirm a swing
    - lookback_window: Must be highest/lowest in last N bars
    - lookahead_window: Must remain highest/lowest for next N bars  
    - min_time_gap: Minimum bars between consecutive swings
    - min_price_distance_pct: Minimum price difference between consecutive same-type swings
    
    Logic Improvements:
    1. Local extremes: Must be highest/lowest in lookback+lookahead window
    2. Time gap: Minimum spacing between swings
    3. Price distance: Meaningful breakout from previous swing levels
    """
    
    result_df = df.copy()
    n = len(df)
    
    # Initialize columns
    result_df['swing_high'] = np.nan
    result_df['swing_low'] = np.nan
    result_df['swing_high_confirmed'] = False
    result_df['swing_low_confirmed'] = False
    result_df['swing_type'] = ''
    result_df['swing_strength'] = 0  # How many bars it was the extreme
    
    threshold = threshold_percent / 100.0
    min_price_distance = min_price_distance_pct / 100.0
    
    # Extract arrays
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    # Track confirmed swings for spacing and price distance checks
    confirmed_swing_highs = []  # (index, price, date)
    confirmed_swing_lows = []   # (index, price, date)
    
    # Step 1: Find potential local extremes using look-back/look-ahead
    potential_swing_points = []
    
    for i in range(lookback_window, n - lookahead_window):
        current_high = highs[i]
        current_low = lows[i]
        
        # Check if this is a local high
        window_start = i - lookback_window
        window_end = i + lookahead_window + 1
        window_highs = highs[window_start:window_end]
        window_lows = lows[window_start:window_end]
        
        # Count how many bars this point was the extreme (strength measure)
        is_local_high = current_high == np.max(window_highs)
        is_local_low = current_low == np.min(window_lows)
        
        if is_local_high:
            # Calculate strength (how dominant this high is)
            strength = np.sum(window_highs <= current_high)
            potential_swing_points.append({
                'index': i,
                'type': 'high',
                'price': current_high,
                'strength': strength,
                'confirmed': False
            })
            
        if is_local_low:
            # Calculate strength (how dominant this low is)
            strength = np.sum(window_lows >= current_low)
            potential_swing_points.append({
                'index': i,
                'type': 'low', 
                'price': current_low,
                'strength': strength,
                'confirmed': False
            })
    
    # Step 2: Confirm swings using percentage reversal + additional filters
    for i in range(n):
        current_high = highs[i]
        current_low = lows[i]
        
        # Check potential swing highs for confirmation
        for swing_point in potential_swing_points:
            if (swing_point['type'] == 'high' and 
                swing_point['index'] < i and 
                not swing_point['confirmed']):
                
                swing_idx = swing_point['index']
                swing_price = swing_point['price']
                
                # 1. Percentage reversal check
                drop_percent = (swing_price - current_low) / swing_price
                
                if drop_percent >= threshold:
                    # 2. Time gap check
                    time_gap_ok = True
                    if confirmed_swing_highs:
                        last_swing_idx = confirmed_swing_highs[-1][0]
                        if swing_idx - last_swing_idx < min_time_gap:
                            time_gap_ok = False
                    
                    # 3. Price distance check
                    price_distance_ok = True
                    if confirmed_swing_highs:
                        last_swing_price = confirmed_swing_highs[-1][1]
                        price_diff_pct = abs(swing_price - last_swing_price) / last_swing_price
                        if price_diff_pct < min_price_distance:
                            price_distance_ok = False
                    
                    # Confirm swing if all conditions met
                    if time_gap_ok and price_distance_ok:
                        swing_point['confirmed'] = True
                        confirmed_swing_highs.append((swing_idx, swing_price, df.iloc[swing_idx].name))
                        
                        # Mark in dataframe
                        result_df.loc[swing_idx, 'swing_high'] = swing_price
                        result_df.loc[swing_idx, 'swing_high_confirmed'] = True
                        result_df.loc[swing_idx, 'swing_strength'] = swing_point['strength']
                        
                        # Update swing type
                        current_type = result_df.loc[swing_idx, 'swing_type']
                        if current_type == 'L':
                            result_df.loc[swing_idx, 'swing_type'] = 'HL'
                        elif current_type == '':
                            result_df.loc[swing_idx, 'swing_type'] = 'H'
        
        # Check potential swing lows for confirmation
        for swing_point in potential_swing_points:
            if (swing_point['type'] == 'low' and 
                swing_point['index'] < i and 
                not swing_point['confirmed']):
                
                swing_idx = swing_point['index']
                swing_price = swing_point['price']
                
                # 1. Percentage reversal check
                rally_percent = (current_high - swing_price) / swing_price
                
                if rally_percent >= threshold:
                    # 2. Time gap check
                    time_gap_ok = True
                    if confirmed_swing_lows:
                        last_swing_idx = confirmed_swing_lows[-1][0]
                        if swing_idx - last_swing_idx < min_time_gap:
                            time_gap_ok = False
                    
                    # 3. Price distance check
                    price_distance_ok = True
                    if confirmed_swing_lows:
                        last_swing_price = confirmed_swing_lows[-1][1]
                        price_diff_pct = abs(swing_price - last_swing_price) / last_swing_price
                        if price_diff_pct < min_price_distance:
                            price_distance_ok = False
                    
                    # Confirm swing if all conditions met
                    if time_gap_ok and price_distance_ok:
                        swing_point['confirmed'] = True
                        confirmed_swing_lows.append((swing_idx, swing_price, df.iloc[swing_idx].name))
                        
                        # Mark in dataframe
                        result_df.loc[swing_idx, 'swing_low'] = swing_price
                        result_df.loc[swing_idx, 'swing_low_confirmed'] = True
                        result_df.loc[swing_idx, 'swing_strength'] = swing_point['strength']
                        
                        # Update swing type
                        current_type = result_df.loc[swing_idx, 'swing_type']
                        if current_type == 'H':
                            result_df.loc[swing_idx, 'swing_type'] = 'HL'
                        elif current_type == '':
                            result_df.loc[swing_idx, 'swing_type'] = 'L'
    
    return result_df

def add_nearest_sr(df, 
                           swing_threshold=12.0,
                           lookback_window=3, 
                           lookahead_window=0,
                           min_time_gap=1, 
                           min_price_distance_pct=5.0):
    
    # Step 1: Detect enhanced swing points
    df_with_swings = detect_enhanced_swings(
        df, 
        threshold_percent=swing_threshold,
        lookback_window=lookback_window,
        lookahead_window=lookahead_window,
        min_time_gap=min_time_gap,
        min_price_distance_pct=min_price_distance_pct
    )
    
    n = len(df_with_swings)
    closes = df_with_swings['close'].values
    
    # Initialize result arrays
    latest_high = np.full(n, np.nan)
    second_latest_high = np.full(n, np.nan)
    latest_low = np.full(n, np.nan)
    second_latest_low = np.full(n, np.nan)
    trend_up = np.full(n, 0, dtype=int)
    nearest_support = np.full(n, np.nan)
    support_distance_pct = np.full(n, np.nan)
    nearest_resistance = np.full(n, np.nan)
    resistance_distance_pct = np.full(n, np.nan)
    
    # Extract confirmed swing points
    swing_highs = []  # (index, price)
    swing_lows = []   # (index, price)
    
    for i in range(n):
        if df_with_swings.loc[i, 'swing_high_confirmed']:
            swing_highs.append((i, df_with_swings.loc[i, 'swing_high']))
        if df_with_swings.loc[i, 'swing_low_confirmed']:
            swing_lows.append((i, df_with_swings.loc[i, 'swing_low']))
    
    # Step 2: For each date, calculate swing-based levels and trend
    for current_idx in range(n):
        
        # Find swing highs up to current date
        valid_highs = [(idx, price) for idx, price in swing_highs if idx <= current_idx]
        valid_highs.sort(key=lambda x: x[0], reverse=True)  # Most recent first
        
        # Find swing lows up to current date  
        valid_lows = [(idx, price) for idx, price in swing_lows if idx <= current_idx]
        valid_lows.sort(key=lambda x: x[0], reverse=True)  # Most recent first
        
        # Step 2.1: Get latest and second latest highs/lows
        if len(valid_highs) >= 1:
            latest_high[current_idx] = valid_highs[0][1]
        if len(valid_highs) >= 2:
            second_latest_high[current_idx] = valid_highs[1][1]
            
        if len(valid_lows) >= 1:
            latest_low[current_idx] = valid_lows[0][1]
        if len(valid_lows) >= 2:
            second_latest_low[current_idx] = valid_lows[1][1]
        
        # Step 2.2: Determine trend direction
        if (not np.isnan(latest_high[current_idx]) and 
            not np.isnan(second_latest_high[current_idx]) and
            not np.isnan(latest_low[current_idx]) and 
            not np.isnan(second_latest_low[current_idx])):
            
            if (latest_high[current_idx] > second_latest_high[current_idx] and 
                latest_low[current_idx] > second_latest_low[current_idx]):
                trend_up[current_idx] = 1
        
        # Step 2.3: Combine all swing levels and find nearest support/resistance
        all_levels = []
        
        # Add all swing highs up to current date
        for idx, price in valid_highs:
            all_levels.append(price)
            
        # Add all swing lows up to current date
        for idx, price in valid_lows:
            all_levels.append(price)
        
        if all_levels:
            current_close = closes[current_idx]
            
            # Find nearest support (closest level below current close)
            support_levels = [level for level in all_levels if level < current_close]
            if support_levels:
                nearest_support[current_idx] = max(support_levels)
                support_distance_pct[current_idx] = ((current_close - nearest_support[current_idx]) / current_close) * 100
            
            # Find nearest resistance (closest level above current close)
            resistance_levels = [level for level in all_levels if level > current_close]
            if resistance_levels:
                nearest_resistance[current_idx] = min(resistance_levels)
                resistance_distance_pct[current_idx] = ((nearest_resistance[current_idx] - current_close) / current_close) * 100
    
    # Step 3: Add all calculated columns to the DataFrame
    df_with_swings['latest_high'] = latest_high
    df_with_swings['second_latest_high'] = second_latest_high
    df_with_swings['latest_low'] = latest_low
    df_with_swings['second_latest_low'] = second_latest_low
    df_with_swings['trend_up'] = trend_up
    df_with_swings['nearest_support'] = nearest_support
    df_with_swings['support_pct'] = support_distance_pct      
    df_with_swings['nearest_resistance'] = nearest_resistance
    df_with_swings['resistance_pct'] = resistance_distance_pct
    
    return df_with_swings










def remove_duplicates(levels, tolerance):
    """Remove duplicate levels that are too close to each other."""
    if not levels:
        return []
    
    levels = sorted(levels, key=lambda x: x[1])
    filtered = [levels[0]]
    
    for current in levels[1:]:
        prev_price = filtered[-1][1]
        current_price = current[1]
        
        # If levels are too close, keep the one with more touches
        if abs(current_price - prev_price) / prev_price < tolerance:
            if current[2] > filtered[-1][2]:
                filtered[-1] = current
        else:
            filtered.append(current)
    
    return filtered

def plot_support_resistance(df, support_levels, resistance_levels, n_bars=100):
    """
    Plot price with clustered support and resistance levels.
    Args:
        df: DataFrame with 'close' prices.
        support_levels: List of (idx, price, touches) tuples.
        resistance_levels: List of (idx, price, touches) tuples.
        n_bars: Number of bars to plot from the end.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 7))
    plot_df = df.tail(n_bars).reset_index(drop=True)
    plt.plot(plot_df['close'], label='Close Price', color='black')
    for idx, price, touches in support_levels:
        if idx >= len(df) - n_bars:
            plt.axhline(price, color='green', linestyle='--', alpha=0.7, label='Support' if touches == support_levels[0][2] else None)
    for idx, price, touches in resistance_levels:
        if idx >= len(df) - n_bars:
            plt.axhline(price, color='red', linestyle='--', alpha=0.7, label='Resistance' if touches == resistance_levels[0][2] else None)
    plt.title('Support and Resistance Levels')
    plt.xlabel('Bar')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_dynamic_support_resistance(df, level_type='support'):
    """Get dynamic support/resistance from moving averages and trend lines."""
    dynamic_levels = []
    
    # EMAs as dynamic support/resistance
    ema_periods = [20, 50, 100, 200]
    current_close = df['close'].iloc[-1]
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        if ema_col in df.columns:
            ema_values = df[ema_col].dropna()
            if len(ema_values) > 0:
                recent_ema = ema_values.iloc[-1]
                strength = period // 10
                
                if level_type == 'support' and recent_ema < current_close:
                    dynamic_levels.append((len(df)-1, recent_ema, strength))
                elif level_type == 'resistance' and recent_ema > current_close:
                    dynamic_levels.append((len(df)-1, recent_ema, strength))
    
    # Trend line support/resistance
    trend_levels = calculate_trend_lines(df, level_type)
    dynamic_levels.extend(trend_levels)
    
    return dynamic_levels

def calculate_trend_lines(df, level_type='support', lookback=50):
    """Calculate trend lines as dynamic support/resistance."""
    if len(df) < lookback:
        return []
    
    trend_levels = []
    recent_df = df.tail(lookback)
    
    if level_type == 'support':
        # Find upward trend line through recent lows
        lows = recent_df['low'].values
        
        # Get bottom 20% of lows for trend line
        n_lows = max(2, len(lows) // 5)
        low_indices = np.argpartition(lows, n_lows)[:n_lows]
        
        if len(low_indices) >= 2:
            slope, intercept, r_value, _, _ = linregress(low_indices, lows[low_indices])
            
            # Project trend line to current position
            current_trend_level = slope * (len(lows) - 1) + intercept
            
            # Only consider as support if trend is upward and correlation is good
            if slope > 0 and r_value > 0.5:
                strength = int(abs(r_value) * 5)
                trend_levels.append((len(df)-1, current_trend_level, strength))
    
    else:  # resistance
        # Find downward trend line through recent highs
        highs = recent_df['high'].values
        
        # Get top 20% of highs for trend line
        n_highs = max(2, len(highs) // 5)
        high_indices = np.argpartition(highs, -n_highs)[-n_highs:]
        
        if len(high_indices) >= 2:
            slope, intercept, r_value, _, _ = linregress(high_indices, highs[high_indices])
            
            # Project trend line to current position
            current_trend_level = slope * (len(highs) - 1) + intercept
            
            # Only consider as resistance if trend is downward and correlation is good
            if slope < 0 and r_value > 0.5:
                strength = int(abs(r_value) * 5)
                trend_levels.append((len(df)-1, current_trend_level, strength))
    
    return trend_levels

def generate_swing_labels(df, target_pct=0.15, window=21, stop_loss_pct=0.05):
    """
    Simplified swing label generation for single target.
    
    Parameters:
    - df: DataFrame with OHLC data
    - target_pct: Target return (default 7%)
    - window: Days to look ahead (default 10)
    - stop_loss_pct: Maximum acceptable loss (default 3%)
    
    Returns:
    - DataFrame with target_hit column (1 if target hit without stop loss, 0 otherwise)
    - Additional columns: max_return, min_return, risk_reward_ratio
    """
    df = df.copy()
    n_rows = len(df)
    
    # Pre-allocate arrays for results
    target_hit = np.full(n_rows, np.nan)
    max_return = np.full(n_rows, np.nan)
    min_return = np.full(n_rows, np.nan)
    risk_reward_ratio = np.full(n_rows, np.nan)
    
    # Get price arrays for vectorized operations
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    # Process each row (except last 'window' rows)
    for i in range(n_rows - window):
        entry_price = close_prices[i]
        
        # Get future prices for the next 'window' days
        future_highs = high_prices[i+1:i+1+window]
        future_lows = low_prices[i+1:i+1+window]
        
        if len(future_highs) == 0:
            continue
            
        # Calculate returns
        max_ret = (np.max(future_highs) - entry_price) / entry_price
        min_ret = (np.min(future_lows) - entry_price) / entry_price
        
        # Store risk metrics
        max_return[i] = max_ret
        min_return[i] = min_ret
        
        # Calculate risk-reward ratio
        if min_ret < 0:
            risk_reward_ratio[i] = max_ret / abs(min_ret)
        else:
            risk_reward_ratio[i] = max_ret / 0.01  # Use small denominator if no downside
        
        # Check if target hit without exceeding stop loss
        if max_ret >= target_pct and min_ret > -stop_loss_pct:
            target_hit[i] = 1
        else:
            target_hit[i] = 0
    
    # Add results to dataframe
    df['target_hit'] = target_hit
    df['max_return'] = max_return
    df['min_return'] = min_return
    df['risk_reward_ratio'] = risk_reward_ratio
    return df

def add_advanced_swing_labels(df, window=10):
    """Add advanced swing trading labels and features."""
    df = df.copy()
    # Trend following vs mean reversion classification
    n_rows = len(df)
    trend_trade = np.full(n_rows, np.nan)
    mean_reversion_trade = np.full(n_rows, np.nan)
    breakout_trade = np.full(n_rows, np.nan)
    
    # Vectorized processing where possible
    target_hit_mask = ~pd.isna(df['target_hit'])
    valid_indices = np.where(target_hit_mask)[0]
    valid_indices = valid_indices[valid_indices < n_rows - window]
    
    for i in valid_indices:
        # Get current market state
        current_trend = df['trend_direction'].iloc[i] if 'trend_direction' in df.columns else 0
        near_support = df['in_support_zone'].iloc[i] if 'in_support_zone' in df.columns else 0
        near_resistance = df['in_resistance_zone'].iloc[i] if 'in_resistance_zone' in df.columns else 0
        
        # Classify trade type
        if current_trend == 1 and near_support:
            trend_trade[i] = 1
            mean_reversion_trade[i] = 0
            breakout_trade[i] = 0
        elif current_trend == 0 and (near_support or near_resistance):
            mean_reversion_trade[i] = 1
            trend_trade[i] = 0
            breakout_trade[i] = 0
        else:
            breakout_trade[i] = 1
            trend_trade[i] = 0
            mean_reversion_trade[i] = 0
    
    df['trend_trade'] = trend_trade
    df['mean_reversion_trade'] = mean_reversion_trade
    df['breakout_trade'] = breakout_trade
    
    # Success rate by trade type
    df['trade_type_success'] = np.nan
    
    # Calculate win rate for different market conditions
    df = add_conditional_success_rates(df)
    
    return df

def add_conditional_success_rates(df):
    """Add success rates based on market conditions."""
    df = df.copy()
    
    # Define market conditions
    conditions = {
        'high_volume': 'volume_ratio > 1.5' if 'volume_ratio' in df.columns else None,
        'low_volatility': 'volatility_5 < volatility_20' if 'volatility_5' in df.columns else None,
        'bullish_regime': 'bull_regime == 1' if 'bull_regime' in df.columns else None,
        'bearish_regime': 'bear_regime == 1' if 'bear_regime' in df.columns else None,
        'high_rsi': 'rsi > 70' if 'rsi' in df.columns else None,
        'low_rsi': 'rsi < 30' if 'rsi' in df.columns else None,
    }
    
    # Calculate success rates for each condition
    for condition_name, condition_query in conditions.items():
        if condition_query is None:
            df[f'{condition_name}_success_rate'] = 0.5
            continue
            
        try:
            condition_mask = df.query(condition_query).index
            if len(condition_mask) > 0:
                success_rate = df.loc[condition_mask, 'target_hit'].mean()
                if pd.isna(success_rate):
                    success_rate = 0.5
            else:
                success_rate = 0.5
            df[f'{condition_name}_success_rate'] = success_rate
        except:
            df[f'{condition_name}_success_rate'] = 0.5
    
    return df

def calculate_swing_quality_score(df):
    """Calculate a comprehensive swing quality score."""
    df = df.copy()
    
    # Initialize score
    swing_score = np.zeros(len(df))
    
    # Technical alignment (30% weight)
    if 'bullish_confluence' in df.columns:
        max_confluence = df['bullish_confluence'].max()
        if max_confluence > 0:
            swing_score += (df['bullish_confluence'] / max_confluence) * 0.3
    
    # Support/Resistance context (25% weight)
    if 'in_support_zone' in df.columns:
        swing_score += df['in_support_zone'].values * 0.25
    
    # Pattern strength (20% weight)
    if 'pattern_strength' in df.columns:
        max_pattern = df['pattern_strength'].max()
        if max_pattern > 0:
            swing_score += (df['pattern_strength'] / max_pattern) * 0.2
    
    # Volume confirmation (15% weight)
    if 'volume_ratio' in df.columns:
        volume_score = np.clip(df['volume_ratio'].values - 1, 0, 1)
        swing_score += volume_score * 0.15
    
    # Risk-reward potential (10% weight)
    if 'risk_reward_ratio' in df.columns:
        rr_score = np.clip(df['risk_reward_ratio'].values / 3, 0, 1)
        swing_score += rr_score * 0.1
    
    df['swing_quality_score'] = swing_score
    
    return df