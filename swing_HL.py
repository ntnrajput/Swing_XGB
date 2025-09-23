import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import HISTORICAL_DATA_FILE

def detect_enhanced_swings(df, 
                          threshold_percent=100.0, 
                          lookback_window=7, 
                          lookahead_window=0,
                          min_time_gap=7, 
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
                        confirmed_swing_highs.append((swing_idx, swing_price, df.iloc[swing_idx]['date']))
                        
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
                        confirmed_swing_lows.append((swing_idx, swing_price, df.iloc[swing_idx]['date']))
                        
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

def plot_enhanced_swings(df_with_swings, title="Enhanced Swing Detection"):
    """
    Plot with enhanced swing visualization showing strength and filters.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
    
    # Reset index for clean plotting
    df_plot = df_with_swings.reset_index(drop=True)
    x_axis = range(len(df_plot))
    
    # Main price chart
    ax1.plot(x_axis, df_plot['high'], 'r-', alpha=0.6, linewidth=1, label='High', zorder=1)
    ax1.plot(x_axis, df_plot['low'], 'g-', alpha=0.6, linewidth=1, label='Low', zorder=1) 
    ax1.plot(x_axis, df_plot['close'], 'b-', linewidth=2, label='Close', zorder=2)
    
    # Get swing data
    swing_highs = df_plot[df_plot['swing_high_confirmed'] == True]
    swing_lows = df_plot[df_plot['swing_low_confirmed'] == True]
    
    # Plot swing points with strength-based sizing
    if not swing_highs.empty:
        # Size markers based on strength
        sizes = swing_highs['swing_strength'] * 20 + 100  # Base size 100, +20 per strength
        colors = ['darkred'] * len(swing_highs)
        
        ax1.scatter(swing_highs.index, swing_highs['swing_high'], 
                   s=sizes, c=colors, marker='^', zorder=6, 
                   label=f'Swing Highs ({len(swing_highs)})', 
                   edgecolor='white', linewidth=2, alpha=0.8)
        
        # Add horizontal lines
        for idx, row in swing_highs.iterrows():
            ax1.axhline(y=row['swing_high'], color='red', linestyle='--', alpha=0.5, 
                       linewidth=1, zorder=3)
            
            # Strength annotation
            ax1.annotate(f'H:{row["swing_high"]:.1f}\n(S:{row["swing_strength"]:.0f})', 
                        (idx, row['swing_high']), 
                        xytext=(0, 25), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, 
                        color='darkred', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    if not swing_lows.empty:
        # Size markers based on strength
        sizes = swing_lows['swing_strength'] * 20 + 100
        colors = ['darkgreen'] * len(swing_lows)
        
        ax1.scatter(swing_lows.index, swing_lows['swing_low'], 
                   s=sizes, c=colors, marker='v', zorder=6,
                   label=f'Swing Lows ({len(swing_lows)})', 
                   edgecolor='white', linewidth=2, alpha=0.8)
        
        # Add horizontal lines  
        for idx, row in swing_lows.iterrows():
            ax1.axhline(y=row['swing_low'], color='green', linestyle='--', alpha=0.5, 
                       linewidth=1, zorder=3)
            
            # Strength annotation
            ax1.annotate(f'L:{row["swing_low"]:.1f}\n(S:{row["swing_strength"]:.0f})', 
                        (idx, row['swing_low']), 
                        xytext=(0, -30), textcoords='offset points',
                        ha='center', va='top', fontsize=8, 
                        color='darkgreen', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Main chart styling
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Swing strength subplot
    ax2.bar(x_axis, df_plot['swing_strength'], alpha=0.6, color='orange', label='Swing Strength')
    ax2.set_ylabel('Strength', fontsize=10)
    ax2.set_xlabel('Time Period', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis dates for both subplots
    n_ticks = min(15, len(df_plot))
    step = max(1, len(df_plot) // n_ticks)
    tick_positions = list(range(0, len(df_plot), step))
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        if 'date' in df_plot.columns:
            tick_labels = []
            for i in tick_positions:
                date_val = df_plot.iloc[i]['date']
                try:
                    if pd.isna(date_val):
                        tick_labels.append(f'Index {i}')
                    elif hasattr(date_val, 'strftime'):
                        tick_labels.append(date_val.strftime('%Y-%m-%d'))
                    else:
                        tick_labels.append(str(date_val))
                except (ValueError, AttributeError):
                    tick_labels.append(f'Index {i}')
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Enhanced summary
    total_swings = len(swing_highs) + len(swing_lows)
    avg_strength = df_plot[df_plot['swing_strength'] > 0]['swing_strength'].mean() if total_swings > 0 else 0
    
    summary_text = f'Enhanced Swings: {total_swings}\n'
    summary_text += f'Highs: {len(swing_highs)} | Lows: {len(swing_lows)}\n'
    summary_text += f'Avg Strength: {avg_strength:.1f}'
    
    ax1.text(0.02, 0.98, summary_text, 
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def analyze_enhanced_summary(df_with_swings, 
                            threshold_percent, 
                            lookback_window, 
                            lookahead_window, 
                            min_time_gap, 
                            min_price_distance_pct):
    """
    Comprehensive analysis of enhanced swing detection.
    """
    swing_highs = df_with_swings[df_with_swings['swing_high_confirmed'] == True]
    swing_lows = df_with_swings[df_with_swings['swing_low_confirmed'] == True]
    
    print("="*80)
    print("ENHANCED SWING DETECTION ANALYSIS")
    print("="*80)
    print("PARAMETERS USED:")
    print(f"  Reversal Threshold: {threshold_percent}%")
    print(f"  Lookback Window: {lookback_window} bars")
    print(f"  Lookahead Window: {lookahead_window} bars") 
    print(f"  Min Time Gap: {min_time_gap} bars")
    print(f"  Min Price Distance: {min_price_distance_pct}%")
    print()
    
    print("RESULTS:")
    print(f"  Total Data Points: {len(df_with_swings)}")
    print(f"  Date Range: {df_with_swings['date'].iloc[0]} to {df_with_swings['date'].iloc[-1]}")
    print(f"  Swing Highs: {len(swing_highs)}")
    print(f"  Swing Lows: {len(swing_lows)}")
    print(f"  Total Swings: {len(swing_highs) + len(swing_lows)}")
    
    if len(swing_highs) > 0 or len(swing_lows) > 0:
        all_strengths = df_with_swings[df_with_swings['swing_strength'] > 0]['swing_strength']
        print(f"  Average Strength: {all_strengths.mean():.2f}")
        print(f"  Max Strength: {all_strengths.max():.0f}")
        print(f"  Min Strength: {all_strengths.min():.0f}")
    
    # Time gap analysis
    if len(swing_highs) > 1:
        high_gaps = np.diff([idx for idx in swing_highs.index])
        print(f"  Avg Gap Between Highs: {high_gaps.mean():.1f} bars")
        
    if len(swing_lows) > 1:
        low_gaps = np.diff([idx for idx in swing_lows.index])
        print(f"  Avg Gap Between Lows: {low_gaps.mean():.1f} bars")
    
    print("\nDETAILED SWING LIST:")
    print("-" * 80)
    print(f"{'Date':<12} {'Type':<6} {'Price':<10} {'Strength':<10} {'Gap':<8}")
    print("-" * 80)
    
    # Combine all swings and sort chronologically
    all_swings = []
    
    for idx, row in swing_highs.iterrows():
        all_swings.append({
            'index': idx,
            'date': row['date'],
            'type': 'HIGH',
            'price': row['swing_high'],
            'strength': row['swing_strength']
        })
    
    for idx, row in swing_lows.iterrows():
        all_swings.append({
            'index': idx,
            'date': row['date'], 
            'type': 'LOW',
            'price': row['swing_low'],
            'strength': row['swing_strength']
        })
    
    # Sort by index (chronological)
    all_swings.sort(key=lambda x: x['index'])
    
    prev_index = None
    for swing in all_swings:
        date_str = str(swing['date'])[:10]
        gap = swing['index'] - prev_index if prev_index is not None else 0
        print(f"{date_str:<12} {swing['type']:<6} {swing['price']:<10.2f} {swing['strength']:<10.0f} {gap:<8.0f}")
        prev_index = swing['index']
    
    print("="*80)

# Example usage - replace your main section with this:
if __name__ == "__main__":
    # Your data loading code
    df = pd.read_parquet(HISTORICAL_DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['symbol']=='NSE:SWSOLAR-EQ'].copy()
    df = df.reset_index(drop=True)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Enhanced swing detection with strict parameters
    print("Running enhanced swing detection...")
    df_with_swings = detect_enhanced_swings(
        df, 
        threshold_percent=12.0,        # 5% reversal still required
        lookback_window=3,            # Must be extreme in last 3 bars  
        lookahead_window=0,           # Must stay extreme for next 3 bars
        min_time_gap=1,               # At least 8 bars between swings
        min_price_distance_pct=5   # At least 3% difference from previous swing
    )
    
    # Analysis
    analyze_enhanced_summary(df_with_swings, 100.0, 3, 0, 7, 5.0)
    
    # Plotting
    fig, (ax1, ax2) = plot_enhanced_swings(df_with_swings, 
                                          "NSE:SWSOLAR-EQ - Enhanced Swing Detection")
    plt.show()
    
    # Compare different parameter sets
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
    param_sets = [
        (5.0, 2, 0, 5, 2.0),   # More sensitive
        (5.0, 3, 0, 8, 3.0),   # Medium (default)
        (7.0, 5, 0, 12, 4.0),  # Less sensitive
    ]
    
    for params in param_sets:
        thresh, lb, la, gap, dist = params
        df_test = detect_enhanced_swings(df, thresh, lb, la, gap, dist)
        highs = len(df_test[df_test['swing_high_confirmed'] == True])
        lows = len(df_test[df_test['swing_low_confirmed'] == True])
        print(f"Params({thresh}%, {lb}/{la}, {gap}gap, {dist}%dist): {highs+lows:2d} swings ({highs}H, {lows}L)")