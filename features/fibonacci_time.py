# features/fibonacci_time.py
"""
Fibonacci time-based features - Markets often show reversals at Fibonacci time intervals.
Tracks time elapsed since significant highs/lows in Fibonacci ratios.
"""

import polars as pl
import numpy as np

# Fibonacci ratios and numbers
FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0, 2.618, 3.236, 4.236]
FIB_NUMBERS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

def add_fibonacci_time_features(df: pl.DataFrame, lookback: int = 250) -> pl.DataFrame:
    """Add Fibonacci time-based features."""
    
    # Find swing highs and lows
    df = df.with_columns([
        # Swing high: higher than N bars on each side
        ((pl.col("high") == pl.col("high").rolling_max(5)) & 
         (pl.col("high") == pl.col("high").shift(-5).rolling_max(5))).alias("is_swing_high"),
        
        # Swing low: lower than N bars on each side  
        ((pl.col("low") == pl.col("low").rolling_min(5)) &
         (pl.col("low") == pl.col("low").shift(-5).rolling_min(5))).alias("is_swing_low"),
    ])
    
    # Calculate bars since last significant high/low
    df = df.with_columns([
        # Bars since last swing high
        pl.when(pl.col("is_swing_high"))
        .then(0)
        .otherwise(pl.arange(0, len(df)))
        .alias("bars_since_high"),
        
        # Bars since last swing low
        pl.when(pl.col("is_swing_low"))
        .then(0)
        .otherwise(pl.arange(0, len(df)))
        .alias("bars_since_low"),
    ])
    
    # Forward fill to get cumulative count
    df = df.with_columns([
        pl.col("bars_since_high").cumsum().over(pl.col("is_swing_high").cumsum()).alias("bars_since_high"),
        pl.col("bars_since_low").cumsum().over(pl.col("is_swing_low").cumsum()).alias("bars_since_low"),
    ])
    
    # Check if current bar is at Fibonacci time interval
    for fib_num in FIB_NUMBERS[:10]:  # Use first 10 Fibonacci numbers
        df = df.with_columns([
            # High-to-current Fibonacci time
            ((pl.col("bars_since_high") % fib_num) == 0).alias(f"fib_time_high_{fib_num}"),
            
            # Low-to-current Fibonacci time
            ((pl.col("bars_since_low") % fib_num) == 0).alias(f"fib_time_low_{fib_num}"),
        ])
    
    # Add Fibonacci ratio-based time projections
    df = df.with_columns([
        # Time symmetry features
        (pl.col("bars_since_high") / (pl.col("bars_since_low") + 1)).alias("high_low_time_ratio"),
        
        # Check if we're near any Fibonacci ratio of time
        pl.lit(False).alias("near_fib_time")  # Initialize
    ])
    
    # Mark bars near Fibonacci time ratios
    for ratio in FIB_RATIOS:
        df = df.with_columns([
            # Within 2% of Fibonacci ratio
            ((pl.col("high_low_time_ratio") > ratio * 0.98) & 
             (pl.col("high_low_time_ratio") < ratio * 1.02)).alias(f"near_fib_ratio_{int(ratio*1000)}"),
        ])
        
        # Update the aggregate flag
        df = df.with_columns([
            pl.col("near_fib_time") | pl.col(f"near_fib_ratio_{int(ratio*1000)}").alias("near_fib_time")
        ])
    
    # Add time cycle features
    df = df.with_columns([
        # Fibonacci spiral time (logarithmic time perception)
        (pl.col("bars_since_high").log() * 1.618).alias("fib_spiral_high"),
        (pl.col("bars_since_low").log() * 1.618).alias("fib_spiral_low"),
        
        # Time acceleration/deceleration
        (pl.col("bars_since_high") / (pl.col("bars_since_low") + 1)).diff().alias("time_acceleration"),
    ])
    
    return df