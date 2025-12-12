# features/stop_hunt.py
import polars as pl
import numpy as np

def stop_hunt_metrics(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Detect potential stop hunts and liquidity sweeps."""
    
    # Calculate wicks and body
    df = df.with_columns([
        # Upper and lower wicks
        (pl.col("high") - pl.max_horizontal([pl.col("open"), pl.col("close")])).alias("upper_wick"),
        (pl.min_horizontal([pl.col("open"), pl.col("close")]) - pl.col("low")).alias("lower_wick"),
        
        # Candle body
        (pl.col("close") - pl.col("open")).abs().alias("body_size"),
        
        # Total range
        (pl.col("high") - pl.col("low")).alias("range"),
    ])
    
    # Wick ratios and z-scores
    df = df.with_columns([
        # Wick to body ratio (large wicks = stop hunts)
        (pl.col("upper_wick") / (pl.col("body_size") + 1e-8)).alias("upper_wick_ratio"),
        (pl.col("lower_wick") / (pl.col("body_size") + 1e-8)).alias("lower_wick_ratio"),
        
        # Wick size relative to average range
        (pl.col("upper_wick") / (pl.col("range").rolling_mean(lookback) + 1e-8)).alias("upper_wick_rel"),
        (pl.col("lower_wick") / (pl.col("range").rolling_mean(lookback) + 1e-8)).alias("lower_wick_rel"),
    ])
    
    # Calculate z-scores for anomaly detection
    df = df.with_columns([
        # Z-scores for wick sizes
        ((pl.col("upper_wick") - pl.col("upper_wick").rolling_mean(lookback)) / 
         (pl.col("upper_wick").rolling_std(lookback) + 1e-8)).alias("upper_wick_z"),
        
        ((pl.col("lower_wick") - pl.col("lower_wick").rolling_mean(lookback)) / 
         (pl.col("lower_wick").rolling_std(lookback) + 1e-8)).alias("lower_wick_z"),
        
        # Volume spike detection
        ((pl.col("volume") - pl.col("volume").rolling_mean(lookback)) / 
         (pl.col("volume").rolling_std(lookback) + 1e-8)).alias("volume_z"),
    ])
    
    # Stop hunt flags
    df = df.with_columns([
        # Upper stop hunt: large upper wick + volume spike
        ((pl.col("upper_wick_z") > 2.5) & (pl.col("volume_z") > 1.5)).alias("upper_stop_hunt"),
        
        # Lower stop hunt: large lower wick + volume spike  
        ((pl.col("lower_wick_z") > 2.5) & (pl.col("volume_z") > 1.5)).alias("lower_stop_hunt"),
        
        # Liquidity sweep: price briefly exceeds recent high/low
        (pl.col("high") > pl.col("high").rolling_max(lookback).shift(1)).alias("high_sweep"),
        (pl.col("low") < pl.col("low").rolling_min(lookback).shift(1)).alias("low_sweep"),
    ])
    
    return df