# features/session_features.py
import polars as pl
import numpy as np

def add_session_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Add trading session features and kill-zone indicators."""
    
    # Fix session hours (UTC times)
    df = df.with_columns([
        # Asian session (Tokyo: 00:00 - 09:00 UTC)
        ((pl.col("time").dt.hour() >= 0) & (pl.col("time").dt.hour() < 9)).alias("is_asian"),
        
        # London session (07:00 - 16:00 UTC) 
        ((pl.col("time").dt.hour() >= 7) & (pl.col("time").dt.hour() < 16)).alias("is_london"),
        
        # NY session (12:00 - 21:00 UTC)
        ((pl.col("time").dt.hour() >= 12) & (pl.col("time").dt.hour() < 21)).alias("is_ny"),
        
        # Kill zones (high volatility periods)
        ((pl.col("time").dt.hour() == 8) | (pl.col("time").dt.hour() == 12)).alias("is_killzone"),
        
        # Session overlaps (most volatile)
        ((pl.col("time").dt.hour() >= 7) & (pl.col("time").dt.hour() < 9)).alias("asian_london_overlap"),
        ((pl.col("time").dt.hour() >= 12) & (pl.col("time").dt.hour() < 16)).alias("london_ny_overlap"),
        
        # Day of week features
        pl.col("time").dt.weekday().alias("weekday"),
        (pl.col("time").dt.weekday() == 5).alias("is_friday"),  # NFP, options expiry
        (pl.col("time").dt.weekday() == 1).alias("is_monday"),  # Weekend gaps
    ])
    
    # Session price ranges
    df = df.with_columns([
        # Intraday range percentage
        ((pl.col("high") - pl.col("low")) / pl.col("open") * 100).alias("range_pct"),
        
        # Distance from daily open
        ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).alias("open_distance_pct"),
    ])
    
    return df