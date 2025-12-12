# features/lunar_cycles.py
"""
Lunar cycle features - Studies show correlation between lunar phases and market volatility.
Full moons historically associated with increased volatility and reversal points.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import ephem

def calculate_moon_phase(date):
    """Calculate moon phase (0-1, where 0.5 is full moon)."""
    observer = ephem.Observer()
    observer.date = date
    moon = ephem.Moon(observer)
    return moon.phase / 100.0

def add_lunar_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add lunar cycle features for trading analysis."""
    
    # Calculate moon phases
    moon_phases = []
    for dt in df['time']:
        if isinstance(dt, pl.datatypes.Datetime):
            dt = dt.to_python()
        phase = calculate_moon_phase(dt)
        moon_phases.append(phase)
    
    df = df.with_columns([
        pl.Series("moon_phase", moon_phases),
    ])
    
    # Add derived lunar features
    df = df.with_columns([
        # Full moon (phase ~1.0 or ~0.0)
        ((pl.col("moon_phase") > 0.95) | (pl.col("moon_phase") < 0.05)).alias("is_full_moon"),
        
        # New moon (phase ~0.5)
        ((pl.col("moon_phase") > 0.45) & (pl.col("moon_phase") < 0.55)).alias("is_new_moon"),
        
        # Waxing vs waning
        (pl.col("moon_phase") < 0.5).alias("is_waxing"),
        
        # Distance from nearest moon extreme (full or new)
        pl.when(pl.col("moon_phase") < 0.25)
        .then(pl.col("moon_phase"))
        .when(pl.col("moon_phase") < 0.5)
        .then(0.5 - pl.col("moon_phase"))
        .when(pl.col("moon_phase") < 0.75)
        .then(pl.col("moon_phase") - 0.5)
        .otherwise(1.0 - pl.col("moon_phase"))
        .alias("moon_extreme_distance"),
        
        # Moon phase change rate (lunar momentum)
        pl.col("moon_phase").diff().alias("moon_phase_change"),
    ])
    
    # Add lunar volatility expectation
    # Studies suggest higher volatility around full/new moons
    df = df.with_columns([
        (1.0 - pl.col("moon_extreme_distance") * 4).clip(0, 1).alias("lunar_volatility_factor")
    ])
    
    return df