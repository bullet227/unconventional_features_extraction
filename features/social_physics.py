# features/social_physics.py
"""
Social physics features - Apply physics concepts to market crowd behavior.
Models market participants as particles with mass, momentum, and energy.
"""

import polars as pl
import numpy as np

def add_social_physics_features(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Add features based on social physics and crowd dynamics."""
    
    # Price momentum as velocity
    df = df.with_columns([
        # Price velocity (rate of change)
        pl.col("close").diff().alias("price_velocity"),
        
        # Price acceleration (change in velocity)
        pl.col("close").diff().diff().alias("price_acceleration"),
        
        # Volume as mass (more volume = more mass/inertia)
        pl.col("volume").alias("market_mass"),
    ])
    
    # Calculate market kinetic energy (1/2 * m * v^2)
    df = df.with_columns([
        (0.5 * pl.col("market_mass") * pl.col("price_velocity")**2).alias("kinetic_energy"),
    ])
    
    # Market momentum (mass * velocity)
    df = df.with_columns([
        (pl.col("market_mass") * pl.col("price_velocity")).alias("market_momentum"),
    ])
    
    # Force indicators (F = ma)
    df = df.with_columns([
        (pl.col("market_mass") * pl.col("price_acceleration")).alias("market_force"),
    ])
    
    # Social temperature (volatility as temperature)
    df = df.with_columns([
        # Market temperature based on price oscillations
        pl.col("price_velocity").rolling_std(lookback).alias("market_temperature"),
        
        # Entropy (disorder) based on price dispersion
        (pl.col("high") - pl.col("low")).rolling_std(lookback).alias("market_entropy"),
    ])
    
    # Pressure features (volume concentration)
    df = df.with_columns([
        # Volume pressure (volume relative to average)
        (pl.col("volume") / pl.col("volume").rolling_mean(lookback)).alias("volume_pressure"),
        
        # Price compression (low volatility = high pressure)
        (1 / (pl.col("high") - pl.col("low") + 0.0001)).alias("price_compression"),
    ])
    
    # Social field strength (how strongly price attracts/repels)
    df = df.with_columns([
        # Gravitational pull to moving averages
        (pl.col("close") - pl.col("close").rolling_mean(lookback)).alias("ma_gravity"),
        (pl.col("close") - pl.col("close").rolling_mean(lookback*2)).alias("ma_gravity_long"),
        
        # Elastic force (mean reversion strength)
        ((pl.col("close") - pl.col("close").rolling_mean(lookback)) / 
         pl.col("close").rolling_std(lookback)).alias("elastic_force"),
    ])
    
    # Oscillator damping (trend exhaustion)
    df = df.with_columns([
        # Momentum decay rate
        (pl.col("market_momentum").diff() / (pl.col("market_momentum") + 1e-8)).alias("momentum_decay"),
        
        # Energy dissipation rate
        (pl.col("kinetic_energy").diff() / (pl.col("kinetic_energy") + 1e-8)).alias("energy_dissipation"),
    ])
    
    # Phase transitions (regime changes)
    df = df.with_columns([
        # Critical points where system behavior changes
        ((pl.col("market_temperature") > pl.col("market_temperature").rolling_mean(lookback*2) * 1.5) |
         (pl.col("market_entropy") > pl.col("market_entropy").rolling_mean(lookback*2) * 1.5)).alias("phase_transition"),
        
        # System stability (low when approaching phase transition)
        (1 / (pl.col("market_temperature") * pl.col("market_entropy") + 1e-8)).alias("system_stability"),
    ])
    
    # Crowd coherence (herd behavior strength)
    df = df.with_columns([
        # Price-volume correlation as coherence measure
        pl.corr("price_velocity", "market_mass").rolling_mean(lookback).alias("crowd_coherence"),
        
        # Synchronization index (how aligned are market forces)
        (pl.col("market_force").rolling_std(lookback) / 
         (pl.col("market_force").rolling_mean(lookback).abs() + 1e-8)).alias("force_synchronization"),
    ])
    
    # Potential energy (stored energy for future moves)
    df = df.with_columns([
        # Energy stored in price compression
        (pl.col("price_compression") * pl.col("market_mass")).alias("potential_energy"),
    ])

    df = df.with_columns([
        # Total mechanical energy
        (pl.col("kinetic_energy") + pl.col("potential_energy")).alias("total_energy"),
    ])

    return df