# features/chaos_theory.py
"""
Chaos theory features - Markets as chaotic systems with strange attractors.
Includes fractal dimensions, Lyapunov exponents, and phase space analysis.
"""

import polars as pl
import numpy as np
from scipy import stats

def calculate_hurst_exponent(series, lags=range(2, 100)):
    """Calculate Hurst exponent for persistence/anti-persistence."""
    if len(series) < 100:
        return 0.5  # Neutral
    
    tau = []
    lagvec = []
    
    for lag in lags:
        if lag >= len(series):
            break
        tau.append(np.std(np.subtract(series[lag:], series[:-lag])))
        lagvec.append(lag)
    
    if len(tau) < 2:
        return 0.5
    
    # Fit power law
    log_tau = np.log(tau)
    log_lag = np.log(lagvec)
    popt = np.polyfit(log_lag, log_tau, 1)
    hurst = popt[0]
    
    return hurst

def add_chaos_features(df: pl.DataFrame, embed_dim: int = 3, tau: int = 1) -> pl.DataFrame:
    """Add chaos theory based features."""
    
    # Price returns for analysis
    df = df.with_columns([
        pl.col("close").pct_change().alias("returns"),
        pl.col("close").log().diff().alias("log_returns"),
    ])
    
    # Fractal dimension (measure of price roughness)
    df = df.with_columns([
        # Box-counting dimension approximation
        ((pl.col("high") - pl.col("low")).log() / 
         (pl.col("volume").log() + 1)).rolling_mean(20).alias("fractal_dimension"),
    ])
    
    # Phase space reconstruction (Takens embedding)
    # Create lagged versions for phase space
    for i in range(embed_dim):
        df = df.with_columns([
            pl.col("returns").shift(i * tau).alias(f"phase_{i}")
        ])
    
    # Phase space metrics
    phase_cols = [f"phase_{i}" for i in range(embed_dim)]
    df = df.with_columns([
        # Phase space velocity (rate of change in phase space)
        pl.sum_horizontal([pl.col(c).diff()**2 for c in phase_cols]).sqrt()
        .alias("phase_velocity"),

        # Phase space expansion rate
        pl.sum_horizontal([pl.col(c) for c in phase_cols]).diff()
        .alias("phase_expansion"),
    ])
    
    # Recurrence features (how often system returns to similar states)
    df = df.with_columns([
        # Simple recurrence: price returns to similar level
        ((pl.col("close") - pl.col("close").shift(20)).abs() < 
         pl.col("close").rolling_std(20) * 0.1).alias("price_recurrence"),
        
        # Pattern recurrence: similar price patterns
        pl.corr("returns", pl.col("returns").shift(20)).rolling_mean(10)
        .alias("pattern_recurrence"),
    ])
    
    # Lyapunov-like features (sensitivity to initial conditions)
    df = df.with_columns([
        # Local divergence rate
        (pl.col("returns").diff().abs() / 
         (pl.col("returns").abs() + 1e-8)).rolling_mean(10)
        .alias("local_lyapunov"),
        
        # Trajectory divergence
        (pl.col("close") - pl.col("close").shift(1)).abs().rolling_std(20)
        .alias("trajectory_divergence"),
    ])
    
    # Strange attractor features
    df = df.with_columns([
        # Distance from multiple moving averages (attractors)
        ((pl.col("close") - pl.col("close").rolling_mean(10))**2 +
         (pl.col("close") - pl.col("close").rolling_mean(20))**2 +
         (pl.col("close") - pl.col("close").rolling_mean(50))**2).sqrt()
        .alias("attractor_distance"),
    ])

    df = df.with_columns([
        # Attractor strength (inverse of distance)
        (1 / (pl.col("attractor_distance") + 1e-8)).alias("attractor_strength"),
    ])
    
    # Information entropy
    df = df.with_columns([
        # Shannon entropy of returns distribution
        pl.col("returns").rolling_quantile(0.25, window_size=20).alias("q25"),
        pl.col("returns").rolling_quantile(0.75, window_size=20).alias("q75"),
    ])
    
    df = df.with_columns([
        # Entropy proxy based on interquartile range
        ((pl.col("q75") - pl.col("q25")) / 
         (pl.col("returns").rolling_std(20) + 1e-8)).alias("entropy_ratio"),
    ])
    
    # Bifurcation indicators (system about to change behavior)
    df = df.with_columns([
        # Variance ratio test for regime change
        (pl.col("returns").rolling_var(10) / 
         (pl.col("returns").rolling_var(50) + 1e-8)).alias("variance_ratio"),
        
        # Critical slowing down (precursor to bifurcation)
        pl.col("returns").rolling_quantile(0.5, window_size=20)
        .diff().abs().rolling_mean(10).alias("critical_slowing"),
    ])
    
    # Deterministic vs stochastic behavior
    df = df.with_columns([
        # Approximate entropy (regularity measure)
        (pl.col("returns").rolling_std(5) /
        (pl.col("returns").rolling_std(20) + 1e-8))
        .alias("approximate_entropy"),
    ])

    df = df.with_columns([
        # Predictability index
        (1 - pl.col("approximate_entropy")).clip(0, 1).alias("predictability"),
    ])
    
    # Clean up temporary columns
    for i in range(embed_dim):
        df = df.drop(f"phase_{i}")
    df = df.drop(["q25", "q75"])
    
    return df