# features/neural_oscillations.py
"""
Neural oscillation features - Market rhythms analogous to brain waves.
Models market activity as neural oscillations (delta, theta, alpha, beta, gamma).
"""

import polars as pl
import numpy as np
from scipy import signal
from typing import Tuple, List

def calculate_power_spectrum(prices: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate power spectral density using Welch's method."""
    if len(prices) < 10:
        return np.array([0]), np.array([0])

    frequencies, power = signal.welch(prices, fs=fs, nperseg=min(len(prices)//2, 128))
    return frequencies, power

def extract_band_power(frequencies: np.ndarray, power: np.ndarray,
                       low_freq: float, high_freq: float) -> float:
    """Extract power in a specific frequency band."""
    mask = (frequencies >= low_freq) & (frequencies <= high_freq)
    if not np.any(mask):
        return 0.0
    return np.mean(power[mask])

def add_neural_oscillation_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add neural oscillation features based on brain wave frequency bands.

    Frequency Bands (adapted to market timeframes):
    - Delta (0.001-0.004 Hz): Ultra-long trends (250-1000 bars)
    - Theta (0.004-0.008 Hz): Long trends (125-250 bars)
    - Alpha (0.008-0.013 Hz): Medium trends (75-125 bars)
    - Beta (0.013-0.030 Hz): Short trends (33-75 bars)
    - Gamma (0.030-0.100 Hz): High-frequency activity (10-33 bars)
    """

    df = df.with_columns([
        pl.col("close").pct_change().alias("returns"),
    ])

    # 1. DELTA WAVES (Deep sleep / Ultra-long trends)
    # Slowest oscillations, represent fundamental market cycles
    df = df.with_columns([
        # Ultra-long moving average
        pl.col("close").rolling_mean(250).alias("delta_ma"),

        # Delta trend strength
        (pl.col("close").rolling_mean(250) - pl.col("close").rolling_mean(500))
        .alias("delta_trend"),

        # Delta phase (position relative to ultra-long trend)
        ((pl.col("close") - pl.col("close").rolling_mean(250)) /
         (pl.col("close").rolling_std(250) + 1e-8)).alias("delta_phase"),
    ])

    # 2. THETA WAVES (Light sleep / Long-term trends)
    # Associated with memory consolidation in brain; market "memory" of trends
    df = df.with_columns([
        pl.col("close").rolling_mean(125).alias("theta_ma"),

        (pl.col("close").rolling_mean(125) - pl.col("close").rolling_mean(250))
        .alias("theta_trend"),

        ((pl.col("close") - pl.col("close").rolling_mean(125)) /
         (pl.col("close").rolling_std(125) + 1e-8)).alias("theta_phase"),
    ])

    # 3. ALPHA WAVES (Relaxed wakefulness / Medium trends)
    # Calm, stable trending in brain; represent normal market trending
    df = df.with_columns([
        pl.col("close").rolling_mean(100).alias("alpha_ma"),

        (pl.col("close").rolling_mean(100) - pl.col("close").rolling_mean(200))
        .alias("alpha_trend"),

        ((pl.col("close") - pl.col("close").rolling_mean(100)) /
         (pl.col("close").rolling_std(100) + 1e-8)).alias("alpha_phase"),
    ])

    # 4. BETA WAVES (Active thinking / Short-term trends)
    # Alert, focused activity in brain; active trading
    df = df.with_columns([
        pl.col("close").rolling_mean(50).alias("beta_ma"),

        (pl.col("close").rolling_mean(50) - pl.col("close").rolling_mean(100))
        .alias("beta_trend"),

        ((pl.col("close") - pl.col("close").rolling_mean(50)) /
         (pl.col("close").rolling_std(50) + 1e-8)).alias("beta_phase"),
    ])

    # 5. GAMMA WAVES (Peak focus / High-frequency activity)
    # Highest frequency, represent rapid market movements
    df = df.with_columns([
        pl.col("close").rolling_mean(20).alias("gamma_ma"),

        (pl.col("close").rolling_mean(20) - pl.col("close").rolling_mean(50))
        .alias("gamma_trend"),

        ((pl.col("close") - pl.col("close").rolling_mean(20)) /
         (pl.col("close").rolling_std(20) + 1e-8)).alias("gamma_phase"),
    ])

    # Wave coherence (synchronization between bands)
    df = df.with_columns([
        # Delta-theta coherence (long-term alignment)
        pl.corr("delta_trend", "theta_trend", ddof=0).rolling_mean(50)
        .alias("delta_theta_coherence"),

        # Alpha-beta coherence (medium-short alignment)
        pl.corr("alpha_trend", "beta_trend", ddof=0).rolling_mean(50)
        .alias("alpha_beta_coherence"),

        # Beta-gamma coherence (short-high frequency alignment)
        pl.corr("beta_trend", "gamma_trend", ddof=0).rolling_mean(20)
        .alias("beta_gamma_coherence"),
    ])

    # Overall coherence (all bands aligned)
    df = df.with_columns([
        ((pl.col("delta_theta_coherence").abs() +
          pl.col("alpha_beta_coherence").abs() +
          pl.col("beta_gamma_coherence").abs()) / 3).alias("overall_coherence"),
    ])

    # Wave dominance (which frequency band is dominant)
    df = df.with_columns([
        # Amplitude of each band
        pl.col("delta_trend").abs().alias("delta_amplitude"),
        pl.col("theta_trend").abs().alias("theta_amplitude"),
        pl.col("alpha_trend").abs().alias("alpha_amplitude"),
        pl.col("beta_trend").abs().alias("beta_amplitude"),
        pl.col("gamma_trend").abs().alias("gamma_amplitude"),
    ])

    # Dominant wave (largest amplitude)
    df = df.with_columns([
        # Delta dominant (ultra-long trend control)
        (pl.col("delta_amplitude") > pl.col("theta_amplitude")).cast(pl.Int32)
        .alias("delta_dominant"),

        # Gamma dominant (high-frequency noise)
        (pl.col("gamma_amplitude") > pl.col("beta_amplitude")).cast(pl.Int32)
        .alias("gamma_dominant"),

        # Alpha dominant (balanced, ideal trading state)
        ((pl.col("alpha_amplitude") > pl.col("delta_amplitude")) &
         (pl.col("alpha_amplitude") > pl.col("gamma_amplitude"))).cast(pl.Int32)
        .alias("alpha_dominant"),
    ])

    # Cross-frequency coupling (phase-amplitude coupling)
    # When phase of slow wave modulates amplitude of fast wave
    df = df.with_columns([
        # Theta-gamma coupling (slow phase, fast amplitude)
        (pl.col("theta_phase") * pl.col("gamma_amplitude"))
        .rolling_mean(20).alias("theta_gamma_coupling"),

        # Alpha-beta coupling
        (pl.col("alpha_phase") * pl.col("beta_amplitude"))
        .rolling_mean(20).alias("alpha_beta_coupling"),
    ])

    # Neural synchronization index (market coherence)
    df = df.with_columns([
        # High synchronization = strong trending
        (pl.col("overall_coherence") > 0.7).alias("high_synchronization"),

        # Low synchronization = choppy/ranging
        (pl.col("overall_coherence") < 0.3).alias("low_synchronization"),

        # Desynchronization (loss of coherence = trend ending)
        (pl.col("overall_coherence").diff() < -0.1).alias("desynchronization"),
    ])

    # Neural complexity (entropy across bands)
    df = df.with_columns([
        # Complexity index (diverse frequencies = complex market)
        ((pl.col("delta_amplitude") + pl.col("theta_amplitude") +
          pl.col("alpha_amplitude") + pl.col("beta_amplitude") +
          pl.col("gamma_amplitude")) / 5).alias("neural_complexity"),

        # Simplicity (single dominant frequency)
        pl.max_horizontal([
            pl.col("delta_amplitude"), pl.col("theta_amplitude"),
            pl.col("alpha_amplitude"), pl.col("beta_amplitude"),
            pl.col("gamma_amplitude")
        ]).alias("neural_simplicity"),
    ])

    df = df.with_columns([
        # Complexity ratio
        (pl.col("neural_complexity") / (pl.col("neural_simplicity") + 1e-8))
        .alias("complexity_ratio"),
    ])

    # State transitions (similar to brain state changes)
    df = df.with_columns([
        # From low to high frequency dominance
        ((pl.col("gamma_dominant").shift(1) == 0) &
         (pl.col("gamma_dominant") == 1)).alias("entering_high_frequency_state"),

        # From high to low frequency dominance
        ((pl.col("delta_dominant").shift(1) == 0) &
         (pl.col("delta_dominant") == 1)).alias("entering_low_frequency_state"),

        # Entering balanced state (alpha dominant)
        ((pl.col("alpha_dominant").shift(1) == 0) &
         (pl.col("alpha_dominant") == 1)).alias("entering_balanced_state"),
    ])

    # Clean up temporary columns
    df = df.drop([
        "delta_ma", "theta_ma", "alpha_ma", "beta_ma", "gamma_ma",
        "delta_amplitude", "theta_amplitude", "alpha_amplitude",
        "beta_amplitude", "gamma_amplitude"
    ])

    return df

def add_brainwave_entropy_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add entropy-based features inspired by neural information theory."""

    df = df.with_columns([
        pl.col("close").pct_change().alias("returns"),
    ])

    # Sample entropy (regularity/predictability)
    df = df.with_columns([
        # Price pattern regularity
        pl.col("returns").rolling_std(10) /
        (pl.col("returns").rolling_std(50) + 1e-8)
        .alias("sample_entropy"),

        # High entropy = unpredictable
        (pl.col("sample_entropy") > 1.5).alias("high_entropy"),

        # Low entropy = predictable patterns
        (pl.col("sample_entropy") < 0.5).alias("low_entropy"),
    ])

    # Neural phase locking (oscillations locked in phase)
    df = df.with_columns([
        # Fast-slow phase synchrony
        pl.corr(
            pl.col("close").rolling_mean(20).pct_change(),
            pl.col("close").rolling_mean(100).pct_change(),
            ddof=0
        ).rolling_mean(20).alias("phase_locking_value"),

        # Strong phase locking = coordinated movement
        (pl.col("phase_locking_value").abs() > 0.8).alias("strong_phase_locking"),
    ])

    return df