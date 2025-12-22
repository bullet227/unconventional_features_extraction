# features/market_psychology.py
"""
Market psychology features - Behavioral finance and crowd psychology indicators.
Models fear, greed, capitulation, and euphoria cycles.
"""

import polars as pl
import numpy as np

def add_market_psychology_features(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Add psychological and behavioral features."""

    # Basic price psychology
    df = df.with_columns([
        # Price relative to recent range (0=bottom, 1=top)
        ((pl.col("close") - pl.col("low").rolling_min(window_size=lookback)) /
         (pl.col("high").rolling_max(window_size=lookback) - pl.col("low").rolling_min(window_size=lookback) + 1e-8))
        .alias("price_position"),

        # Distance from round numbers (psychological levels)
        (pl.col("close") % 0.01).alias("cents_distance"),  # Distance from .00
        (pl.col("close") % 0.005).alias("half_cent_distance"),  # Distance from .x0 or .x5
    ])

    # Fear and Greed indicators
    df = df.with_columns([
        # Fear: rapid selling with high volume
        ((pl.col("close") < pl.col("open")) &
         (pl.col("volume") > pl.col("volume").rolling_mean(window_size=lookback) * 1.5) &
         (pl.col("close").pct_change() < -0.005)).alias("fear_spike"),

        # Greed: rapid buying with high volume
        ((pl.col("close") > pl.col("open")) &
         (pl.col("volume") > pl.col("volume").rolling_mean(window_size=lookback) * 1.5) &
         (pl.col("close").pct_change() > 0.005)).alias("greed_spike"),
    ])

    # Cumulative fear/greed index
    df = df.with_columns([
        pl.col("fear_spike").cast(pl.Int32).rolling_sum(window_size=lookback).alias("fear_index"),
        pl.col("greed_spike").cast(pl.Int32).rolling_sum(window_size=lookback).alias("greed_index"),
    ])

    # Market mood (fear-greed balance)
    df = df.with_columns([
        (pl.col("greed_index") - pl.col("fear_index")).alias("market_mood"),

        # Mood volatility (emotional instability)
        (pl.col("greed_index") - pl.col("fear_index")).rolling_std(window_size=lookback)
        .alias("mood_volatility"),
    ])

    # Capitulation and Euphoria detection
    df = df.with_columns([
        # Capitulation: extreme fear + volume spike + oversold
        ((pl.col("fear_index") > lookback * 0.7) &
         (pl.col("price_position") < 0.2) &
         (pl.col("volume") > pl.col("volume").rolling_max(window_size=lookback*2) * 0.8))
        .alias("capitulation_signal"),

        # Euphoria: extreme greed + volume spike + overbought
        ((pl.col("greed_index") > lookback * 0.7) &
         (pl.col("price_position") > 0.8) &
         (pl.col("volume") > pl.col("volume").rolling_max(window_size=lookback*2) * 0.8))
        .alias("euphoria_signal"),
    ])

    # Herding behavior (everyone moving together)
    df = df.with_columns([
        # Directional consensus (low volatility + trend)
        (pl.col("close").rolling_std(window_size=lookback) /
         (pl.col("close").rolling_mean(window_size=lookback) + 1e-8)).alias("price_dispersion"),

        # Volume consensus (everyone trading)
        (pl.col("volume") / pl.col("volume").rolling_mean(window_size=lookback*3))
        .alias("volume_consensus"),
    ])

    df = df.with_columns([
        # Herd strength: low dispersion + high volume = strong herding
        ((1 / (pl.col("price_dispersion") + 0.01)) * pl.col("volume_consensus"))
        .alias("herd_strength"),
    ])

    # Regret and Relief patterns
    df = df.with_columns([
        # Regret: price moves away after flat period (missed opportunity)
        ((pl.col("close").rolling_std(window_size=5) < pl.col("close").rolling_std(window_size=lookback) * 0.5) &
         (pl.col("close").diff(5).abs() > pl.col("close").rolling_std(window_size=lookback) * 2))
        .alias("regret_pattern"),

        # Relief: price stabilizes after volatile period
        ((pl.col("close").rolling_std(window_size=5) < pl.col("close").rolling_std(window_size=lookback) * 0.5) &
         (pl.col("close").shift(5).rolling_std(window_size=5) > pl.col("close").rolling_std(window_size=lookback) * 1.5))
        .alias("relief_pattern"),
    ])

    # Anchoring bias (sticky prices)
    df = df.with_columns([
        # Time spent near recent high/low (anchoring)
        ((pl.col("close") - pl.col("high").rolling_max(window_size=lookback)).abs() <
         pl.col("close") * 0.001).rolling_sum(window_size=lookback).alias("high_anchoring"),

        ((pl.col("close") - pl.col("low").rolling_min(window_size=lookback)).abs() <
         pl.col("close") * 0.001).rolling_sum(window_size=lookback).alias("low_anchoring"),
    ])

    # Overreaction and Underreaction
    df = df.with_columns([
        # Overreaction: large move followed by reversal
        ((pl.col("close").pct_change().abs() > 0.01) &
         (pl.col("close").pct_change() * pl.col("close").pct_change().shift(-1) < 0))
        .alias("overreaction"),

        # Underreaction: small moves in trending market
        ((pl.col("close").pct_change().abs() < 0.002) &
         ((pl.col("close").rolling_mean(window_size=5) - pl.col("close").rolling_mean(window_size=20)).abs() >
          pl.col("close") * 0.005))
        .alias("underreaction"),
    ])

    # Cognitive load (complexity causing paralysis)
    df = df.with_columns([
        # High volatility + no clear trend = confusion
        ((pl.col("close").rolling_std(window_size=lookback) > pl.col("close").rolling_std(window_size=lookback*3) * 1.5) &
         ((pl.col("close").rolling_mean(window_size=5) - pl.col("close").rolling_mean(window_size=20)).abs() <
          pl.col("close") * 0.002))
        .alias("market_confusion"),

        # Decision fatigue (too many reversals)
        ((pl.col("close") > pl.col("open")) != (pl.col("close").shift(1) > pl.col("open").shift(1)))
        .rolling_sum(window_size=lookback).alias("decision_fatigue"),
    ])

    # Loss aversion patterns
    df = df.with_columns([
        # Holding losers: low volume during drawdown
        ((pl.col("close") < pl.col("close").rolling_max(window_size=lookback) * 0.95) &
         (pl.col("volume") < pl.col("volume").rolling_mean(window_size=lookback) * 0.8))
        .alias("loss_aversion"),

        # Profit taking: volume spike near resistance
        ((pl.col("close") > pl.col("close").rolling_max(window_size=lookback) * 0.98) &
         (pl.col("volume") > pl.col("volume").rolling_mean(window_size=lookback) * 1.2))
        .alias("profit_taking"),
    ])

    return df
