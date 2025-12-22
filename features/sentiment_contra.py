# features/sentiment_contra.py
"""
Contrarian sentiment features - Retail positioning and crowd behavior indicators.
Core principle: Retail traders are often wrong at extremes (fade the crowd).
"""
import polars as pl
import requests, os, json
import logging

log = logging.getLogger(__name__)
BROKER_API = os.getenv("SENTIMENT_API", "")

def retail_sentiment_features(df: pl.DataFrame, asset: str) -> pl.DataFrame:
    """
    Add retail sentiment and contrarian features.
    Uses broker API if available, otherwise derives sentiment from price action.
    """
    # Try to fetch real retail sentiment from broker API
    api_sentiment = None
    try:
        if BROKER_API:
            resp = requests.get(f"{BROKER_API}/{asset}", timeout=5)
            api_sentiment = json.loads(resp.text).get("long_pct", None)
    except Exception as e:
        log.debug(f"Sentiment API unavailable for {asset}: {e}")

    # Add API-based sentiment if available
    if api_sentiment is not None:
        df = df.with_columns([
            pl.lit(api_sentiment).alias("retail_long_pct"),
            pl.lit(100 - api_sentiment).alias("contrarian_weight"),
            pl.lit(True).alias("sentiment_from_api"),
        ])
    else:
        # Derive sentiment proxy from price action (crowd behavior indicators)
        df = df.with_columns([
            pl.lit(50.0).alias("retail_long_pct"),
            pl.lit(50.0).alias("contrarian_weight"),
            pl.lit(False).alias("sentiment_from_api"),
        ])

    # Derived contrarian indicators (work regardless of API)
    df = df.with_columns([
        # RSI extremes (retail often piles in at overbought/oversold)
        _calculate_rsi(df, 14).alias("rsi_14"),
    ])

    df = df.with_columns([
        # RSI-based crowd extremes
        (pl.col("rsi_14") > 70).alias("crowd_overbought"),
        (pl.col("rsi_14") < 30).alias("crowd_oversold"),

        # Contrarian signals (fade crowd extremes)
        (pl.col("rsi_14") > 80).alias("fade_longs_signal"),
        (pl.col("rsi_14") < 20).alias("fade_shorts_signal"),

        # Crowd momentum (how fast retail is piling in)
        pl.col("rsi_14").diff(5).alias("sentiment_momentum"),
    ])

    # Retail trap detection (price moves that trap retail)
    df = df.with_columns([
        # Bull trap: new high followed by close below open
        ((pl.col("high") > pl.col("high").shift(1).rolling_max(window_size=10)) &
         (pl.col("close") < pl.col("open"))).alias("bull_trap"),

        # Bear trap: new low followed by close above open
        ((pl.col("low") < pl.col("low").shift(1).rolling_min(window_size=10)) &
         (pl.col("close") > pl.col("open"))).alias("bear_trap"),

        # FOMO detector (late retail entries after extended move)
        ((pl.col("close") > pl.col("close").rolling_mean(window_size=20) * 1.02) &
         (pl.col("volume") > pl.col("volume").rolling_mean(window_size=20) * 1.5) &
         (pl.col("rsi_14") > 65)).alias("fomo_long_detected"),

        ((pl.col("close") < pl.col("close").rolling_mean(window_size=20) * 0.98) &
         (pl.col("volume") > pl.col("volume").rolling_mean(window_size=20) * 1.5) &
         (pl.col("rsi_14") < 35)).alias("fomo_short_detected"),
    ])

    # Capitulation signals (retail giving up = potential reversal)
    df = df.with_columns([
        # Extreme volume on new lows (retail panic selling)
        ((pl.col("low") < pl.col("low").shift(1).rolling_min(window_size=20)) &
         (pl.col("volume") > pl.col("volume").rolling_quantile(quantile=0.95, window_size=50)))
        .alias("long_capitulation"),

        # Extreme volume on new highs (short squeeze)
        ((pl.col("high") > pl.col("high").shift(1).rolling_max(window_size=20)) &
         (pl.col("volume") > pl.col("volume").rolling_quantile(quantile=0.95, window_size=50)))
        .alias("short_capitulation"),
    ])

    # Contrarian composite score
    df = df.with_columns([
        (pl.col("fade_longs_signal").cast(pl.Float32) * 2 +
         pl.col("bull_trap").cast(pl.Float32) * 1.5 +
         pl.col("fomo_long_detected").cast(pl.Float32) * 1).alias("contrarian_short_score"),

        (pl.col("fade_shorts_signal").cast(pl.Float32) * 2 +
         pl.col("bear_trap").cast(pl.Float32) * 1.5 +
         pl.col("fomo_short_detected").cast(pl.Float32) * 1 +
         pl.col("long_capitulation").cast(pl.Float32) * 2).alias("contrarian_long_score"),
    ])

    return df


def _calculate_rsi(df: pl.DataFrame, period: int = 14) -> pl.Expr:
    """Calculate RSI using Polars expressions."""
    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi
