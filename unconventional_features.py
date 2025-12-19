#!/usr/bin/env python3
"""
Unconventional Feature Extraction for Forex Trading.

This module extracts 300+ unconventional/contrarian features from OHLCV data
organized into 5 tiers:
1. Foundation: Session, Stop Hunt, Sentiment
2. Behavioral/Technical: Psychology, Fibonacci, Lunar
3. Advanced Mathematical: Chaos Theory, Social Physics, Game Theory
4. Quantum & Neural: Quantum Finance, Neural Oscillations
5. Microstructure & Esoteric: HFT, Order Flow, Intracandle, Astro

Usage (standalone):
    from unconventional_features import UnconventionalFeatureExtractor

    extractor = UnconventionalFeatureExtractor()
    enriched_df = extractor.enrich(ohlcv_df, asset='EURUSD', timeframe='H1')

Usage (with database):
    python unconventional_features.py  # Runs ETL pipeline from forex_trading_data
"""
from __future__ import annotations
import os
import logging
import time
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import polars as pl

# Feature module imports
from features.session_features import add_session_stats
from features.chaos_theory import add_chaos_features
from features.social_physics import add_social_physics_features
from features.lunar_cycles import add_lunar_features
from features.image_features import candle_image_to_vector
from features.market_psychology import add_market_psychology_features
from features.fibonacci_time import add_fibonacci_time_features
from features.stop_hunt import stop_hunt_metrics
from features.sentiment_contra import retail_sentiment_features
from features.orderflow_mm_detection import add_orderflow_features
from features.intracandle_dynamics import add_intracandle_features
from features.hft_scalping import add_hft_scalping_features
from features.quantum_finance import add_quantum_features, add_quantum_field_theory_features
from features.game_theory import add_game_theory_features
from features.neural_oscillations import add_neural_oscillation_features
from features.astro_finance import add_astro_features

# Optional dependencies
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import prometheus_client as prom
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("unconventional")


class UnconventionalFeatureExtractor:
    """
    Extract unconventional features from OHLCV data.

    This class provides a clean interface for feature extraction that works
    with in-memory DataFrames, without requiring database connections.

    Example:
        extractor = UnconventionalFeatureExtractor()
        df = extractor.enrich(ohlcv_df, asset='EURUSD', timeframe='H1')
    """

    def __init__(
        self,
        enable_gpu_features: bool = False,
        tier_config: Optional[Dict[str, bool]] = None,
    ):
        """
        Initialize the feature extractor.

        Args:
            enable_gpu_features: Whether to enable GPU-accelerated image features
            tier_config: Optional dict to enable/disable feature tiers
                Example: {'tier1': True, 'tier2': True, 'tier3': False, ...}
        """
        self.enable_gpu_features = enable_gpu_features or os.getenv("ENABLE_IMAGE_FEATURES", "false").lower() == "true"

        # Default: all tiers enabled
        self.tier_config = tier_config or {
            'tier1_foundation': True,
            'tier2_behavioral': True,
            'tier3_mathematical': True,
            'tier4_quantum_neural': True,
            'tier5_microstructure': True,
            'tier5_esoteric': True,
        }

        log.info(f"Initialized extractor (GPU features: {self.enable_gpu_features})")

    def enrich(
        self,
        df: pl.DataFrame,
        asset: str = 'UNKNOWN',
        timeframe: str = 'H1',
    ) -> pl.DataFrame:
        """
        Enrich OHLCV data with unconventional features.

        Args:
            df: Polars DataFrame with columns: time, open, high, low, close, volume
            asset: Currency pair (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., 'H1', 'M15')

        Returns:
            Polars DataFrame with original columns plus extracted features
        """
        # Convert pandas to polars if needed
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

        log.info(f"Enriching {len(df)} rows for {asset}/{timeframe}")

        try:
            # Tier 1: Foundation Features
            if self.tier_config.get('tier1_foundation', True):
                df = add_session_stats(df)
                df = stop_hunt_metrics(df)
                df = retail_sentiment_features(df, asset)

            # Tier 2: Behavioral/Technical
            if self.tier_config.get('tier2_behavioral', True):
                df = add_market_psychology_features(df)
                df = add_fibonacci_time_features(df)
                df = add_lunar_features(df)

            # Tier 3: Advanced Mathematical
            if self.tier_config.get('tier3_mathematical', True):
                df = add_chaos_features(df)
                df = add_social_physics_features(df)
                df = add_game_theory_features(df)

            # Tier 4: Quantum & Neural
            if self.tier_config.get('tier4_quantum_neural', True):
                df = add_quantum_features(df)
                df = add_quantum_field_theory_features(df)
                df = add_neural_oscillation_features(df)

            # Tier 5: Microstructure
            if self.tier_config.get('tier5_microstructure', True):
                df = add_hft_scalping_features(df)
                df = add_orderflow_features(df)
                df = add_intracandle_features(df)

            # Tier 5: Esoteric
            if self.tier_config.get('tier5_esoteric', True):
                df = add_astro_features(df)

                # GPU-accelerated image features
                if self.enable_gpu_features:
                    vecs = candle_image_to_vector(df)
                    df = df.with_columns(pl.Series("img_vec", vecs))

            feature_count = len(df.columns) - 6  # Subtract OHLCV columns
            log.info(f"Extracted {feature_count} features")

            return df

        except Exception as e:
            log.error(f"Feature extraction failed: {e}")
            raise

    def get_feature_names(self, df: pl.DataFrame) -> List[str]:
        """Get list of feature column names (excluding OHLCV)."""
        ohlcv_cols = {'time', 'open', 'high', 'low', 'close', 'volume'}
        return [c for c in df.columns if c.lower() not in ohlcv_cols]

    def get_tier_features(self, df: pl.DataFrame) -> Dict[str, List[str]]:
        """Categorize features by tier/domain."""
        feature_names = self.get_feature_names(df)

        tier_mapping = {
            'tier1_session': ['session_', 'tokyo_', 'london_', 'ny_'],
            'tier1_stop_hunt': ['stop_hunt_', 'sweep_'],
            'tier1_sentiment': ['sentiment_', 'contrarian_', 'crowd_', 'fomo_', 'trap_'],
            'tier2_psychology': ['fear_', 'greed_', 'psychology_'],
            'tier2_fibonacci': ['fib_', 'golden_'],
            'tier2_lunar': ['lunar_', 'moon_'],
            'tier3_chaos': ['lyapunov_', 'entropy_', 'hurst_', 'fractal_'],
            'tier3_social': ['herding_', 'social_', 'cascade_'],
            'tier3_game': ['nash_', 'minimax_', 'game_'],
            'tier4_quantum': ['quantum_', 'tunneling_', 'superposition_'],
            'tier4_neural': ['neural_', 'brain_wave_', 'oscillation_'],
            'tier5_hft': ['hft_', 'microsecond_', 'latency_'],
            'tier5_orderflow': ['orderflow_', 'mm_', 'market_maker_', 'iceberg_'],
            'tier5_intracandle': ['intracandle_', 'wick_', 'body_'],
            'tier5_astro': ['astro_', 'mercury_', 'venus_', 'mars_', 'jupiter_'],
        }

        result = {k: [] for k in tier_mapping}
        result['other'] = []

        for feature in feature_names:
            feature_lower = feature.lower()
            found = False
            for tier, prefixes in tier_mapping.items():
                if any(prefix in feature_lower for prefix in prefixes):
                    result[tier].append(feature)
                    found = True
                    break
            if not found:
                result['other'].append(feature)

        return {k: v for k, v in result.items() if v}


# ============================================================================
# Database ETL Pipeline (only runs when executed directly)
# ============================================================================

class FeatureETLPipeline:
    """
    ETL Pipeline for extracting features from forex_trading_data and storing
    them in the features database.

    Only used when running this module as a script for batch processing.
    """

    def __init__(
        self,
        forex_db: str = None,
        target_db: str = None,
        host: str = None,
        port: str = None,
        user: str = None,
        password: str = None,
        batch_size: int = 250000,
        workers: int = 12,
    ):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy required for ETL pipeline. pip install sqlalchemy psycopg")

        self.forex_db = forex_db or os.getenv("FOREX_DB", "forex_trading_data")
        self.target_db = target_db or os.getenv("UNCONVENTIONAL_DB", "features_data")
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or os.getenv("POSTGRES_PORT", "5432")
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "")
        self.batch_size = batch_size
        self.workers = workers

        # Prometheus metrics
        self.gauge_processed = None
        self.counter_errors = None
        if PROM_AVAILABLE and os.getenv("ENABLE_PROMETHEUS", "false").lower() == "true":
            self.gauge_processed = prom.Gauge('rows_processed', 'Rows processed')
            self.counter_errors = prom.Counter('errors_total', 'Total errors')
            prom.start_http_server(8000)

        # Create engines
        self.read_eng = self._create_engine(self.forex_db)
        self.write_eng = self._create_engine(self.target_db)

        self.extractor = UnconventionalFeatureExtractor()

    def _create_engine(self, db: str, retries: int = 5):
        url = f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{db}"
        for attempt in range(retries):
            try:
                return create_engine(url, pool_pre_ping=True, pool_size=20, max_overflow=40)
            except OperationalError as e:
                if self.counter_errors:
                    self.counter_errors.inc()
                log.error(f"DB connect fail (attempt {attempt+1}): {e}")
                time.sleep(10)
        raise RuntimeError("DB connection failed")

    def stream_table(self, table: str):
        """Stream data from source table in chunks."""
        with self.read_eng.connect().execution_options(stream_results=True) as conn:
            for chunk in pd.read_sql(
                text(f"SELECT * FROM {table} ORDER BY time"),
                conn,
                chunksize=self.batch_size
            ):
                yield chunk

    def process_and_write(self, asset: str, timeframe: str):
        """Process a single asset/timeframe combination."""
        src_table = f"{asset.lower()}_{timeframe.lower()}_candles"
        dst_table = f"unconventional.{asset.lower()}_{timeframe}_features"

        log.info(f"Processing {asset} {timeframe}")

        for chunk in tqdm(self.stream_table(src_table), desc=f"{asset}_{timeframe}"):
            try:
                # Enrich
                pl_df = pl.from_pandas(chunk)
                enriched = self.extractor.enrich(pl_df, asset=asset, timeframe=timeframe)
                enriched_pd = enriched.to_pandas()

                if self.gauge_processed:
                    self.gauge_processed.inc(len(chunk))

                # Write
                if not enriched_pd.empty:
                    enriched_pd.to_sql(
                        dst_table,
                        self.write_eng,
                        schema="unconventional",
                        if_exists="append",
                        index=False,
                        method="multi",
                        chunksize=50000
                    )
                    log.info(f"Wrote {len(enriched_pd)} rows to {dst_table}")

            except Exception as e:
                if self.counter_errors:
                    self.counter_errors.inc()
                log.error(f"Processing error: {e}")

    def run(self, assets: List[str] = None, timeframes: List[str] = None):
        """Run the full ETL pipeline."""
        if assets is None:
            with self.read_eng.connect() as conn:
                result = conn.execute(text("SELECT DISTINCT instrument FROM instruments"))
                assets = [r[0] for r in result]

        if timeframes is None:
            timeframes = ["S5", "S10", "S15", "S30", "M1", "M2", "M4", "M5", "M10", "M15", "M30",
                         "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D", "W", "M"]

        tasks = [(a, t) for a in assets for t in timeframes]

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self.process_and_write, a, t) for a, t in tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    if self.counter_errors:
                        self.counter_errors.inc()
                    log.error(f"Task failed: {e}")


def main():
    """Run the ETL pipeline from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract unconventional features from forex data')
    parser.add_argument('--assets', type=str, default=None, help='Comma-separated assets (default: all)')
    parser.add_argument('--timeframes', type=str, default=None, help='Comma-separated timeframes (default: all)')
    parser.add_argument('--batch-size', type=int, default=250000, help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=12, help='Number of parallel workers')

    args = parser.parse_args()

    assets = args.assets.split(',') if args.assets else None
    timeframes = args.timeframes.split(',') if args.timeframes else None

    pipeline = FeatureETLPipeline(
        batch_size=args.batch_size,
        workers=args.workers,
    )
    pipeline.run(assets=assets, timeframes=timeframes)


if __name__ == "__main__":
    main()
