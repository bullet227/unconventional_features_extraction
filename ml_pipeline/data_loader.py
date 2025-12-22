# ml_pipeline/data_loader.py
"""
Data loading and storage for forex trading ML pipeline.
Handles connection to forex_trading_data and unconventional feature databases.
"""

import os
import logging
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

import polars as pl
import pandas as pd
import numpy as np

try:
    import psycopg
    from psycopg.rows import dict_row
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

log = logging.getLogger(__name__)


class DataLoader:
    """
    Load OHLCV data from forex_trading_data and store/retrieve enriched features.

    Example usage:
        loader = DataLoader()
        df = loader.load_ohlcv('EURUSD', 'H1', start='2023-01-01', end='2024-01-01')
        loader.save_features(enriched_df, 'EURUSD', 'H1')
        features_df = loader.load_features('EURUSD', 'H1')
    """

    def __init__(
        self,
        forex_db: str = None,
        unconventional_db: str = None,
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
    ):
        """
        Initialize database connections.

        Args:
            forex_db: Source database with OHLCV data
            unconventional_db: Target database for enriched features
            host, port, user, password: Connection details (fallback to env vars)
        """
        self.forex_db = forex_db or os.getenv("FOREX_DB", "forex_trading_data")
        # FEATURES_DB is the canonical env var; ML_DB and UNCONVENTIONAL_DB are legacy aliases
        self.unconventional_db = unconventional_db or os.getenv("FEATURES_DB") or os.getenv("ML_DB") or os.getenv("UNCONVENTIONAL_DB", "features_data")
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "")

        self._forex_engine = None
        self._ml_engine = None

    def _get_connection_string(self, db: str) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{db}"

    def _get_forex_engine(self):
        """Get SQLAlchemy engine for forex database."""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy required. Install with: pip install sqlalchemy")
        if self._forex_engine is None:
            self._forex_engine = create_engine(self._get_connection_string(self.forex_db))
        return self._forex_engine

    def _get_ml_engine(self):
        """Get SQLAlchemy engine for ML/unconventional database."""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy required. Install with: pip install sqlalchemy")
        if self._ml_engine is None:
            self._ml_engine = create_engine(self._get_connection_string(self.unconventional_db))
        return self._ml_engine

    def get_available_assets(self) -> List[str]:
        """Get list of available currency pairs from the database."""
        engine = self._get_forex_engine()
        query = """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE '%_candles'
            ORDER BY table_name
        """
        with engine.connect() as conn:
            result = conn.execute(text(query))
            tables = [row[0] for row in result]

        # Extract unique assets (e.g., 'eurusd_h1_candles' -> 'EURUSD')
        assets = set()
        for table in tables:
            parts = table.split('_')
            if len(parts) >= 2:
                assets.add(parts[0].upper())

        return sorted(list(assets))

    def get_available_timeframes(self, asset: str) -> List[str]:
        """Get available timeframes for an asset."""
        engine = self._get_forex_engine()
        asset_lower = asset.lower()
        query = f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE '{asset_lower}_%_candles'
            ORDER BY table_name
        """
        with engine.connect() as conn:
            result = conn.execute(text(query))
            tables = [row[0] for row in result]

        # Extract timeframes
        timeframes = []
        for table in tables:
            parts = table.replace('_candles', '').split('_')
            if len(parts) >= 2:
                timeframes.append(parts[1].upper())

        return timeframes

    def load_ohlcv(
        self,
        asset: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Load OHLCV data from forex database.

        Args:
            asset: Currency pair (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., 'H1', 'M15', 'D')
            start: Start date (ISO format)
            end: End date (ISO format)
            limit: Max rows to return

        Returns:
            Polars DataFrame with time, open, high, low, close, volume columns
        """
        table_name = f"{asset.lower()}_{timeframe.lower()}_candles"
        engine = self._get_forex_engine()

        # Build query
        query = f"SELECT time, open, high, low, close, volume FROM {table_name}"
        conditions = []

        if start:
            conditions.append(f"time >= '{start}'")
        if end:
            conditions.append(f"time <= '{end}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY time ASC"

        if limit:
            query += f" LIMIT {limit}"

        log.info(f"Loading OHLCV: {asset}/{timeframe} [{start} to {end}]")

        # Load with pandas then convert to polars for better compatibility
        df_pd = pd.read_sql(query, engine)
        df = pl.from_pandas(df_pd)

        log.info(f"Loaded {len(df)} rows")
        return df

    def save_features(
        self,
        df: pl.DataFrame,
        asset: str,
        timeframe: str,
        if_exists: str = 'replace',
    ) -> None:
        """
        Save enriched features to the unconventional database.

        Args:
            df: DataFrame with features
            asset: Currency pair
            timeframe: Timeframe
            if_exists: 'replace', 'append', or 'fail'
        """
        table_name = f"{asset.lower()}_{timeframe.lower()}_features"
        engine = self._get_ml_engine()

        # Convert to pandas for SQLAlchemy compatibility
        if isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df

        log.info(f"Saving {len(df_pd)} rows to {table_name}")
        df_pd.to_sql(table_name, engine, if_exists=if_exists, index=False)
        log.info(f"Saved successfully")

    def load_features(
        self,
        asset: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Load enriched features from the unconventional database.

        Args:
            asset: Currency pair
            timeframe: Timeframe
            start: Start date
            end: End date
            columns: Specific columns to load (None for all)

        Returns:
            Polars DataFrame with features
        """
        table_name = f"{asset.lower()}_{timeframe.lower()}_features"
        engine = self._get_ml_engine()

        # Build column selection
        col_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {col_str} FROM {table_name}"

        conditions = []
        if start:
            conditions.append(f"time >= '{start}'")
        if end:
            conditions.append(f"time <= '{end}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY time ASC"

        log.info(f"Loading features: {asset}/{timeframe}")
        df_pd = pd.read_sql(query, engine)
        df = pl.from_pandas(df_pd)

        log.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df

    def create_ml_dataset(
        self,
        asset: str,
        timeframe: str,
        target_type: str = 'direction',
        target_horizon: int = 1,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Tuple[pl.DataFrame, pl.Series]:
        """
        Create ML-ready dataset with features and target.

        Args:
            asset: Currency pair
            timeframe: Timeframe
            target_type: 'direction' (up/down), 'returns', or 'volatility'
            target_horizon: Number of periods ahead for target
            start: Start date
            end: End date

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = self.load_features(asset, timeframe, start, end)

        # Create target based on type
        if target_type == 'direction':
            # Binary: 1 if price goes up, 0 if down
            target = (df['close'].shift(-target_horizon) > df['close']).cast(pl.Int32)
        elif target_type == 'returns':
            # Continuous: percentage return
            target = (df['close'].shift(-target_horizon) - df['close']) / df['close'] * 100
        elif target_type == 'volatility':
            # Continuous: future volatility (using high-low range as proxy)
            future_range = (df['high'].shift(-target_horizon) - df['low'].shift(-target_horizon))
            target = future_range / df['close'] * 100
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        # Remove target-related columns from features
        feature_cols = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]
        features = df.select(feature_cols)

        # Remove rows with NaN target (last N rows due to shift)
        valid_mask = target.is_not_null()
        features = features.filter(valid_mask)
        target = target.filter(valid_mask)

        log.info(f"Created dataset: {len(features)} samples, {len(feature_cols)} features, target={target_type}")

        return features, target


def create_sample_data(n_rows: int = 1000) -> pl.DataFrame:
    """
    Create sample OHLCV data for testing without database.

    Args:
        n_rows: Number of rows to generate

    Returns:
        Polars DataFrame with synthetic OHLCV data
    """
    np.random.seed(42)

    dates = [datetime(2023, 1, 1, 9, 0) + timedelta(hours=i) for i in range(n_rows)]
    base_price = 1.1000
    prices = base_price + np.cumsum(np.random.randn(n_rows) * 0.0005)

    highs = prices + np.abs(np.random.randn(n_rows) * 0.0003)
    lows = prices - np.abs(np.random.randn(n_rows) * 0.0003)
    opens = prices + np.random.randn(n_rows) * 0.0001
    closes = prices + np.random.randn(n_rows) * 0.0001
    volumes = np.random.randint(100, 10000, n_rows).astype(float)

    return pl.DataFrame({
        'time': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })
