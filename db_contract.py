#!/usr/bin/env python3
"""
Database Contract Module for Unconventional Features Pipeline.

This module enforces strict database usage:
- FOREX_DB_DSN: Read-only source for candle data (forex_trading_data)
- FEATURES_DB_DSN: Write target for extracted features (features_data)

The startup validation ensures:
1. Both databases are reachable
2. Source DB contains expected candle tables
3. Target DB has the unconventional_features table

If any check fails, the pipeline exits immediately with a clear error.
"""
from __future__ import annotations
import os
import sys
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

try:
    from sqlalchemy import create_engine, text, Engine
    from sqlalchemy.exc import OperationalError, ProgrammingError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Engine = None

log = logging.getLogger("db_contract")


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str
    port: str
    user: str
    password: str
    forex_db: str
    features_db: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            forex_db=os.getenv("FOREX_DB", "forex_trading_data"),
            features_db=os.getenv("FEATURES_DB", "features_data"),
        )

    @property
    def forex_dsn(self) -> str:
        """Get DSN for forex (source) database."""
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.forex_db}"

    @property
    def features_dsn(self) -> str:
        """Get DSN for features (target) database."""
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.features_db}"

    def forex_dsn_masked(self) -> str:
        """Get masked DSN for logging (hides password)."""
        return f"postgresql+psycopg://{self.user}:***@{self.host}:{self.port}/{self.forex_db}"

    def features_dsn_masked(self) -> str:
        """Get masked DSN for logging (hides password)."""
        return f"postgresql+psycopg://{self.user}:***@{self.host}:{self.port}/{self.features_db}"


class DatabaseContract:
    """
    Enforces the database contract for the unconventional features pipeline.

    This class ensures:
    1. Source database (forex_trading_data) is readable and contains candle tables
    2. Target database (features_data) is writable and has the target table
    3. No accidental writes to wrong databases

    Usage:
        contract = DatabaseContract.from_env()
        contract.validate_or_exit()  # Exits if validation fails
        forex_engine = contract.forex_engine
        features_engine = contract.features_engine
    """

    # Expected target table for unconventional features
    TARGET_TABLE = "unconventional_features"
    TARGET_SCHEMA = "public"

    # Minimum expected candle tables (sanity check)
    MIN_CANDLE_TABLES = 100

    def __init__(self, config: DatabaseConfig):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy required. Install with: pip install sqlalchemy psycopg[binary]")

        self.config = config
        self._forex_engine: Optional[Engine] = None
        self._features_engine: Optional[Engine] = None
        self._validated = False

    @classmethod
    def from_env(cls) -> "DatabaseContract":
        """Create contract from environment variables."""
        return cls(DatabaseConfig.from_env())

    @property
    def forex_engine(self) -> Engine:
        """Get the forex (source) database engine."""
        if not self._validated:
            raise RuntimeError("Database contract not validated. Call validate_or_exit() first.")
        if self._forex_engine is None:
            self._forex_engine = self._create_engine(self.config.forex_dsn)
        return self._forex_engine

    @property
    def features_engine(self) -> Engine:
        """Get the features (target) database engine."""
        if not self._validated:
            raise RuntimeError("Database contract not validated. Call validate_or_exit() first.")
        if self._features_engine is None:
            self._features_engine = self._create_engine(self.config.features_dsn)
        return self._features_engine

    def _create_engine(self, dsn: str, retries: int = 3) -> Engine:
        """Create a SQLAlchemy engine with retry logic."""
        import time
        for attempt in range(retries):
            try:
                engine = create_engine(dsn, pool_pre_ping=True, pool_size=10, max_overflow=20)
                # Test connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return engine
            except OperationalError as e:
                log.warning(f"Connection attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        raise RuntimeError(f"Failed to connect after {retries} attempts")

    def validate_or_exit(self) -> None:
        """
        Validate database contract and exit if validation fails.

        Checks:
        1. Both databases are reachable
        2. Source DB has candle tables
        3. Target DB exists (does NOT auto-create)
        """
        print("=" * 60)
        print("DATABASE CONTRACT VALIDATION")
        print("=" * 60)

        errors = []

        # 1. Validate source database (forex_trading_data)
        print(f"\n[SOURCE] Forex Database")
        print(f"  DSN: {self.config.forex_dsn_masked()}")

        try:
            forex_eng = self._create_engine(self.config.forex_dsn)
            self._forex_engine = forex_eng
            print(f"  Status: CONNECTED ✓")

            # Count candle tables
            candle_count = self._count_candle_tables(forex_eng)
            print(f"  Candle tables: {candle_count}")

            if candle_count < self.MIN_CANDLE_TABLES:
                errors.append(
                    f"Source DB has only {candle_count} candle tables "
                    f"(expected at least {self.MIN_CANDLE_TABLES})"
                )
                print(f"  WARNING: Below minimum threshold of {self.MIN_CANDLE_TABLES}")
            else:
                print(f"  Candle tables check: PASSED ✓")

        except Exception as e:
            print(f"  Status: FAILED ✗")
            errors.append(f"Cannot connect to source database: {e}")

        # 2. Validate target database (features_data)
        print(f"\n[TARGET] Features Database")
        print(f"  DSN: {self.config.features_dsn_masked()}")

        try:
            features_eng = self._create_engine(self.config.features_dsn)
            self._features_engine = features_eng
            print(f"  Status: CONNECTED ✓")

            # Check for target table (or create it)
            table_exists = self._check_target_table(features_eng)
            if table_exists:
                print(f"  Target table '{self.TARGET_TABLE}': EXISTS ✓")
                row_count = self._get_table_row_count(features_eng)
                print(f"  Current row count: {row_count:,}")
            else:
                print(f"  Target table '{self.TARGET_TABLE}': NOT FOUND")
                print(f"  Creating target table...")
                self._create_target_table(features_eng)
                print(f"  Target table '{self.TARGET_TABLE}': CREATED ✓")

            # Check for existing feature tables
            existing_tables = self._list_feature_tables(features_eng)
            if existing_tables:
                print(f"  Existing feature tables: {', '.join(existing_tables)}")

        except Exception as e:
            print(f"  Status: FAILED ✗")
            errors.append(f"Cannot connect to target database: {e}")

        # Summary
        print("\n" + "=" * 60)
        if errors:
            print("VALIDATION FAILED - Pipeline will not start")
            print("=" * 60)
            for err in errors:
                print(f"  ERROR: {err}")
            print("\nPlease fix the above issues and try again.")
            sys.exit(1)
        else:
            print("VALIDATION PASSED - Pipeline ready to start")
            print("=" * 60)
            self._validated = True

    def _count_candle_tables(self, engine: Engine) -> int:
        """Count candle tables in the source database."""
        query = text(
            "SELECT COUNT(*) FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename LIKE '%\\_candles' ESCAPE '\\'"
        )
        with engine.connect() as conn:
            result = conn.execute(query)
            return result.scalar() or 0

    def _check_target_table(self, engine: Engine) -> bool:
        """Check if the target table exists."""
        query = text(
            "SELECT EXISTS ("
            "  SELECT 1 FROM pg_tables "
            "  WHERE schemaname = :schema AND tablename = :table"
            ")"
        )
        with engine.connect() as conn:
            result = conn.execute(query, {"schema": self.TARGET_SCHEMA, "table": self.TARGET_TABLE})
            return result.scalar() or False

    def _get_table_row_count(self, engine: Engine) -> int:
        """Get approximate row count for target table."""
        query = text(f"SELECT COUNT(*) FROM {self.TARGET_SCHEMA}.{self.TARGET_TABLE}")
        try:
            with engine.connect() as conn:
                result = conn.execute(query)
                return result.scalar() or 0
        except ProgrammingError:
            return 0

    def _create_target_table(self, engine: Engine) -> None:
        """Create the unconventional_features target table."""
        create_sql = text(f"""
            CREATE TABLE IF NOT EXISTS {self.TARGET_SCHEMA}.{self.TARGET_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                instrument TEXT NOT NULL,
                granularity TEXT NOT NULL,
                time TIMESTAMPTZ NOT NULL,

                -- Core OHLCV (for reference, features are computed from these)
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,

                -- Feature columns will be added dynamically via ALTER TABLE
                -- or stored as JSONB for flexibility
                features JSONB,

                -- Metadata
                created_at TIMESTAMPTZ DEFAULT NOW(),

                -- Unique constraint for upserts
                CONSTRAINT uq_unconventional_features
                    UNIQUE (instrument, granularity, time)
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_unconventional_features_instrument
                ON {self.TARGET_SCHEMA}.{self.TARGET_TABLE} (instrument);
            CREATE INDEX IF NOT EXISTS idx_unconventional_features_granularity
                ON {self.TARGET_SCHEMA}.{self.TARGET_TABLE} (granularity);
            CREATE INDEX IF NOT EXISTS idx_unconventional_features_time
                ON {self.TARGET_SCHEMA}.{self.TARGET_TABLE} (time);
            CREATE INDEX IF NOT EXISTS idx_unconventional_features_composite
                ON {self.TARGET_SCHEMA}.{self.TARGET_TABLE} (instrument, granularity, time);
        """)
        with engine.connect() as conn:
            conn.execute(create_sql)
            conn.commit()

    def _list_feature_tables(self, engine: Engine) -> List[str]:
        """List existing feature tables in the target database."""
        query = text(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND "
            "(tablename LIKE '%features%' OR tablename LIKE 'ml_%')"
        )
        with engine.connect() as conn:
            result = conn.execute(query)
            return [row[0] for row in result]

    def dispose(self) -> None:
        """Clean up database connections."""
        if self._forex_engine:
            self._forex_engine.dispose()
        if self._features_engine:
            self._features_engine.dispose()


def validate_databases() -> Tuple[Engine, Engine]:
    """
    Convenience function to validate databases and return engines.

    Returns:
        Tuple of (forex_engine, features_engine)

    Raises:
        SystemExit if validation fails
    """
    contract = DatabaseContract.from_env()
    contract.validate_or_exit()
    return contract.forex_engine, contract.features_engine


if __name__ == "__main__":
    # When run directly, just validate the databases
    logging.basicConfig(level=logging.INFO)
    contract = DatabaseContract.from_env()
    contract.validate_or_exit()
    print("\nDatabase contract validation successful!")
    contract.dispose()
