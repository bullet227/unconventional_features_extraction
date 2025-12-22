#!/bin/bash
# init-databases.sh
# Creates multiple PostgreSQL databases for the forex ML pipeline
#
# Database layout:
# - forex_trading_data: Source database with OHLCV candle tables (READ)
# - features_data: Target database for extracted ML features (WRITE)

set -e

# Function to create a database if it doesn't exist
create_database() {
    local database=$1
    echo "Creating database '$database' if not exists..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
        SELECT 'CREATE DATABASE $database'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
EOSQL
}

# Create forex_trading_data database (source - candles)
create_database "forex_trading_data"

# Create features_data database (target - extracted features)
create_database "features_data"

# Initialize features_data with required tables and schemas
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "features_data" <<-EOSQL
    -- Create extension for better text search performance
    CREATE EXTENSION IF NOT EXISTS pg_trgm;

    -- Create unconventional_features table for extracted features
    CREATE TABLE IF NOT EXISTS unconventional_features (
        id BIGSERIAL PRIMARY KEY,
        instrument TEXT NOT NULL,
        granularity TEXT NOT NULL,
        time TIMESTAMPTZ NOT NULL,

        -- Core OHLCV (for reference)
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume DOUBLE PRECISION,

        -- Feature columns stored as JSONB for flexibility
        features JSONB,

        -- Metadata
        created_at TIMESTAMPTZ DEFAULT NOW(),

        -- Unique constraint for upserts
        CONSTRAINT uq_unconventional_features
            UNIQUE (instrument, granularity, time)
    );

    -- Create indexes for common query patterns
    CREATE INDEX IF NOT EXISTS idx_unconventional_features_instrument
        ON unconventional_features (instrument);
    CREATE INDEX IF NOT EXISTS idx_unconventional_features_granularity
        ON unconventional_features (granularity);
    CREATE INDEX IF NOT EXISTS idx_unconventional_features_time
        ON unconventional_features (time);
    CREATE INDEX IF NOT EXISTS idx_unconventional_features_composite
        ON unconventional_features (instrument, granularity, time);

    -- Create ml_features table for ML pipeline outputs
    CREATE TABLE IF NOT EXISTS ml_features (
        id BIGSERIAL PRIMARY KEY,
        instrument TEXT NOT NULL,
        granularity TEXT NOT NULL,
        time TIMESTAMPTZ NOT NULL,
        features JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        CONSTRAINT uq_ml_features UNIQUE (instrument, granularity, time)
    );

    -- Grant permissions
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $POSTGRES_USER;
EOSQL

echo "=============================================="
echo "Database initialization complete!"
echo "=============================================="
echo "Created databases:"
echo "  - forex_trading_data (source: candles)"
echo "  - features_data (target: ML features)"
echo "=============================================="
