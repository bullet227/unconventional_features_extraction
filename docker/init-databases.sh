#!/bin/bash
# init-databases.sh
# Creates multiple PostgreSQL databases for the forex ML pipeline

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

# Create forex_trading_data database
create_database "forex_trading_data"

# Create unconventional_features database
create_database "unconventional_features"

# Create schema for unconventional features
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "unconventional_features" <<-EOSQL
    CREATE SCHEMA IF NOT EXISTS unconventional;

    -- Grant permissions
    GRANT ALL PRIVILEGES ON SCHEMA unconventional TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA unconventional TO $POSTGRES_USER;

    -- Create extension for better performance
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
EOSQL

echo "Database initialization complete!"
