#!/usr/bin/env python3
"""
Audit script for candle tables in forex_trading_data.

This script:
1. Lists every *_candles table in forex_trading_data
2. Parses table names into (instrument, granularity) pairs
3. Compares against expected instruments/granularities
4. Reports unexpected tables, missing tables, and parse failures

Usage:
    python scripts/audit_candle_tables.py [--strict]

Options:
    --strict    Exit with error code if any issues found
"""
from __future__ import annotations
import os
import sys
import argparse
from typing import Set, Tuple, List, Dict
from dataclasses import dataclass

try:
    from sqlalchemy import create_engine, text
except ImportError:
    print("ERROR: SQLAlchemy required. Install with: pip install sqlalchemy psycopg[binary]")
    sys.exit(1)


# Expected instruments (77 forex pairs + metals + indices)
EXPECTED_INSTRUMENTS = {
    # Major pairs
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    # Minor pairs
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "CADJPY", "CADCHF", "NZDJPY", "NZDCHF", "NZDCAD", "CHFJPY",
    # Exotic pairs (commonly traded)
    "USDHKD", "USDSGD", "USDZAR", "USDMXN", "USDTRY", "USDPLN", "USDSEK", "USDNOK", "USDDKK",
    "EURHKD", "EURSGD", "EURZAR", "EURMXN", "EURTRY", "EURPLN", "EURSEK", "EURNOK", "EURDKK",
    "GBPHKD", "GBPSGD", "GBPZAR", "GBPMXN", "GBPTRY", "GBPPLN", "GBPSEK", "GBPNOK",
    # Metals
    "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
    # Indices (CFDs)
    "US30", "US500", "NAS100", "UK100", "DE30", "FR40", "JP225", "AU200",
    # Crypto (if supported)
    "BTCUSD", "ETHUSD",
}

# Expected granularities (21 OANDA timeframes)
EXPECTED_GRANULARITIES = {
    "S5", "S10", "S15", "S30",  # Seconds
    "M1", "M2", "M4", "M5", "M10", "M15", "M30",  # Minutes
    "H1", "H2", "H3", "H4", "H6", "H8", "H12",  # Hours
    "D", "W", "M",  # Daily, Weekly, Monthly
}


@dataclass
class AuditResult:
    """Results from candle table audit."""
    total_tables: int
    parsed_tables: List[Tuple[str, str, str]]  # (table_name, instrument, granularity)
    parse_failures: List[str]  # Tables that couldn't be parsed
    unexpected_instruments: Set[str]
    missing_instruments: Set[str]
    unexpected_granularities: Set[str]
    missing_granularities: Set[str]
    duplicate_pairs: List[Tuple[str, str]]  # (instrument, granularity) pairs with duplicates

    @property
    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return bool(
            self.parse_failures or
            self.unexpected_instruments or
            self.unexpected_granularities or
            self.duplicate_pairs
        )


def get_db_connection():
    """Create database connection from environment."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    db = os.getenv("FOREX_DB", "forex_trading_data")

    dsn = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
    return create_engine(dsn, pool_pre_ping=True)


def list_candle_tables(engine) -> List[str]:
    """Get all candle tables from the database."""
    query = text(
        "SELECT tablename FROM pg_tables "
        "WHERE schemaname = 'public' AND tablename LIKE '%\\_candles' ESCAPE '\\' "
        "ORDER BY tablename"
    )
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result]


def parse_table_name(table_name: str) -> Tuple[str, str] | None:
    """
    Parse a candle table name into (instrument, granularity).

    Expected format: {instrument}_{granularity}_candles
    Examples:
        eurusd_h1_candles -> (EURUSD, H1)
        xauusd_m15_candles -> (XAUUSD, M15)
    """
    if not table_name.endswith("_candles"):
        return None

    # Remove _candles suffix
    name = table_name[:-8]  # len("_candles") = 8

    # Split by underscore from the right (granularity is last)
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return None

    instrument = parts[0].upper()
    granularity = parts[1].upper()

    return (instrument, granularity)


def audit_tables(tables: List[str]) -> AuditResult:
    """Perform full audit of candle tables."""
    parsed_tables = []
    parse_failures = []
    found_instruments = set()
    found_granularities = set()
    pair_counts: Dict[Tuple[str, str], int] = {}

    for table in tables:
        parsed = parse_table_name(table)
        if parsed is None:
            parse_failures.append(table)
        else:
            instrument, granularity = parsed
            parsed_tables.append((table, instrument, granularity))
            found_instruments.add(instrument)
            found_granularities.add(granularity)

            # Track duplicates
            pair = (instrument, granularity)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Find unexpected/missing
    unexpected_instruments = found_instruments - EXPECTED_INSTRUMENTS
    missing_instruments = EXPECTED_INSTRUMENTS - found_instruments
    unexpected_granularities = found_granularities - EXPECTED_GRANULARITIES
    missing_granularities = EXPECTED_GRANULARITIES - found_granularities

    # Find duplicates
    duplicates = [pair for pair, count in pair_counts.items() if count > 1]

    return AuditResult(
        total_tables=len(tables),
        parsed_tables=parsed_tables,
        parse_failures=parse_failures,
        unexpected_instruments=unexpected_instruments,
        missing_instruments=missing_instruments,
        unexpected_granularities=unexpected_granularities,
        missing_granularities=missing_granularities,
        duplicate_pairs=duplicates,
    )


def print_report(result: AuditResult) -> None:
    """Print the audit report."""
    print("=" * 70)
    print("CANDLE TABLES AUDIT REPORT")
    print("=" * 70)

    print(f"\nTotal candle tables found: {result.total_tables}")
    print(f"Successfully parsed: {len(result.parsed_tables)}")
    print(f"Parse failures: {len(result.parse_failures)}")

    # Unique instruments and granularities
    instruments = {p[1] for p in result.parsed_tables}
    granularities = {p[2] for p in result.parsed_tables}
    print(f"\nUnique instruments: {len(instruments)}")
    print(f"Unique granularities: {len(granularities)}")

    # Expected coverage
    expected_total = len(EXPECTED_INSTRUMENTS) * len(EXPECTED_GRANULARITIES)
    print(f"\nExpected combinations: {expected_total}")
    print(f"Actual combinations: {len(result.parsed_tables)}")

    # Issues
    if result.parse_failures:
        print(f"\n⚠️  PARSE FAILURES ({len(result.parse_failures)}):")
        for table in result.parse_failures[:10]:
            print(f"    - {table}")
        if len(result.parse_failures) > 10:
            print(f"    ... and {len(result.parse_failures) - 10} more")

    if result.unexpected_instruments:
        print(f"\n⚠️  UNEXPECTED INSTRUMENTS ({len(result.unexpected_instruments)}):")
        for inst in sorted(result.unexpected_instruments)[:10]:
            print(f"    - {inst}")
        if len(result.unexpected_instruments) > 10:
            print(f"    ... and {len(result.unexpected_instruments) - 10} more")

    if result.missing_instruments:
        print(f"\nℹ️  MISSING INSTRUMENTS ({len(result.missing_instruments)}):")
        for inst in sorted(result.missing_instruments)[:10]:
            print(f"    - {inst}")
        if len(result.missing_instruments) > 10:
            print(f"    ... and {len(result.missing_instruments) - 10} more")

    if result.unexpected_granularities:
        print(f"\n⚠️  UNEXPECTED GRANULARITIES ({len(result.unexpected_granularities)}):")
        for gran in sorted(result.unexpected_granularities):
            print(f"    - {gran}")

    if result.missing_granularities:
        print(f"\nℹ️  MISSING GRANULARITIES ({len(result.missing_granularities)}):")
        for gran in sorted(result.missing_granularities):
            print(f"    - {gran}")

    if result.duplicate_pairs:
        print(f"\n⚠️  DUPLICATE PAIRS ({len(result.duplicate_pairs)}):")
        for inst, gran in result.duplicate_pairs[:10]:
            print(f"    - {inst}_{gran}")

    # Summary
    print("\n" + "=" * 70)
    if result.has_issues:
        print("AUDIT RESULT: ISSUES FOUND")
    else:
        print("AUDIT RESULT: PASSED ✓")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Audit candle tables in forex_trading_data")
    parser.add_argument("--strict", action="store_true", help="Exit with error if issues found")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    try:
        engine = get_db_connection()
        tables = list_candle_tables(engine)

        if not tables:
            print("ERROR: No candle tables found in database")
            sys.exit(1)

        result = audit_tables(tables)

        if args.json:
            import json
            output = {
                "total_tables": result.total_tables,
                "parsed_count": len(result.parsed_tables),
                "parse_failures": result.parse_failures,
                "unexpected_instruments": list(result.unexpected_instruments),
                "missing_instruments": list(result.missing_instruments),
                "unexpected_granularities": list(result.unexpected_granularities),
                "missing_granularities": list(result.missing_granularities),
                "duplicate_pairs": result.duplicate_pairs,
                "has_issues": result.has_issues,
            }
            print(json.dumps(output, indent=2))
        else:
            print_report(result)

        if args.strict and result.has_issues:
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
