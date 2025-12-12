# features/astro_finance.py
"""
Astro-financial features - Planetary alignments and solar activity correlations.
Based on research showing statistical correlations between celestial events and market behavior.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import ephem

def calculate_planetary_positions(date) -> dict:
    """Calculate positions of major planets."""
    observer = ephem.Observer()
    observer.date = date

    positions = {}
    planets = {
        'sun': ephem.Sun(),
        'mercury': ephem.Mercury(),
        'venus': ephem.Venus(),
        'mars': ephem.Mars(),
        'jupiter': ephem.Jupiter(),
        'saturn': ephem.Saturn(),
    }

    for name, planet in planets.items():
        planet.compute(observer)
        # Get ecliptic longitude (0-360 degrees)
        positions[name] = float(planet.hlon) * 180 / np.pi

    return positions

def calculate_aspects(pos1: float, pos2: float) -> dict:
    """
    Calculate astrological aspects (angular relationships).
    Aspects: conjunction(0°), sextile(60°), square(90°), trine(120°), opposition(180°)
    """
    diff = abs(pos1 - pos2) % 360
    diff = min(diff, 360 - diff)  # Take smaller angle

    aspects = {
        'conjunction': abs(diff - 0) < 8,    # ±8° orb
        'sextile': abs(diff - 60) < 6,       # ±6° orb
        'square': abs(diff - 90) < 8,        # ±8° orb (traditionally bearish)
        'trine': abs(diff - 120) < 8,        # ±8° orb (traditionally bullish)
        'opposition': abs(diff - 180) < 8,   # ±8° orb (traditionally volatile)
    }

    return aspects

def add_astro_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add astro-financial features based on planetary positions."""

    # Calculate planetary positions for each timestamp
    planetary_data = []
    for dt in df['time']:
        if isinstance(dt, pl.datatypes.Datetime):
            dt = dt.to_python()

        positions = calculate_planetary_positions(dt)

        # Calculate key aspects
        # Sun-Jupiter (expansion, optimism)
        sun_jupiter = calculate_aspects(positions['sun'], positions['jupiter'])

        # Sun-Saturn (contraction, fear)
        sun_saturn = calculate_aspects(positions['sun'], positions['saturn'])

        # Mercury retrograde zone (communication disruption)
        mercury_speed = 1.0  # Simplified; actual calculation requires ephemeris velocity

        planetary_data.append({
            'mercury_longitude': positions['mercury'],
            'venus_longitude': positions['venus'],
            'mars_longitude': positions['mars'],
            'jupiter_longitude': positions['jupiter'],
            'saturn_longitude': positions['saturn'],
            'sun_jupiter_trine': sun_jupiter['trine'],
            'sun_jupiter_square': sun_jupiter['square'],
            'sun_saturn_square': sun_saturn['square'],
            'sun_saturn_opposition': sun_saturn['opposition'],
        })

    # Add to dataframe
    planet_df = pl.DataFrame(planetary_data)
    df = pl.concat([df, planet_df], how="horizontal")

    # Derived astro features
    df = df.with_columns([
        # Bullish planetary alignment (Jupiter aspects)
        pl.col("sun_jupiter_trine").cast(pl.Int32).alias("bullish_alignment"),

        # Bearish planetary alignment (Saturn aspects)
        (pl.col("sun_saturn_square").cast(pl.Int32) |
         pl.col("sun_saturn_opposition").cast(pl.Int32)).alias("bearish_alignment"),

        # Volatile planetary alignment (squares and oppositions)
        (pl.col("sun_jupiter_square").cast(pl.Int32) |
         pl.col("sun_saturn_square").cast(pl.Int32)).alias("volatile_alignment"),
    ])

    # Planetary cycles
    df = df.with_columns([
        # Mercury cycle (short-term, ~88 days)
        (pl.col("mercury_longitude") / 360.0).alias("mercury_cycle"),

        # Venus cycle (sentiment, ~225 days)
        (pl.col("venus_longitude") / 360.0).alias("venus_cycle"),

        # Mars cycle (energy, ~687 days)
        (pl.col("mars_longitude") / 360.0).alias("mars_cycle"),

        # Jupiter cycle (expansion, ~12 years)
        (pl.col("jupiter_longitude") / 360.0).alias("jupiter_cycle"),

        # Saturn cycle (contraction, ~29 years)
        (pl.col("saturn_longitude") / 360.0).alias("saturn_cycle"),
    ])

    # Cycle harmonics (when cycles align)
    df = df.with_columns([
        # Fast-slow harmony (Mercury-Jupiter)
        (pl.col("mercury_cycle") - pl.col("jupiter_cycle")).abs()
        .alias("mercury_jupiter_harmony"),

        # Expansion-contraction balance (Jupiter-Saturn)
        (pl.col("jupiter_cycle") - pl.col("saturn_cycle")).abs()
        .alias("jupiter_saturn_balance"),
    ])

    # Historical correlation zones
    df = df.with_columns([
        # Jupiter in "expansion" zone (historically bullish)
        ((pl.col("jupiter_longitude") > 0) &
         (pl.col("jupiter_longitude") < 120)).alias("jupiter_expansion_zone"),

        # Saturn in "contraction" zone (historically bearish)
        ((pl.col("saturn_longitude") > 180) &
         (pl.col("saturn_longitude") < 300)).alias("saturn_contraction_zone"),
    ])

    # Planetary strength index (combined bullish/bearish signals)
    df = df.with_columns([
        (pl.col("bullish_alignment").cast(pl.Int32) -
         pl.col("bearish_alignment").cast(pl.Int32)).alias("planetary_strength"),
    ])

    return df

def add_solar_activity_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add solar activity features (sunspots, solar flares).
    Requires external data source for actual solar activity.
    This is a template showing the structure.
    """

    # Placeholder for solar activity data
    # In production, fetch from NOAA/NASA APIs
    df = df.with_columns([
        # Solar cycle (11-year cycle)
        # 0 = solar minimum (low activity), 1 = solar maximum (high activity)
        pl.lit(0.5).alias("solar_cycle_phase"),

        # Sunspot number (proxy for solar activity)
        pl.lit(100.0).alias("sunspot_number"),

        # Geomagnetic activity (Kp index)
        pl.lit(3.0).alias("geomagnetic_index"),
    ])

    # Derived solar features
    df = df.with_columns([
        # High solar activity (historically correlated with volatility)
        (pl.col("sunspot_number") > 150).alias("high_solar_activity"),

        # Geomagnetic storm (Kp > 5)
        (pl.col("geomagnetic_index") > 5).alias("geomagnetic_storm"),

        # Solar cycle position (early, mid, late)
        (pl.col("solar_cycle_phase") * 3).floor().alias("solar_cycle_stage"),
    ])

    return df

def add_eclipse_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add solar and lunar eclipse features."""

    # Calculate eclipses (simplified approach)
    eclipse_data = []
    for dt in df['time']:
        if isinstance(dt, pl.datatypes.Datetime):
            dt = dt.to_python()

        # Check for eclipse proximity (within ±14 days)
        # This is simplified; actual eclipse calculation requires detailed ephemeris
        observer = ephem.Observer()
        observer.date = dt

        sun = ephem.Sun(observer)
        moon = ephem.Moon(observer)

        # Eclipses occur when sun and moon are aligned (same longitude)
        sun_lon = float(sun.hlon) * 180 / np.pi
        moon_lon = float(moon.hlon) * 180 / np.pi

        alignment = abs(sun_lon - moon_lon) % 360
        alignment = min(alignment, 360 - alignment)

        eclipse_data.append({
            'eclipse_proximity': alignment < 15,  # Within eclipse zone
            'eclipse_distance': alignment,
        })

    eclipse_df = pl.DataFrame(eclipse_data)
    df = pl.concat([df, eclipse_df], how="horizontal")

    # Eclipse windows (historically volatile periods)
    df = df.with_columns([
        # Pre-eclipse tension (5 days before)
        pl.col("eclipse_proximity").alias("eclipse_window"),

        # Eclipse strength (closer = stronger effect)
        (1 / (pl.col("eclipse_distance") + 1)).alias("eclipse_strength"),
    ])

    return df