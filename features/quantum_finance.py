# features/quantum_finance.py
"""
Quantum-inspired financial features - Applies quantum mechanics concepts to market behavior.
Includes superposition states, entanglement correlation, wave-particle duality, and uncertainty principle.
"""

import polars as pl
import numpy as np
from typing import Tuple

def calculate_wave_function(prices: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    """
    Calculate market wave function (price-momentum superposition).
    Inspired by quantum wave functions psi(x,p).
    """
    # Normalize
    p_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
    m_norm = (momentum - np.mean(momentum)) / (np.std(momentum) + 1e-8)

    # Wave function as complex superposition
    wave = np.exp(1j * p_norm) * np.exp(-1j * m_norm)
    return np.abs(wave)

def add_quantum_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add quantum-inspired market features."""

    # Price and momentum for quantum calculations
    df = df.with_columns([
        pl.col("close").pct_change().alias("returns"),
        pl.col("close").diff().alias("momentum"),
    ])

    # 1. WAVE-PARTICLE DUALITY
    # Markets exhibit both wave (trending) and particle (discrete) behavior
    df = df.with_columns([
        # Wave behavior: smooth trends
        pl.col("close").rolling_mean(20).alias("wave_component"),

        # Particle behavior: discrete jumps
        (pl.col("close") - pl.col("close").shift(1)).abs()
        .rolling_sum(20).alias("particle_component"),
    ])

    df = df.with_columns([
        # Duality ratio: wave vs particle dominance
        (pl.col("wave_component") / (pl.col("particle_component") + 1e-8))
        .alias("wave_particle_ratio"),

        # Complementarity: high in one means low in other
        (1 / (1 + pl.col("wave_particle_ratio"))).alias("particle_probability"),
        (pl.col("wave_particle_ratio") / (1 + pl.col("wave_particle_ratio")))
        .alias("wave_probability"),
    ])

    # 2. HEISENBERG UNCERTAINTY PRINCIPLE
    # Cannot know both price and momentum with perfect precision
    df = df.with_columns([
        # Price uncertainty (standard deviation)
        pl.col("close").rolling_std(10).alias("price_uncertainty"),

        # Momentum uncertainty
        pl.col("momentum").rolling_std(10).alias("momentum_uncertainty"),
    ])

    df = df.with_columns([
        # Uncertainty product (should have a minimum value)
        (pl.col("price_uncertainty") * pl.col("momentum_uncertainty"))
        .alias("uncertainty_product"),

        # When uncertainty is low, precision is high (good for trading)
        (1 / (pl.col("uncertainty_product") + 1e-8)).alias("precision_index"),
    ])

    # 3. QUANTUM SUPERPOSITION
    # Market exists in multiple states simultaneously until "measured"
    df = df.with_columns([
        # Bull state probability
        ((pl.col("close") > pl.col("close").rolling_mean(20)).cast(pl.Int32)
         .rolling_mean(10)).alias("bull_state_prob"),

        # Bear state probability
        ((pl.col("close") < pl.col("close").rolling_mean(20)).cast(pl.Int32)
         .rolling_mean(10)).alias("bear_state_prob"),

        # Range state probability
        ((pl.col("close") - pl.col("close").rolling_mean(20)).abs() <
         pl.col("close").rolling_std(20) * 0.5).cast(pl.Int32)
        .rolling_mean(10).alias("range_state_prob"),
    ])

    # Superposition coherence (how mixed the states are)
    df = df.with_columns([
        (pl.col("bull_state_prob") * pl.col("bear_state_prob") *
         pl.col("range_state_prob") * 27).alias("superposition_coherence"),
    ])

    # 4. QUANTUM ENTANGLEMENT (correlation persistence)
    # When assets are entangled, measuring one affects the other
    df = df.with_columns([
        # Self-entanglement: current price with its own past
        pl.corr("close", pl.col("close").shift(20), ddof=0)
        .rolling(10).alias("temporal_entanglement"),

        # Momentum entanglement
        pl.corr("returns", pl.col("returns").shift(10), ddof=0)
        .rolling(10).alias("momentum_entanglement"),
    ])

    # Entanglement strength (persistent correlation)
    df = df.with_columns([
        (pl.col("temporal_entanglement").abs() +
         pl.col("momentum_entanglement").abs()).alias("entanglement_strength"),
    ])

    # 5. QUANTUM TUNNELING
    # Price can "tunnel" through resistance/support barriers
    df = df.with_columns([
        # Resistance level (recent high)
        pl.col("high").rolling_max(50).alias("resistance"),

        # Support level (recent low)
        pl.col("low").rolling_min(50).alias("support"),
    ])

    df = df.with_columns([
        # Barrier height (distance to resistance)
        ((pl.col("resistance") - pl.col("close")) / pl.col("close"))
        .alias("resistance_barrier"),

        # Support barrier
        ((pl.col("close") - pl.col("support")) / pl.col("close"))
        .alias("support_barrier"),

        # Tunneling probability (inverse of barrier height)
        (1 / (pl.col("resistance_barrier") + 0.01)).alias("upside_tunnel_prob"),
        (1 / (pl.col("support_barrier") + 0.01)).alias("downside_tunnel_prob"),
    ])

    # 6. ENERGY STATES (Quantized levels)
    # Market exists in discrete energy levels
    df = df.with_columns([
        # Kinetic energy (momentum-based)
        (pl.col("returns")**2 * pl.col("volume")).alias("kinetic_energy"),

        # Potential energy (distance from mean)
        ((pl.col("close") - pl.col("close").rolling_mean(50))**2)
        .alias("potential_energy"),
    ])

    df = df.with_columns([
        # Total energy (conserved in quantum systems)
        (pl.col("kinetic_energy") + pl.col("potential_energy"))
        .alias("total_energy"),

        # Energy level (quantized into discrete states)
        (pl.col("total_energy") / pl.col("total_energy").rolling_mean(50))
        .alias("energy_level"),
    ])

    # 7. QUANTUM DECOHERENCE
    # Superposition collapses to definite state (trend emerges)
    df = df.with_columns([
        # Coherence time: how long superposition lasts
        pl.col("superposition_coherence").rolling_mean(5).alias("coherence_time"),

        # Decoherence: collapse to definite state
        (1 - pl.col("superposition_coherence")).alias("decoherence"),

        # State collapse indicator (high decoherence = clear trend)
        (pl.col("decoherence") > 0.7).alias("state_collapsed"),
    ])

    # 8. QUANTUM INTERFERENCE
    # Waves interfere constructively (trends align) or destructively (chop)
    df = df.with_columns([
        # Fast wave (short-term trend)
        pl.col("close").rolling_mean(5).alias("fast_wave"),

        # Slow wave (long-term trend)
        pl.col("close").rolling_mean(20).alias("slow_wave"),
    ])

    df = df.with_columns([
        # Constructive interference (waves align)
        ((pl.col("fast_wave") - pl.col("close").shift(5)) *
         (pl.col("slow_wave") - pl.col("close").shift(20)) > 0)
        .cast(pl.Int32).alias("constructive_interference"),

        # Destructive interference (waves oppose)
        ((pl.col("fast_wave") - pl.col("close").shift(5)) *
         (pl.col("slow_wave") - pl.col("close").shift(20)) < 0)
        .cast(pl.Int32).alias("destructive_interference"),
    ])

    # Interference pattern strength
    df = df.with_columns([
        (pl.col("constructive_interference").rolling_sum(10) -
         pl.col("destructive_interference").rolling_sum(10))
        .alias("interference_pattern"),
    ])

    # Clean up temporary columns
    df = df.drop([
        "wave_component", "particle_component", "resistance", "support",
        "fast_wave", "slow_wave"
    ])

    return df

def add_quantum_field_theory_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Advanced quantum field theory features.
    Models market as a quantum field with virtual particles and vacuum energy.
    """

    df = df.with_columns([
        pl.col("close").pct_change().alias("returns"),
    ])

    # Virtual particles (micro-fluctuations that appear and disappear)
    df = df.with_columns([
        # Vacuum energy: fluctuations even at equilibrium
        (pl.col("close") - pl.col("close").rolling_mean(100)).abs()
        .rolling_mean(10).alias("vacuum_fluctuation"),

        # Virtual particle density (micro-movements)
        pl.col("returns").abs().rolling_sum(10).alias("virtual_particle_density"),
    ])

    # Field strength (market force field)
    df = df.with_columns([
        # Field gradient (rate of change of returns)
        pl.col("returns").diff().abs().rolling_mean(10).alias("field_gradient"),

        # Field coupling (how strongly price couples to volume)
        pl.corr("returns", pl.col("volume").pct_change(), ddof=0)
        .rolling(20).alias("field_coupling"),
    ])

    return df