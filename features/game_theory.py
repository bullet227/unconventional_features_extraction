# features/game_theory.py
"""
Game theory features - Models markets as multi-player games.
Includes Nash equilibrium, prisoner's dilemma, and strategic behavior patterns.
"""

import polars as pl
import numpy as np
from typing import Tuple

def add_game_theory_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add game theory based market features."""

    df = df.with_columns([
        pl.col("close").pct_change().alias("returns"),
    ])

    # 1. PRISONER'S DILEMMA (coordination failure)
    # When everyone sells (defects), prices crash; cooperation creates stability
    df = df.with_columns([
        # Selling pressure (defection)
        ((pl.col("close") < pl.col("open")) &
         (pl.col("volume") > pl.col("volume").rolling_mean(20))).cast(pl.Int32)
        .rolling_sum(10).alias("selling_pressure"),

        # Buying pressure (cooperation)
        ((pl.col("close") > pl.col("open")) &
         (pl.col("volume") > pl.col("volume").rolling_mean(20))).cast(pl.Int32)
        .rolling_sum(10).alias("buying_pressure"),
    ])

    df = df.with_columns([
        # Coordination index (cooperation vs defection)
        ((pl.col("buying_pressure") - pl.col("selling_pressure")) /
         (pl.col("buying_pressure") + pl.col("selling_pressure") + 1))
        .alias("coordination_index"),

        # Defection cascade (everyone rushing to exit)
        (pl.col("selling_pressure") > 7).alias("defection_cascade"),
    ])

    df = df.with_columns([
        # Cooperation breakdown (coordination failure imminent)
        (pl.col("coordination_index") < -0.5).alias("cooperation_breakdown"),
    ])

    # 2. NASH EQUILIBRIUM (stable strategy)
    # Price levels where no participant can improve by changing strategy
    df = df.with_columns([
        # Price stability (low volatility = near equilibrium)
        pl.col("close").rolling_std(20).alias("price_volatility"),

        # Volume stability (consistent volume = equilibrium)
        (pl.col("volume").rolling_std(20) / (pl.col("volume").rolling_mean(20) + 1))
        .alias("volume_stability"),
    ])

    df = df.with_columns([
        # Nash equilibrium proxy (stable price + stable volume)
        ((1 / (pl.col("price_volatility") + 0.01)) *
         (1 / (pl.col("volume_stability") + 0.01))).alias("nash_equilibrium_strength"),

        # Equilibrium break (departure from stable state)
        ((pl.col("price_volatility") > pl.col("price_volatility").rolling_mean(50)) |
         (pl.col("volume_stability") > pl.col("volume_stability").rolling_mean(50)))
        .alias("equilibrium_break"),
    ])

    # 3. ZERO-SUM GAME (every winner has a loser)
    # Options/futures markets are zero-sum
    df = df.with_columns([
        # Winner concentration (few big winners)
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("intraday_range_pct"),

        # Loser concentration (trapped traders)
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-8))
        .alias("close_position_in_range"),
    ])

    df = df.with_columns([
        # Top-heavy close (bulls won today)
        (pl.col("close_position_in_range") > 0.7).alias("bulls_won"),

        # Bottom-heavy close (bears won today)
        (pl.col("close_position_in_range") < 0.3).alias("bears_won"),

        # Balanced game (no clear winner)
        ((pl.col("close_position_in_range") >= 0.4) &
         (pl.col("close_position_in_range") <= 0.6)).alias("balanced_game"),
    ])

    # 4. STAG HUNT (coordination game)
    # High reward requires coordination; safe option is low reward
    df = df.with_columns([
        # Trend strength (coordination success = trend)
        (pl.col("close").rolling_mean(10) - pl.col("close").rolling_mean(50)).abs()
        .alias("trend_strength"),

        # Range trading (safe but low reward)
        (pl.col("high").rolling_max(20) - pl.col("low").rolling_min(20))
        .alias("trading_range"),
    ])

    df = df.with_columns([
        # Stag hunt success (coordinated trend following)
        (pl.col("trend_strength") > pl.col("trend_strength").rolling_mean(50))
        .alias("coordination_success"),

        # Risk aversion (prefer safe range over coordinated hunt)
        (pl.col("trend_strength") < pl.col("trading_range") * 0.3)
        .alias("risk_aversion"),
    ])

    # 5. CHICKEN GAME (who exits first?)
    # Two traders heading for collision; first to exit loses face but survives
    df = df.with_columns([
        # Momentum clash (opposite forces meeting)
        (pl.col("returns").rolling_mean(5) * pl.col("returns").rolling_mean(20))
        .alias("momentum_product"),

        # Volume surge (both sides committed)
        (pl.col("volume") > pl.col("volume").rolling_mean(20) * 1.5)
        .alias("volume_surge"),
    ])

    df = df.with_columns([
        # Chicken game active (opposing momentums + high volume)
        ((pl.col("momentum_product") < 0) &
         pl.col("volume_surge")).alias("chicken_game_active"),

        # Someone blinked (momentum reversal)
        (pl.col("momentum_product").shift(1) * pl.col("momentum_product") < 0)
        .alias("someone_blinked"),
    ])

    # 6. REPEATED GAME (history matters)
    # Players remember and punish/reward based on past actions
    df = df.with_columns([
        # Retaliation pattern (punishment for past defection)
        (pl.col("returns").shift(1) < -0.01).cast(pl.Int32)
        .rolling_sum(5).alias("past_losses"),

        # Reward pattern (cooperation success)
        (pl.col("returns").shift(1) > 0.01).cast(pl.Int32)
        .rolling_sum(5).alias("past_wins"),
    ])

    df = df.with_columns([
        # Tit-for-tat strategy (mirror opponent's last move)
        (pl.col("returns").shift(1).sign()).alias("tit_for_tat_signal"),

        # Grudge formation (repeated losses breed revenge)
        (pl.col("past_losses") > 3).alias("grudge_active"),

        # Trust building (repeated wins encourage cooperation)
        (pl.col("past_wins") > 3).alias("trust_built"),
    ])

    # 7. MINIMAX STRATEGY (minimize maximum loss)
    # Conservative strategy: limit worst-case scenario
    df = df.with_columns([
        # Maximum recent loss
        pl.col("returns").rolling_min(10).alias("max_recent_loss"),

        # Maximum recent gain
        pl.col("returns").rolling_max(10).alias("max_recent_gain"),
    ])

    df = df.with_columns([
        # Minimax ratio (gain/loss asymmetry)
        (pl.col("max_recent_gain") / (pl.col("max_recent_loss").abs() + 1e-8))
        .alias("minimax_ratio"),

        # Loss aversion active (minimizing max loss)
        (pl.col("max_recent_loss").abs() > pl.col("max_recent_gain"))
        .alias("loss_aversion_active"),
    ])

    # 8. DOMINANT STRATEGY (always best choice)
    # Rare in markets, but indicators when one side has clear advantage
    df = df.with_columns([
        # Consecutive wins (dominant strategy working)
        (pl.col("returns") > 0).cast(pl.Int32).alias("win_streak_indicator"),
    ])

    df = df.with_columns([
        # Current win streak
        pl.col("win_streak_indicator").rolling_sum(10).alias("win_streak"),

        # Current loss streak
        (1 - pl.col("win_streak_indicator")).rolling_sum(10).alias("loss_streak"),
    ])

    df = df.with_columns([
        # Dominant bull strategy (consistent wins)
        (pl.col("win_streak") > 7).alias("dominant_bull_strategy"),

        # Dominant bear strategy (consistent losses)
        (pl.col("loss_streak") > 7).alias("dominant_bear_strategy"),

        # Strategy uncertainty (no dominant strategy)
        ((pl.col("win_streak") <= 5) & (pl.col("loss_streak") <= 5))
        .alias("strategy_uncertainty"),
    ])

    # 9. PARETO EFFICIENCY (optimal allocation)
    # Can't improve one without hurting another
    df = df.with_columns([
        # Efficient frontier: high return per unit risk
        (pl.col("returns").rolling_mean(20) /
         (pl.col("returns").rolling_std(20) + 1e-8)).alias("risk_adjusted_return"),

        # Volume efficiency: price movement per unit volume
        (pl.col("returns").abs() / (pl.col("volume").pct_change().abs() + 1e-8))
        .alias("volume_efficiency"),
    ])

    df = df.with_columns([
        # Pareto optimal (efficient on both dimensions)
        ((pl.col("risk_adjusted_return") > pl.col("risk_adjusted_return").rolling_mean(50)) &
         (pl.col("volume_efficiency") > pl.col("volume_efficiency").rolling_mean(50)))
        .alias("pareto_optimal"),

        # Pareto improvement possible (inefficient current state)
        ((pl.col("risk_adjusted_return") < pl.col("risk_adjusted_return").rolling_mean(50)) |
         (pl.col("volume_efficiency") < pl.col("volume_efficiency").rolling_mean(50)))
        .alias("pareto_improvement_possible"),
    ])

    # 10. MECHANISM DESIGN (market structure effects)
    # How market rules affect participant behavior
    df = df.with_columns([
        # Bid-ask spread proxy (transaction cost)
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("spread_proxy"),
    ])

    df = df.with_columns([
        # Liquidity quality (tight spread + high volume)
        (pl.col("volume") / (pl.col("spread_proxy") + 1e-8)).alias("liquidity_quality"),
    ])

    df = df.with_columns([
        # High-quality mechanism (efficient market structure)
        (pl.col("liquidity_quality") > pl.col("liquidity_quality").rolling_mean(50))
        .alias("efficient_mechanism"),

        # Poor mechanism (wide spreads, low liquidity)
        (pl.col("liquidity_quality") < pl.col("liquidity_quality").rolling_mean(50) * 0.5)
        .alias("inefficient_mechanism"),
    ])

    # Clean up temporary columns
    df = df.drop([
        "win_streak_indicator", "price_volatility", "volume_stability",
        "momentum_product", "intraday_range_pct"
    ])

    return df