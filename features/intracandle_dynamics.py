# utils/intracandle_dynamics.py
"""
Intra-candle dynamics - Analyzes price behavior during candle formation.
Captures micro-movements, formation patterns, and sub-minute momentum.
"""

import polars as pl
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import deque

@dataclass
class IntraCandleState:
    """Tracks the state of a forming candle."""
    open_time: float
    open_price: float
    current_price: float
    high: float
    low: float
    volume: float = 0
    tick_count: int = 0
    
    # Formation tracking
    high_time: float = 0  # When high was reached (0-1, 0=open, 1=close)
    low_time: float = 0   # When low was reached
    
    # Path tracking
    price_path: List[float] = None
    volume_path: List[float] = None
    time_path: List[float] = None
    
    def __post_init__(self):
        if self.price_path is None:
            self.price_path = [self.open_price]
        if self.volume_path is None:
            self.volume_path = [0]
        if self.time_path is None:
            self.time_path = [0]


def analyze_candle_formation(state: IntraCandleState) -> Dict[str, float]:
    """Analyze how a candle formed from its path data."""
    
    features = {}
    
    if len(state.price_path) < 2:
        return features
    
    # Path efficiency (how direct was the move)
    total_distance = sum(abs(state.price_path[i] - state.price_path[i-1]) 
                        for i in range(1, len(state.price_path)))
    net_distance = abs(state.current_price - state.open_price)
    features['path_efficiency'] = net_distance / (total_distance + 1e-8)
    
    # Formation shape
    normalized_path = [(p - state.low) / (state.high - state.low + 1e-8) 
                      for p in state.price_path]
    
    # Early vs late movement
    mid_point = len(normalized_path) // 2
    early_movement = abs(normalized_path[mid_point] - normalized_path[0])
    late_movement = abs(normalized_path[-1] - normalized_path[mid_point])
    features['early_vs_late'] = early_movement - late_movement
    
    # Smoothness (low variance = smooth, high = choppy)
    if len(normalized_path) > 2:
        features['path_smoothness'] = 1 / (np.std(normalized_path) + 0.01)
    
    # Volume distribution
    if len(state.volume_path) > 1:
        volume_front = sum(state.volume_path[:len(state.volume_path)//2])
        volume_back = sum(state.volume_path[len(state.volume_path)//2:])
        features['volume_skew'] = (volume_front - volume_back) / (volume_front + volume_back + 1)
    
    # Reversal count (V-shapes, M-shapes)
    reversals = 0
    for i in range(1, len(state.price_path) - 1):
        if (state.price_path[i] > state.price_path[i-1] and 
            state.price_path[i] > state.price_path[i+1]) or \
           (state.price_path[i] < state.price_path[i-1] and 
            state.price_path[i] < state.price_path[i+1]):
            reversals += 1
    features['reversal_count'] = reversals
    
    # Time at extremes
    features['high_time_pct'] = state.high_time
    features['low_time_pct'] = state.low_time
    features['extreme_time_diff'] = abs(state.high_time - state.low_time)
    
    return features


def add_intracandle_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add features based on intra-candle price formation."""
    
    # Formation patterns
    df = df.with_columns([
        # Candle body position within range
        ((pl.col("open") + pl.col("close")) / 2 - pl.col("low")) / 
        (pl.col("high") - pl.col("low") + 0.0001)
        .alias("body_position"),
        
        # Body to range ratio
        (pl.col("close") - pl.col("open")).abs() / 
        (pl.col("high") - pl.col("low") + 0.0001)
        .alias("body_range_ratio"),
    ])
    
    # Wick dynamics
    df = df.with_columns([
        # Upper wick momentum (large upper wick = selling pressure)
        (pl.col("high") - pl.max_horizontal([pl.col("open"), pl.col("close")])) / 
        pl.col("close") * 100
        .alias("upper_wick_pct"),
        
        # Lower wick momentum (large lower wick = buying pressure)
        (pl.min_horizontal([pl.col("open"), pl.col("close")]) - pl.col("low")) / 
        pl.col("close") * 100
        .alias("lower_wick_pct"),
        
        # Wick ratio (upper vs lower dominance)
        ((pl.col("high") - pl.max_horizontal([pl.col("open"), pl.col("close")])) / 
         (pl.min_horizontal([pl.col("open"), pl.col("close")]) - pl.col("low") + 0.0001))
        .alias("wick_ratio"),
    ])
    
    # Micro-momentum indicators
    df = df.with_columns([
        # Closing momentum (where in range did we close)
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 0.0001))
        .alias("close_position"),
        
        # Opening momentum (gap from previous close)
        ((pl.col("open") - pl.col("close").shift(1)) / pl.col("close").shift(1) * 100)
        .alias("open_gap_pct"),
        
        # Momentum shift (open to close direction vs previous)
        ((pl.col("close") - pl.col("open")) * (pl.col("close").shift(1) - pl.col("open").shift(1)) > 0)
        .alias("momentum_continuation"),
    ])
    
    # Candle type classification
    df = df.with_columns([
        # Hammer pattern (bullish reversal)
        ((pl.col("lower_wick_pct") > pl.col("body_range_ratio") * 2) & 
         (pl.col("upper_wick_pct") < pl.col("body_range_ratio") * 0.5) &
         (pl.col("close_position") > 0.65))
        .alias("is_hammer"),
        
        # Shooting star (bearish reversal)
        ((pl.col("upper_wick_pct") > pl.col("body_range_ratio") * 2) & 
         (pl.col("lower_wick_pct") < pl.col("body_range_ratio") * 0.5) &
         (pl.col("close_position") < 0.35))
        .alias("is_shooting_star"),
        
        # Doji variations
        (pl.col("body_range_ratio") < 0.1).alias("is_doji"),
        
        # Marubozu (no wicks, strong momentum)
        ((pl.col("upper_wick_pct") < 0.01) & (pl.col("lower_wick_pct") < 0.01))
        .alias("is_marubozu"),
    ])
    
    # Micro-trend detection
    df = df.with_columns([
        # 3-candle micro trend
        ((pl.col("close") > pl.col("close").shift(1)) & 
         (pl.col("close").shift(1) > pl.col("close").shift(2)))
        .alias("micro_uptrend"),
        
        ((pl.col("close") < pl.col("close").shift(1)) & 
         (pl.col("close").shift(1) < pl.col("close").shift(2)))
        .alias("micro_downtrend"),
        
        # Acceleration (increasing range)
        ((pl.col("high") - pl.col("low")) > (pl.col("high").shift(1) - pl.col("low").shift(1)))
        .alias("range_expansion"),
    ])
    
    # Volume-price dynamics
    df = df.with_columns([
        # Volume surge detection
        (pl.col("volume") > pl.col("volume").rolling_quantile(0.9, window_size=20))
        .alias("volume_surge"),
        
        # Volume-price divergence
        ((pl.col("volume") > pl.col("volume").shift(1)) & 
         ((pl.col("close") - pl.col("open")).abs() < 
          (pl.col("close").shift(1) - pl.col("open").shift(1)).abs()))
        .alias("volume_price_divergence"),
        
        # Accumulation/Distribution
        ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / 
        (pl.col("high") - pl.col("low") + 0.0001) * pl.col("volume")
        .alias("accumulation_distribution"),
    ])
    
    # Scalping opportunity scores
    df = df.with_columns([
        # Momentum scalp score
        (pl.col("micro_uptrend").cast(pl.Float32) * 2 + 
         pl.col("momentum_continuation").cast(pl.Float32) * 1.5 +
         pl.col("volume_surge").cast(pl.Float32) * 1 +
         pl.col("range_expansion").cast(pl.Float32) * 0.5 +
         (1 - pl.col("upper_wick_pct") / 100).clip(0, 1))
        .alias("long_scalp_score"),
        
        # Reversal scalp score
        (pl.col("is_hammer").cast(pl.Float32) * 3 +
         pl.col("is_shooting_star").cast(pl.Float32) * 3 +
         pl.col("volume_surge").cast(pl.Float32) * 1 +
         pl.col("is_doji").cast(pl.Float32) * 0.5)
        .alias("reversal_scalp_score"),
    ])
    
    return df


def calculate_tick_features(ticks: List[Dict], window_seconds: int = 60) -> Dict[str, float]:
    """Calculate features from raw tick data within a time window."""
    
    if not ticks:
        return {}
    
    features = {}
    
    # Price features
    prices = [t['price'] for t in ticks]
    features['tick_count'] = len(ticks)
    features['price_mean'] = np.mean(prices)
    features['price_std'] = np.std(prices)
    features['price_skew'] = np.sign(prices[-1] - prices[0]) * (features['price_std'] / features['price_mean'])
    
    # Tick frequency
    if len(ticks) > 1:
        time_diffs = [(ticks[i]['time'] - ticks[i-1]['time']).total_seconds() 
                     for i in range(1, len(ticks))]
        features['avg_tick_interval'] = np.mean(time_diffs)
        features['tick_clustering'] = np.std(time_diffs) / (np.mean(time_diffs) + 1e-8)
    
    # Price jumps
    price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    if price_changes:
        features['avg_price_jump'] = np.mean(price_changes)
        features['max_price_jump'] = max(price_changes)
        features['jump_frequency'] = sum(1 for j in price_changes if j > np.mean(price_changes) * 2) / len(price_changes)
    
    # Microtrend
    if len(prices) >= 3:
        # Linear regression slope
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        features['micro_trend_slope'] = slope
        
        # Trend strength (R-squared)
        y_pred = np.polyval([slope, np.mean(prices)], x)
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        features['trend_strength'] = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Buy/sell pressure from tick direction
    buy_ticks = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
    sell_ticks = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])
    total_directional = buy_ticks + sell_ticks
    
    if total_directional > 0:
        features['buy_pressure'] = buy_ticks / total_directional
        features['sell_pressure'] = sell_ticks / total_directional
    
    return features


class IntraCandleTracker:
    """Tracks forming candles in real-time for HFT decisions."""
    
    def __init__(self, candle_period_seconds: int = 60):
        self.period = candle_period_seconds
        self.current_candle = None
        self.completed_candles = deque(maxlen=100)
        self.tick_buffer = deque(maxlen=1000)
        
    def process_tick(self, timestamp: float, price: float, volume: float) -> Dict[str, any]:
        """Process a new tick and return current candle state + signals."""
        
        # Initialize or roll candle
        candle_start = (timestamp // self.period) * self.period
        
        if self.current_candle is None or self.current_candle.open_time != candle_start:
            if self.current_candle:
                # Finalize previous candle
                final_features = analyze_candle_formation(self.current_candle)
                self.completed_candles.append({
                    'time': self.current_candle.open_time,
                    'open': self.current_candle.open_price,
                    'high': self.current_candle.high,
                    'low': self.current_candle.low,
                    'close': self.current_candle.current_price,
                    'volume': self.current_candle.volume,
                    **final_features
                })
            
            # Start new candle
            self.current_candle = IntraCandleState(
                open_time=candle_start,
                open_price=price,
                current_price=price,
                high=price,
                low=price
            )
        
        # Update current candle
        elapsed = (timestamp - self.current_candle.open_time) / self.period
        
        self.current_candle.current_price = price
        self.current_candle.volume += volume
        self.current_candle.tick_count += 1
        
        if price > self.current_candle.high:
            self.current_candle.high = price
            self.current_candle.high_time = elapsed
            
        if price < self.current_candle.low:
            self.current_candle.low = price
            self.current_candle.low_time = elapsed
        
        self.current_candle.price_path.append(price)
        self.current_candle.volume_path.append(volume)
        self.current_candle.time_path.append(elapsed)
        
        # Add to tick buffer
        self.tick_buffer.append({
            'time': timestamp,
            'price': price,
            'volume': volume
        })
        
        # Generate real-time features
        features = self._calculate_realtime_features()
        signals = self._generate_signals(features)
        
        return {
            'candle_state': self.current_candle,
            'features': features,
            'signals': signals
        }
    
    def _calculate_realtime_features(self) -> Dict[str, float]:
        """Calculate features for the forming candle."""
        
        if not self.current_candle:
            return {}
        
        features = analyze_candle_formation(self.current_candle)
        
        # Add real-time specific features
        features['completion_pct'] = self.current_candle.time_path[-1] if self.current_candle.time_path else 0
        features['current_position'] = (self.current_candle.current_price - self.current_candle.low) / \
                                     (self.current_candle.high - self.current_candle.low + 1e-8)
        
        # Recent tick features
        recent_ticks = list(self.tick_buffer)[-20:]  # Last 20 ticks
        if len(recent_ticks) >= 3:
            tick_features = calculate_tick_features(recent_ticks)
            features.update({f'tick_{k}': v for k, v in tick_features.items()})
        
        return features
    
    def _generate_signals(self, features: Dict[str, float]) -> Dict[str, bool]:
        """Generate trading signals based on current features."""
        
        signals = {}
        
        # Early momentum detection (first 30% of candle)
        if features.get('completion_pct', 0) < 0.3:
            signals['early_momentum_long'] = (
                features.get('path_efficiency', 0) > 0.8 and
                features.get('tick_buy_pressure', 0.5) > 0.7 and
                features.get('current_position', 0.5) > 0.7
            )
            signals['early_momentum_short'] = (
                features.get('path_efficiency', 0) > 0.8 and
                features.get('tick_sell_pressure', 0.5) > 0.7 and
                features.get('current_position', 0.5) < 0.3
            )
        
        # Mid-candle reversal (30-70% of candle)
        elif features.get('completion_pct', 0) < 0.7:
            signals['mid_reversal_long'] = (
                features.get('current_position', 0.5) < 0.2 and
                features.get('reversal_count', 0) >= 1 and
                features.get('tick_micro_trend_slope', 0) > 0
            )
            signals['mid_reversal_short'] = (
                features.get('current_position', 0.5) > 0.8 and
                features.get('reversal_count', 0) >= 1 and
                features.get('tick_micro_trend_slope', 0) < 0
            )
        
        # Late candle continuation (70%+ of candle)
        else:
            signals['late_continuation_long'] = (
                features.get('current_position', 0.5) > 0.6 and
                features.get('path_smoothness', 0) > 2 and
                features.get('early_vs_late', 0) < 0  # Late movement
            )
            signals['late_continuation_short'] = (
                features.get('current_position', 0.5) < 0.4 and
                features.get('path_smoothness', 0) > 2 and
                features.get('early_vs_late', 0) < 0
            )
        
        # Risk signals
        signals['high_volatility'] = features.get('tick_price_std', 0) > features.get('tick_price_mean', 1) * 0.002
        signals['low_liquidity'] = features.get('tick_avg_tick_interval', 1) > 2  # More than 2 seconds between ticks
        
        return signals