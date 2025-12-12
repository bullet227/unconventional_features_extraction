# utils/hft_scalping.py
"""
HFT Scalping features - Analyzes tick/sub-minute data during candle formation.
Captures microstructure, order flow, and ultra-short-term momentum.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import asyncio
from datetime import datetime, timedelta

class TickAggregator:
    """Aggregates tick data into microstructure features."""
    
    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self.tick_buffer = deque(maxlen=10000)  # Store recent ticks
        self.micro_candles = []  # 5-second candles within the minute
        
    def add_tick(self, timestamp: datetime, price: float, volume: float, side: str = None):
        """Add a new tick to the buffer."""
        self.tick_buffer.append({
            'time': timestamp,
            'price': price,
            'volume': volume,
            'side': side  # 'buy' or 'sell' if known
        })
        
    def get_micro_candles(self, seconds: int = 5) -> List[Dict]:
        """Create micro-candles from tick data."""
        if not self.tick_buffer:
            return []
        
        candles = []
        current_candle = None
        
        for tick in self.tick_buffer:
            candle_time = tick['time'].replace(second=(tick['time'].second // seconds) * seconds)
            
            if current_candle is None or current_candle['time'] != candle_time:
                if current_candle:
                    candles.append(current_candle)
                    
                current_candle = {
                    'time': candle_time,
                    'open': tick['price'],
                    'high': tick['price'],
                    'low': tick['price'],
                    'close': tick['price'],
                    'volume': 0,
                    'tick_count': 0,
                    'buy_volume': 0,
                    'sell_volume': 0
                }
            
            # Update candle
            current_candle['high'] = max(current_candle['high'], tick['price'])
            current_candle['low'] = min(current_candle['low'], tick['price'])
            current_candle['close'] = tick['price']
            current_candle['volume'] += tick['volume']
            current_candle['tick_count'] += 1
            
            if tick['side'] == 'buy':
                current_candle['buy_volume'] += tick['volume']
            elif tick['side'] == 'sell':
                current_candle['sell_volume'] += tick['volume']
        
        if current_candle:
            candles.append(current_candle)
            
        return candles

def add_hft_scalping_features(df: pl.DataFrame, tick_data: Dict = None) -> pl.DataFrame:
    """Add HFT scalping features based on intra-candle dynamics."""
    
    # Initialize tick aggregator if tick data provided
    if tick_data:
        aggregator = TickAggregator()
        # Process tick data here
    
    # Microstructure features
    df = df.with_columns([
        # Tick-based momentum (requires tick data)
        pl.lit(0.0).alias("tick_momentum"),  # Placeholder
        
        # Micro-trend detection (from 5-second candles)
        pl.lit(0.0).alias("micro_trend"),  # Placeholder
        
        # Order flow imbalance
        pl.lit(0.0).alias("order_flow_imbalance"),  # Placeholder
    ])
    
    # Price action during candle formation
    df = df.with_columns([
        # Open-to-current efficiency (how direct was the move)
        ((pl.col("close") - pl.col("open")) / 
         (pl.col("high") - pl.col("low") + 0.0001)).alias("price_efficiency"),
        
        # Wick formation timing (early vs late wicks)
        ((pl.col("high") - pl.max_horizontal([pl.col("open"), pl.col("close")])) /
         (pl.col("high") - pl.col("low") + 0.0001)).alias("upper_wick_ratio"),
        
        ((pl.min_horizontal([pl.col("open"), pl.col("close")]) - pl.col("low")) /
         (pl.col("high") - pl.col("low") + 0.0001)).alias("lower_wick_ratio"),
    ])
    
    # Volume profile within candle
    df = df.with_columns([
        # Volume acceleration (front-loaded vs back-loaded)
        pl.col("volume").diff().alias("volume_acceleration"),
        
        # Volume-price correlation (strong moves have aligned volume)
        pl.corr("volume", "close").rolling(5).alias("volume_price_correlation"),
    ])
    
    # Momentum burst detection
    df = df.with_columns([
        # Rapid price movement in last N seconds
        (pl.col("close") - pl.col("open")).abs().alias("price_movement"),
        
        # Momentum relative to recent average
        ((pl.col("close") - pl.col("open")).abs() / 
         pl.col("close").rolling_std(20)).alias("relative_momentum"),
    ])
    
    # Micro-reversal patterns
    df = df.with_columns([
        # Quick rejection from highs/lows
        ((pl.col("high") - pl.col("close")) > (pl.col("close") - pl.col("open")).abs() * 2)
        .alias("upper_rejection"),
        
        ((pl.col("close") - pl.col("low")) > (pl.col("close") - pl.col("open")).abs() * 2)
        .alias("lower_rejection"),
        
        # Doji detection (indecision)
        ((pl.col("close") - pl.col("open")).abs() < pl.col("close") * 0.0001)
        .alias("is_doji"),
    ])
    
    # Speed and acceleration
    df = df.with_columns([
        # Price speed (pips per second estimated)
        ((pl.col("high") - pl.col("low")) / 60).alias("price_speed_pps"),
        
        # Speed change (acceleration)
        ((pl.col("high") - pl.col("low")) / 60).diff().alias("price_acceleration_pps"),
        
        # Volatility burst
        (pl.col("high") - pl.col("low")).rolling_quantile(0.95, window_size=60)
        .alias("volatility_95pct"),
    ])
    
    # Liquidity grab detection
    df = df.with_columns([
        # Quick spike beyond recent range
        ((pl.col("high") > pl.col("high").shift(1).rolling_max(5)) & 
         (pl.col("close") < pl.col("high"))).alias("liquidity_grab_high"),
        
        ((pl.col("low") < pl.col("low").shift(1).rolling_min(5)) & 
         (pl.col("close") > pl.col("low"))).alias("liquidity_grab_low"),
    ])
    
    # Market maker activity detection
    df = df.with_columns([
        # Narrow range with high volume (accumulation)
        ((pl.col("high") - pl.col("low")) < pl.col("close") * 0.0005) & 
        (pl.col("volume") > pl.col("volume").rolling_mean(20) * 1.5)
        .alias("mm_accumulation"),
        
        # Range expansion with volume (distribution)
        ((pl.col("high") - pl.col("low")) > pl.col("close").rolling_std(20) * 2) & 
        (pl.col("volume") > pl.col("volume").rolling_mean(20) * 2)
        .alias("mm_distribution"),
    ])
    
    # Entry timing features
    df = df.with_columns([
        # Optimal entry zones based on micro-structure
        ((pl.col("close") - pl.col("low")) < (pl.col("high") - pl.col("low")) * 0.25)
        .alias("near_low_entry"),
        
        ((pl.col("high") - pl.col("close")) < (pl.col("high") - pl.col("low")) * 0.25)
        .alias("near_high_entry"),
        
        # Momentum continuation probability
        (pl.col("relative_momentum") > 2).alias("strong_momentum"),
        
        # Mean reversion probability
        ((pl.col("close") - pl.col("close").rolling_mean(5)).abs() > 
         pl.col("close").rolling_std(20) * 2).alias("overextended"),
    ])
    
    # Risk metrics for scalping
    df = df.with_columns([
        # Maximum adverse excursion within candle
        pl.when(pl.col("close") > pl.col("open"))
        .then(pl.col("open") - pl.col("low"))
        .otherwise(pl.col("high") - pl.col("open"))
        .alias("max_adverse_excursion"),
        
        # Risk-reward ratio potential
        pl.when(pl.col("close") > pl.col("open"))
        .then((pl.col("high") - pl.col("close")) / (pl.col("close") - pl.col("low") + 0.0001))
        .otherwise((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("close") + 0.0001))
        .alias("risk_reward_ratio"),
    ])
    
    return df


def add_order_flow_features(df: pl.DataFrame, level2_data: Dict = None) -> pl.DataFrame:
    """Add order flow and market depth features."""
    
    # These would use Level 2 data if available
    df = df.with_columns([
        # Bid-ask spread features
        pl.lit(0.0).alias("spread_width"),  # Placeholder for actual spread
        pl.lit(0.0).alias("spread_volatility"),
        
        # Order book imbalance
        pl.lit(0.0).alias("bid_ask_ratio"),
        pl.lit(0.0).alias("order_book_pressure"),
        
        # Large order detection
        pl.lit(False).alias("large_bid_detected"),
        pl.lit(False).alias("large_ask_detected"),
    ])
    
    # Tape reading features
    df = df.with_columns([
        # Trade size distribution
        pl.col("volume").rolling_quantile(0.9, window_size=20).alias("large_trade_threshold"),
        
        # Aggressive buying/selling
        (pl.col("close") > pl.col("open")).rolling_sum(5).alias("buying_pressure"),
        (pl.col("close") < pl.col("open")).rolling_sum(5).alias("selling_pressure"),
    ])
    
    return df


def add_execution_signals(df: pl.DataFrame) -> pl.DataFrame:
    """Add specific scalping execution signals."""
    
    # Scalping entry signals
    df = df.with_columns([
        # Momentum scalp long
        ((pl.col("relative_momentum") > 1.5) & 
         (pl.col("price_efficiency") > 0.7) &
         (pl.col("buying_pressure") > 3) &
         ~pl.col("overextended"))
        .alias("scalp_long_signal"),
        
        # Momentum scalp short
        ((pl.col("relative_momentum") > 1.5) & 
         (pl.col("price_efficiency") < -0.7) &
         (pl.col("selling_pressure") > 3) &
         ~pl.col("overextended"))
        .alias("scalp_short_signal"),
        
        # Mean reversion scalp
        (pl.col("overextended") & 
         (pl.col("upper_rejection") | pl.col("lower_rejection")))
        .alias("mean_reversion_signal"),
        
        # Breakout scalp
        ((pl.col("liquidity_grab_high") | pl.col("liquidity_grab_low")) &
         (pl.col("volume_acceleration") > 0))
        .alias("breakout_scalp_signal"),
    ])
    
    # Exit signals
    df = df.with_columns([
        # Quick profit target hit
        pl.lit(False).alias("profit_target_hit"),  # Would be calculated based on entry
        
        # Momentum exhaustion
        (pl.col("relative_momentum") < 0.5).alias("momentum_exhausted"),
        
        # Adverse movement
        (pl.col("max_adverse_excursion") > pl.col("close") * 0.001).alias("stop_loss_triggered"),
    ])
    
    # Position sizing based on volatility
    df = df.with_columns([
        # Dynamic position size (inverse volatility)
        (1 / (pl.col("price_speed_pps") + 0.0001)).alias("position_size_factor"),
        
        # Maximum position based on recent volatility
        (pl.col("close") * 0.001 / (pl.col("volatility_95pct") + 0.0001))
        .alias("max_position_size"),
    ])
    
    return df


def create_hft_features_realtime(tick_stream):
    """Real-time feature calculation for live trading."""
    
    class RealtimeHFTProcessor:
        def __init__(self):
            self.aggregator = TickAggregator()
            self.features = {}
            self.positions = []
            
        async def process_tick(self, tick):
            """Process incoming tick and update features."""
            self.aggregator.add_tick(
                tick['timestamp'],
                tick['price'],
                tick['volume'],
                tick.get('side')
            )
            
            # Update micro-candles
            micro_candles = self.aggregator.get_micro_candles(5)
            
            # Calculate real-time features
            if len(micro_candles) >= 3:
                self.features['micro_trend'] = self._calculate_micro_trend(micro_candles)
                self.features['order_flow'] = self._calculate_order_flow(micro_candles)
                self.features['momentum_burst'] = self._detect_momentum_burst(micro_candles)
                
            return self.features
            
        def _calculate_micro_trend(self, candles):
            """Calculate trend from micro candles."""
            if len(candles) < 3:
                return 0
            
            closes = [c['close'] for c in candles[-3:]]
            return (closes[-1] - closes[0]) / closes[0]
            
        def _calculate_order_flow(self, candles):
            """Calculate order flow imbalance."""
            total_buy = sum(c.get('buy_volume', 0) for c in candles[-5:])
            total_sell = sum(c.get('sell_volume', 0) for c in candles[-5:])
            
            if total_buy + total_sell == 0:
                return 0
                
            return (total_buy - total_sell) / (total_buy + total_sell)
            
        def _detect_momentum_burst(self, candles):
            """Detect sudden momentum bursts."""
            if len(candles) < 2:
                return False
                
            current_move = abs(candles[-1]['close'] - candles[-1]['open'])
            avg_move = np.mean([abs(c['close'] - c['open']) for c in candles[-10:-1]])
            
            return current_move > avg_move * 3
    
    return RealtimeHFTProcessor()