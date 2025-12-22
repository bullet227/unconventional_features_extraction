# utils/orderflow_mm_detection.py
"""
Order flow analysis and market maker detection for HFT scalping.
Identifies institutional activity, order flow imbalances, and market maker patterns.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
import asyncio

@dataclass 
class OrderBookSnapshot:
    """Level 2 order book snapshot."""
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]
    
    @property
    def best_bid(self) -> Tuple[float, float]:
        return self.bids[0] if self.bids else (0, 0)
    
    @property
    def best_ask(self) -> Tuple[float, float]:
        return self.asks[0] if self.asks else (0, 0)
    
    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.asks[0][0] + self.bids[0][0]) / 2
        return 0


class OrderFlowAnalyzer:
    """Analyzes order flow patterns and detects market maker activity."""
    
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        self.book_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=5000)
        self.mm_patterns = defaultdict(int)
        
    def process_book_update(self, book: OrderBookSnapshot) -> Dict[str, float]:
        """Process order book update and extract features."""
        
        features = {}
        self.book_history.append(book)
        
        # Basic spread metrics
        features['spread'] = book.spread
        features['spread_bps'] = book.spread / book.mid_price * 10000 if book.mid_price > 0 else 0
        
        # Order book imbalance
        features.update(self._calculate_book_imbalance(book))
        
        # Market maker detection
        features.update(self._detect_market_maker_activity(book))
        
        # Depth analysis
        features.update(self._analyze_book_depth(book))
        
        # Dynamic features (requires history)
        if len(self.book_history) > 10:
            features.update(self._calculate_dynamic_features())
        
        return features
    
    def _calculate_book_imbalance(self, book: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate order book imbalance metrics."""
        
        features = {}
        
        # Volume imbalance at different levels
        for level in [1, 3, 5, 10]:
            bid_volume = sum(size for _, size in book.bids[:level])
            ask_volume = sum(size for _, size in book.asks[:level])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                features[f'imbalance_l{level}'] = (bid_volume - ask_volume) / total_volume
                features[f'bid_ratio_l{level}'] = bid_volume / total_volume
            else:
                features[f'imbalance_l{level}'] = 0
                features[f'bid_ratio_l{level}'] = 0.5
        
        # Weighted imbalance (closer levels have more weight)
        weighted_bid = sum(size / (i + 1) for i, (_, size) in enumerate(book.bids[:5]))
        weighted_ask = sum(size / (i + 1) for i, (_, size) in enumerate(book.asks[:5]))
        total_weighted = weighted_bid + weighted_ask
        
        features['weighted_imbalance'] = (weighted_bid - weighted_ask) / total_weighted if total_weighted > 0 else 0
        
        # Pressure score (large orders near touch)
        if book.bids and book.asks:
            bid_pressure = sum(size for p, size in book.bids if p >= book.best_bid[0] * 0.9995)
            ask_pressure = sum(size for p, size in book.asks if p <= book.best_ask[0] * 1.0005)
            features['bid_pressure'] = bid_pressure
            features['ask_pressure'] = ask_pressure
            features['pressure_ratio'] = bid_pressure / (ask_pressure + 1) 
        
        return features
    
    def _detect_market_maker_activity(self, book: OrderBookSnapshot) -> Dict[str, float]:
        """Detect patterns indicative of market maker activity."""
        
        features = {}
        
        # Layering detection (multiple orders at regular intervals)
        if len(book.bids) >= 5 and len(book.asks) >= 5:
            bid_intervals = [book.bids[i][0] - book.bids[i+1][0] for i in range(4)]
            ask_intervals = [book.asks[i+1][0] - book.asks[i][0] for i in range(4)]
            
            # Check for regular spacing (market maker layering)
            bid_regularity = np.std(bid_intervals) / (np.mean(bid_intervals) + 1e-8)
            ask_regularity = np.std(ask_intervals) / (np.mean(ask_intervals) + 1e-8)
            
            features['bid_layering_score'] = 1 / (1 + bid_regularity)
            features['ask_layering_score'] = 1 / (1 + ask_regularity)
            features['mm_layering_detected'] = bid_regularity < 0.2 or ask_regularity < 0.2
        
        # Quote stuffing detection (rapid changes)
        if len(self.book_history) >= 10:
            recent_spreads = [b.spread for b in list(self.book_history)[-10:]]
            spread_changes = sum(1 for i in range(1, len(recent_spreads)) 
                               if recent_spreads[i] != recent_spreads[i-1])
            features['quote_stuffing_score'] = spread_changes / len(recent_spreads)
        
        # Spoofing detection (large orders that disappear)
        features['spoof_risk'] = self._detect_spoofing()
        
        # Market maker spread pattern
        if book.spread > 0 and book.mid_price > 0:
            normalized_spread = book.spread / book.mid_price
            # MMs often maintain specific spread ratios
            common_spreads = [0.0001, 0.0002, 0.0005, 0.001]  # 1, 2, 5, 10 bps
            min_diff = min(abs(normalized_spread - s) for s in common_spreads)
            features['mm_spread_pattern'] = 1 / (1 + min_diff * 10000)
        
        return features
    
    def _analyze_book_depth(self, book: OrderBookSnapshot) -> Dict[str, float]:
        """Analyze order book depth characteristics."""
        
        features = {}
        
        # Depth metrics
        if len(book.bids) > 1 and len(book.asks) > 1:
            # Average order size by level
            for side, orders in [('bid', book.bids), ('ask', book.asks)]:
                level_sizes = [size for _, size in orders[:5]]
                if level_sizes:
                    features[f'{side}_avg_size'] = np.mean(level_sizes)
                    features[f'{side}_size_skew'] = level_sizes[0] / (np.mean(level_sizes) + 1)
                    
            # Cliff detection (sudden drop in liquidity)
            bid_sizes = [size for _, size in book.bids[:10]]
            ask_sizes = [size for _, size in book.asks[:10]]
            
            if len(bid_sizes) > 3:
                bid_cliff = max(bid_sizes[i] / (bid_sizes[i+1] + 1) for i in range(len(bid_sizes)-1))
                features['bid_cliff_ratio'] = bid_cliff
                
            if len(ask_sizes) > 3:
                ask_cliff = max(ask_sizes[i] / (ask_sizes[i+1] + 1) for i in range(len(ask_sizes)-1))
                features['ask_cliff_ratio'] = ask_cliff
        
        # Liquidity concentration
        total_bid_volume = sum(size for _, size in book.bids)
        total_ask_volume = sum(size for _, size in book.asks)
        
        if total_bid_volume > 0:
            top3_bid_volume = sum(size for _, size in book.bids[:3])
            features['bid_concentration'] = top3_bid_volume / total_bid_volume
            
        if total_ask_volume > 0:
            top3_ask_volume = sum(size for _, size in book.asks[:3])
            features['ask_concentration'] = top3_ask_volume / total_ask_volume
        
        return features
    
    def _calculate_dynamic_features(self) -> Dict[str, float]:
        """Calculate features that require historical data."""
        
        features = {}
        recent_books = list(self.book_history)[-20:]
        
        # Spread dynamics
        spreads = [b.spread for b in recent_books]
        features['spread_volatility'] = np.std(spreads) / (np.mean(spreads) + 1e-8)
        features['spread_trend'] = (spreads[-1] - spreads[0]) / (spreads[0] + 1e-8)
        
        # Liquidity migration
        bid_volumes = [sum(size for _, size in b.bids[:5]) for b in recent_books]
        ask_volumes = [sum(size for _, size in b.asks[:5]) for b in recent_books]
        
        features['bid_liquidity_trend'] = (bid_volumes[-1] - bid_volumes[0]) / (bid_volumes[0] + 1)
        features['ask_liquidity_trend'] = (ask_volumes[-1] - ask_volumes[0]) / (ask_volumes[0] + 1)
        
        # Order book stability
        mid_prices = [b.mid_price for b in recent_books if b.mid_price > 0]
        if len(mid_prices) > 2:
            features['price_stability'] = 1 / (np.std(mid_prices) / np.mean(mid_prices) + 0.001)
        
        return features
    
    def _detect_spoofing(self) -> float:
        """Detect potential spoofing behavior."""
        
        if len(self.book_history) < 5:
            return 0
        
        # Look for large orders that appear and disappear quickly
        recent_books = list(self.book_history)[-5:]
        
        # Track large orders
        large_order_threshold = np.mean([
            np.mean([size for _, size in b.bids[:5]] + [size for _, size in b.asks[:5]])
            for b in recent_books
        ]) * 3
        
        # Count disappearing large orders
        spoof_count = 0
        for i in range(1, len(recent_books)):
            prev_large_bids = [(p, s) for p, s in recent_books[i-1].bids if s > large_order_threshold]
            prev_large_asks = [(p, s) for p, s in recent_books[i-1].asks if s > large_order_threshold]
            
            curr_prices_bids = [p for p, _ in recent_books[i].bids]
            curr_prices_asks = [p for p, _ in recent_books[i].asks]
            
            # Check if large orders disappeared
            for price, _ in prev_large_bids:
                if price not in curr_prices_bids:
                    spoof_count += 1
                    
            for price, _ in prev_large_asks:
                if price not in curr_prices_asks:
                    spoof_count += 1
        
        return spoof_count / (len(recent_books) - 1)


def add_orderflow_features(df: pl.DataFrame, book_data: Optional[List[OrderBookSnapshot]] = None) -> pl.DataFrame:
    """Add order flow features to candle data."""
    
    # Time and Sales (Tape) features
    df = df.with_columns([
        # Buy/sell volume ratio
        pl.when(pl.col("close") > pl.col("open"))
        .then(pl.col("volume") * 0.7)  # Estimate 70% as buy volume
        .otherwise(pl.col("volume") * 0.3)
        .alias("est_buy_volume"),
        
        pl.when(pl.col("close") < pl.col("open"))
        .then(pl.col("volume") * 0.7)  # Estimate 70% as sell volume
        .otherwise(pl.col("volume") * 0.3)
        .alias("est_sell_volume"),
    ])
    
    df = df.with_columns([
        # Order flow metrics
        (pl.col("est_buy_volume") - pl.col("est_sell_volume")).alias("net_order_flow"),
        
        (pl.col("est_buy_volume") / (pl.col("est_sell_volume") + 1)).alias("buy_sell_ratio"),
        
        # Large trade detection
        (pl.col("volume") > pl.col("volume").rolling_quantile(quantile=0.95, window_size=100))
        .alias("large_trade"),
        
        # Cumulative delta
        ((pl.col("est_buy_volume") - pl.col("est_sell_volume"))).cum_sum()
        .alias("cumulative_delta"),
    ])
    
    # Delta divergence
    df = df.with_columns([
        # Price up but delta down = bearish divergence
        ((pl.col("close") > pl.col("close").shift(1)) & 
         (pl.col("cumulative_delta") < pl.col("cumulative_delta").shift(1)))
        .alias("bearish_delta_divergence"),
        
        # Price down but delta up = bullish divergence
        ((pl.col("close") < pl.col("close").shift(1)) & 
         (pl.col("cumulative_delta") > pl.col("cumulative_delta").shift(1)))
        .alias("bullish_delta_divergence"),
    ])
    
    # Absorption detection (high volume, small price move)
    df = df.with_columns([
        # Buying absorption
        ((pl.col("volume") > pl.col("volume").rolling_mean(window_size=20) * 2) &
         ((pl.col("close") - pl.col("open")).abs() < pl.col("close").rolling_std(window_size=20) * 0.5) &
         (pl.col("close") < pl.col("open")))
        .alias("buying_absorption"),

        # Selling absorption
        ((pl.col("volume") > pl.col("volume").rolling_mean(window_size=20) * 2) &
         ((pl.col("close") - pl.col("open")).abs() < pl.col("close").rolling_std(window_size=20) * 0.5) &
         (pl.col("close") > pl.col("open")))
        .alias("selling_absorption"),
    ])
    
    # Institutional activity proxies
    df = df.with_columns([
        # Block trade detection
        (pl.col("volume") > pl.col("volume").rolling_mean(window_size=100) * 5)
        .alias("potential_block_trade"),

        # Iceberg order detection (consistent volume at price levels)
        (pl.col("volume").rolling_std(window_size=10) / (pl.col("volume").rolling_mean(window_size=10) + 1))
        .alias("volume_consistency"),

        # TWAP/VWAP execution detection
        ((pl.col("volume").rolling_std(window_size=20) / pl.col("volume").rolling_mean(window_size=20)) < 0.3)
        .alias("algo_execution_likely"),
    ])
    
    # Market maker participation
    df = df.with_columns([
        # Narrow range + high volume = MM accumulation/distribution
        (((pl.col("high") - pl.col("low")) < pl.col("close").rolling_std(window_size=50) * 0.5) &
        (pl.col("volume") > pl.col("volume").rolling_mean(window_size=50)))
        .alias("mm_participation"),

        # Two-way market making (balanced volume)
        (1 / (1 + (pl.col("buy_sell_ratio") - 1).abs())).alias("mm_balance_score"),
    ])
    
    return df


class RealTimeOrderFlow:
    """Real-time order flow processor for live trading."""
    
    def __init__(self):
        self.analyzer = OrderFlowAnalyzer()
        self.trade_tape = deque(maxlen=1000)
        self.current_features = {}
        self.mm_confidence = 0
        
    async def process_book_update(self, book: OrderBookSnapshot) -> Dict[str, any]:
        """Process order book update and generate signals."""
        
        # Extract features
        book_features = self.analyzer.process_book_update(book)
        self.current_features.update(book_features)
        
        # Update MM confidence
        self._update_mm_confidence(book_features)
        
        # Generate signals
        signals = self._generate_orderflow_signals()
        
        return {
            'features': self.current_features,
            'mm_confidence': self.mm_confidence,
            'signals': signals
        }
    
    async def process_trade(self, price: float, size: float, side: str, timestamp: float):
        """Process individual trade from tape."""
        
        trade = {
            'price': price,
            'size': size,
            'side': side,
            'timestamp': timestamp
        }
        
        self.trade_tape.append(trade)
        
        # Update tape-based features
        self._update_tape_features()
        
    def _update_mm_confidence(self, features: Dict[str, float]):
        """Update confidence in market maker presence."""
        
        mm_indicators = [
            features.get('mm_layering_detected', False),
            features.get('mm_spread_pattern', 0) > 0.8,
            features.get('quote_stuffing_score', 0) > 0.5,
            features.get('bid_layering_score', 0) > 0.7,
            features.get('ask_layering_score', 0) > 0.7,
        ]
        
        # Exponential moving average of MM presence
        current_mm = sum(mm_indicators) / len(mm_indicators)
        self.mm_confidence = 0.7 * self.mm_confidence + 0.3 * current_mm
        
    def _update_tape_features(self):
        """Update features from trade tape."""
        
        if len(self.trade_tape) < 10:
            return
        
        recent_trades = list(self.trade_tape)[-50:]
        
        # Trade size distribution
        sizes = [t['size'] for t in recent_trades]
        self.current_features['avg_trade_size'] = np.mean(sizes)
        self.current_features['large_trade_pct'] = sum(1 for s in sizes if s > np.mean(sizes) * 3) / len(sizes)
        
        # Buy/sell pressure from tape
        buy_volume = sum(t['size'] for t in recent_trades if t['side'] == 'buy')
        sell_volume = sum(t['size'] for t in recent_trades if t['side'] == 'sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            self.current_features['tape_buy_ratio'] = buy_volume / total_volume
            self.current_features['tape_order_flow'] = (buy_volume - sell_volume) / total_volume
        
        # Trade clustering (rapid trades)
        if len(recent_trades) > 1:
            time_gaps = [recent_trades[i]['timestamp'] - recent_trades[i-1]['timestamp'] 
                        for i in range(1, len(recent_trades))]
            self.current_features['trade_intensity'] = 1 / (np.mean(time_gaps) + 0.1)
            
    def _generate_orderflow_signals(self) -> Dict[str, bool]:
        """Generate trading signals from order flow."""
        
        signals = {}
        features = self.current_features
        
        # Momentum ignition
        signals['momentum_ignition_long'] = (
            features.get('weighted_imbalance', 0) > 0.3 and
            features.get('tape_buy_ratio', 0.5) > 0.7 and
            features.get('large_trade_pct', 0) > 0.1 and
            not features.get('selling_absorption', False)
        )
        
        signals['momentum_ignition_short'] = (
            features.get('weighted_imbalance', 0) < -0.3 and
            features.get('tape_buy_ratio', 0.5) < 0.3 and
            features.get('large_trade_pct', 0) > 0.1 and
            not features.get('buying_absorption', False)
        )
        
        # Liquidity grab setup
        signals['liquidity_grab_long'] = (
            features.get('bid_cliff_ratio', 1) > 3 and
            features.get('spoof_risk', 0) < 0.2 and
            features.get('bid_pressure', 0) > features.get('ask_pressure', 0) * 1.5
        )
        
        signals['liquidity_grab_short'] = (
            features.get('ask_cliff_ratio', 1) > 3 and
            features.get('spoof_risk', 0) < 0.2 and
            features.get('ask_pressure', 0) > features.get('bid_pressure', 0) * 1.5
        )
        
        # Market maker fade
        signals['fade_mm_long'] = (
            self.mm_confidence > 0.7 and
            features.get('mm_balance_score', 0) > 0.8 and
            features.get('weighted_imbalance', 0) < -0.2  # MM showing offers
        )
        
        signals['fade_mm_short'] = (
            self.mm_confidence > 0.7 and
            features.get('mm_balance_score', 0) > 0.8 and
            features.get('weighted_imbalance', 0) > 0.2  # MM showing bids
        )
        
        # Absorption plays
        signals['absorption_reversal_long'] = features.get('buying_absorption', False)
        signals['absorption_reversal_short'] = features.get('selling_absorption', False)
        
        # Risk warnings
        signals['high_spoof_risk'] = features.get('spoof_risk', 0) > 0.5
        signals['low_liquidity'] = (
            features.get('bid_concentration', 0) > 0.8 or
            features.get('ask_concentration', 0) > 0.8
        )
        
        return signals