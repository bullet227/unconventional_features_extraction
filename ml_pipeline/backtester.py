# ml_pipeline/backtester.py
"""
Backtesting engine for forex ML trading strategies.
Simulates trading based on model predictions with realistic conditions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import polars as pl
import pandas as pd

log = logging.getLogger(__name__)


class PositionType(Enum):
    """Trade position type."""
    LONG = 1
    SHORT = -1
    FLAT = 0


class PositionSizing(Enum):
    """Position sizing strategy."""
    FIXED = "fixed"              # Fixed lot size
    PERCENT_EQUITY = "percent"   # Percentage of equity
    KELLY = "kelly"              # Kelly criterion
    VOLATILITY = "volatility"    # Volatility-adjusted


@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_type: PositionType
    size: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_periods: int = 0
    exit_reason: str = ""

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestConfig:
    """Configuration for backtest simulation."""
    initial_capital: float = 100000.0
    position_sizing: PositionSizing = PositionSizing.FIXED
    fixed_size: float = 1.0           # Fixed lots
    risk_per_trade: float = 0.02      # 2% risk per trade
    max_position_size: float = 10.0   # Max lots

    # Costs
    spread_pips: float = 1.0          # Spread in pips
    commission_per_lot: float = 0.0   # Commission per lot
    slippage_pips: float = 0.5        # Slippage in pips

    # Risk management
    stop_loss_pips: Optional[float] = None   # Stop loss in pips
    take_profit_pips: Optional[float] = None # Take profit in pips
    max_holding_periods: int = 0      # 0 = no limit
    max_trades_per_day: int = 0       # 0 = no limit

    # Signal thresholds
    long_threshold: float = 0.5       # Probability threshold for long
    short_threshold: float = 0.5      # Probability threshold for short
    use_probability: bool = True      # Use probabilities vs binary signals

    # Pip value (for forex pairs - EURUSD standard)
    pip_value: float = 0.0001


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest."""
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0
    downside_volatility: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trading
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Average trade
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    avg_holding_period: float = 0.0

    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Expectancy
    expectancy: float = 0.0  # Expected value per trade
    edge_ratio: float = 0.0  # avg_winner / avg_loser


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: List[Trade]
    equity_curve: np.ndarray
    returns: np.ndarray
    drawdown_curve: np.ndarray
    timestamps: List[datetime]
    final_equity: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to DataFrame."""
        return pd.DataFrame({
            'time': self.timestamps,
            'equity': self.equity_curve,
            'returns': self.returns,
            'drawdown': self.drawdown_curve,
        })

    def trade_history_df(self) -> pd.DataFrame:
        """Convert trade history to DataFrame."""
        data = []
        for t in self.trades:
            data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'type': t.position_type.name,
                'size': t.size,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'holding_periods': t.holding_periods,
                'exit_reason': t.exit_reason,
                'is_winner': t.is_winner,
            })
        return pd.DataFrame(data)


class Backtester:
    """
    Event-driven backtesting engine for ML-based trading strategies.

    Simulates trading with:
    - Realistic costs (spread, slippage, commission)
    - Multiple position sizing strategies
    - Risk management (stop-loss, take-profit)
    - Detailed performance analytics

    Example usage:
        backtester = Backtester(config=BacktestConfig())
        result = backtester.run(
            prices=df['close'].to_numpy(),
            signals=model.predict_proba(features),
            timestamps=df['time'].to_list(),
            highs=df['high'].to_numpy(),
            lows=df['low'].to_numpy(),
        )
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()

        # State variables
        self._equity = 0.0
        self._position = PositionType.FLAT
        self._position_size = 0.0
        self._entry_price = 0.0
        self._entry_time = None
        self._entry_idx = 0

        # Tracking
        self._trades: List[Trade] = []
        self._equity_curve: List[float] = []
        self._daily_trades = 0
        self._last_day = None

    def reset(self):
        """Reset backtester state."""
        self._equity = self.config.initial_capital
        self._position = PositionType.FLAT
        self._position_size = 0.0
        self._entry_price = 0.0
        self._entry_time = None
        self._entry_idx = 0
        self._trades = []
        self._equity_curve = []
        self._daily_trades = 0
        self._last_day = None

    def _calculate_position_size(
        self,
        equity: float,
        price: float,
        volatility: Optional[float] = None,
    ) -> float:
        """Calculate position size based on strategy."""
        if self.config.position_sizing == PositionSizing.FIXED:
            size = self.config.fixed_size

        elif self.config.position_sizing == PositionSizing.PERCENT_EQUITY:
            # Risk-based: risk_per_trade of equity
            risk_amount = equity * self.config.risk_per_trade
            if self.config.stop_loss_pips:
                pip_value_per_lot = self.config.pip_value * 100000  # Standard lot
                risk_per_lot = self.config.stop_loss_pips * pip_value_per_lot
                size = risk_amount / risk_per_lot
            else:
                # Fallback: use percentage of equity
                lot_value = price * 100000
                size = (equity * self.config.risk_per_trade) / lot_value

        elif self.config.position_sizing == PositionSizing.VOLATILITY:
            # Volatility-adjusted position sizing
            if volatility and volatility > 0:
                target_risk = equity * self.config.risk_per_trade
                size = target_risk / (volatility * price * 100000)
            else:
                size = self.config.fixed_size

        elif self.config.position_sizing == PositionSizing.KELLY:
            # Kelly criterion (simplified)
            # Requires win_rate and avg_win/loss ratio
            # Use conservative fraction (1/4 Kelly)
            size = self.config.fixed_size * 0.25
        else:
            size = self.config.fixed_size

        # Apply limits
        size = min(size, self.config.max_position_size)
        size = max(size, 0.01)  # Minimum size

        return size

    def _apply_slippage(self, price: float, is_entry: bool, is_long: bool) -> float:
        """Apply slippage to price."""
        slippage = self.config.slippage_pips * self.config.pip_value

        if is_entry:
            # Entry: pay more for long, receive less for short
            if is_long:
                return price + slippage
            return price - slippage
        else:
            # Exit: receive less for long, pay more for short
            if is_long:
                return price - slippage
            return price + slippage

    def _apply_spread(self, price: float, is_long: bool) -> float:
        """Apply spread to price."""
        spread = self.config.spread_pips * self.config.pip_value
        if is_long:
            # Long entry: buy at ask (price + spread/2)
            return price + spread / 2
        else:
            # Short entry: sell at bid (price - spread/2)
            return price - spread / 2

    def _calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        position_type: PositionType,
        size: float,
    ) -> Tuple[float, float]:
        """Calculate trade PnL."""
        lot_value = 100000  # Standard lot

        if position_type == PositionType.LONG:
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price

        # PnL in account currency
        pnl = price_diff * lot_value * size

        # Commission
        pnl -= self.config.commission_per_lot * size * 2  # Entry + exit

        # PnL percentage
        position_value = entry_price * lot_value * size
        pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0.0

        return pnl, pnl_pct

    def _check_stop_loss(
        self,
        low: float,
        high: float,
        position_type: PositionType,
    ) -> Optional[float]:
        """Check if stop loss was hit. Returns exit price if stopped."""
        if not self.config.stop_loss_pips:
            return None

        stop_distance = self.config.stop_loss_pips * self.config.pip_value

        if position_type == PositionType.LONG:
            stop_price = self._entry_price - stop_distance
            if low <= stop_price:
                return stop_price
        else:
            stop_price = self._entry_price + stop_distance
            if high >= stop_price:
                return stop_price

        return None

    def _check_take_profit(
        self,
        low: float,
        high: float,
        position_type: PositionType,
    ) -> Optional[float]:
        """Check if take profit was hit. Returns exit price if hit."""
        if not self.config.take_profit_pips:
            return None

        tp_distance = self.config.take_profit_pips * self.config.pip_value

        if position_type == PositionType.LONG:
            tp_price = self._entry_price + tp_distance
            if high >= tp_price:
                return tp_price
        else:
            tp_price = self._entry_price - tp_distance
            if low <= tp_price:
                return tp_price

        return None

    def _open_position(
        self,
        idx: int,
        timestamp: datetime,
        price: float,
        position_type: PositionType,
        volatility: Optional[float] = None,
    ):
        """Open a new position."""
        # Apply costs
        is_long = position_type == PositionType.LONG
        entry_price = self._apply_spread(price, is_long)
        entry_price = self._apply_slippage(entry_price, True, is_long)

        # Calculate size
        size = self._calculate_position_size(self._equity, price, volatility)

        # Update state
        self._position = position_type
        self._position_size = size
        self._entry_price = entry_price
        self._entry_time = timestamp
        self._entry_idx = idx

        log.debug(f"Opened {position_type.name} @ {entry_price:.5f}, size={size:.2f}")

    def _close_position(
        self,
        idx: int,
        timestamp: datetime,
        price: float,
        reason: str = "signal",
    ) -> Trade:
        """Close current position and record trade."""
        is_long = self._position == PositionType.LONG

        # Apply costs
        exit_price = self._apply_slippage(price, False, is_long)

        # Calculate PnL
        pnl, pnl_pct = self._calculate_pnl(
            self._entry_price,
            exit_price,
            self._position,
            self._position_size,
        )

        # Record trade
        trade = Trade(
            entry_time=self._entry_time,
            exit_time=timestamp,
            entry_price=self._entry_price,
            exit_price=exit_price,
            position_type=self._position,
            size=self._position_size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_periods=idx - self._entry_idx,
            exit_reason=reason,
        )

        # Update equity
        self._equity += pnl

        # Reset position
        self._position = PositionType.FLAT
        self._position_size = 0.0
        self._entry_price = 0.0
        self._entry_time = None

        log.debug(f"Closed {trade.position_type.name} @ {exit_price:.5f}, PnL={pnl:.2f}")

        return trade

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        timestamps: List[datetime],
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            prices: Close prices array
            signals: Model signals (probabilities or binary predictions)
            timestamps: Datetime for each bar
            highs: High prices (for stop-loss checking)
            lows: Low prices (for stop-loss checking)
            volatility: Optional volatility for position sizing

        Returns:
            BacktestResult with performance metrics and trade history
        """
        self.reset()
        n = len(prices)

        # Validate inputs
        if len(signals) != n or len(timestamps) != n:
            raise ValueError("All inputs must have the same length")

        if highs is None:
            highs = prices
        if lows is None:
            lows = prices

        log.info(f"Starting backtest: {n} bars, {timestamps[0]} to {timestamps[-1]}")

        for i in range(n):
            price = prices[i]
            signal = signals[i]
            timestamp = timestamps[i]
            high = highs[i]
            low = lows[i]
            vol = volatility[i] if volatility is not None else None

            # Skip NaN
            if np.isnan(price) or np.isnan(signal):
                self._equity_curve.append(self._equity)
                continue

            # Track daily trades
            current_day = timestamp.date() if hasattr(timestamp, 'date') else None
            if current_day != self._last_day:
                self._daily_trades = 0
                self._last_day = current_day

            # Check if we have a position
            if self._position != PositionType.FLAT:
                # Check stop loss
                stop_price = self._check_stop_loss(low, high, self._position)
                if stop_price:
                    trade = self._close_position(i, timestamp, stop_price, "stop_loss")
                    self._trades.append(trade)
                    self._equity_curve.append(self._equity)
                    continue

                # Check take profit
                tp_price = self._check_take_profit(low, high, self._position)
                if tp_price:
                    trade = self._close_position(i, timestamp, tp_price, "take_profit")
                    self._trades.append(trade)
                    self._equity_curve.append(self._equity)
                    continue

                # Check max holding period
                holding = i - self._entry_idx
                if self.config.max_holding_periods > 0 and holding >= self.config.max_holding_periods:
                    trade = self._close_position(i, timestamp, price, "max_holding")
                    self._trades.append(trade)
                    self._equity_curve.append(self._equity)
                    continue

                # Check for exit signal
                if self.config.use_probability:
                    # Exit long if signal drops, exit short if signal rises
                    if self._position == PositionType.LONG and signal < self.config.short_threshold:
                        trade = self._close_position(i, timestamp, price, "signal")
                        self._trades.append(trade)
                    elif self._position == PositionType.SHORT and signal > self.config.long_threshold:
                        trade = self._close_position(i, timestamp, price, "signal")
                        self._trades.append(trade)
                else:
                    # Binary signal: reverse position
                    current_signal = 1 if self._position == PositionType.LONG else 0
                    if signal != current_signal:
                        trade = self._close_position(i, timestamp, price, "signal")
                        self._trades.append(trade)

            # Check for entry signal (only if flat)
            if self._position == PositionType.FLAT:
                # Check daily trade limit
                if self.config.max_trades_per_day > 0:
                    if self._daily_trades >= self.config.max_trades_per_day:
                        self._equity_curve.append(self._equity)
                        continue

                if self.config.use_probability:
                    if signal > self.config.long_threshold:
                        self._open_position(i, timestamp, price, PositionType.LONG, vol)
                        self._daily_trades += 1
                    elif signal < (1 - self.config.short_threshold):
                        # For binary classification, low probability = short signal
                        self._open_position(i, timestamp, price, PositionType.SHORT, vol)
                        self._daily_trades += 1
                else:
                    if signal == 1:
                        self._open_position(i, timestamp, price, PositionType.LONG, vol)
                        self._daily_trades += 1
                    elif signal == 0:
                        self._open_position(i, timestamp, price, PositionType.SHORT, vol)
                        self._daily_trades += 1

            self._equity_curve.append(self._equity)

        # Close any remaining position
        if self._position != PositionType.FLAT:
            trade = self._close_position(n - 1, timestamps[-1], prices[-1], "end_of_data")
            self._trades.append(trade)

        # Calculate metrics
        equity_curve = np.array(self._equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = np.insert(returns, 0, 0.0)  # Pad first return

        # Drawdown calculation
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak

        metrics = self._calculate_metrics(equity_curve, returns, drawdown, timestamps)

        log.info(f"Backtest complete: {metrics.total_trades} trades, "
                 f"Return={metrics.total_return_pct:.2f}%, "
                 f"Sharpe={metrics.sharpe_ratio:.2f}")

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=self._trades,
            equity_curve=equity_curve,
            returns=returns,
            drawdown_curve=drawdown,
            timestamps=timestamps,
            final_equity=self._equity,
        )

    def _calculate_metrics(
        self,
        equity_curve: np.ndarray,
        returns: np.ndarray,
        drawdown: np.ndarray,
        timestamps: List[datetime],
    ) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        metrics = BacktestMetrics()

        # Basic returns
        metrics.total_return = equity_curve[-1] - self.config.initial_capital
        metrics.total_return_pct = (metrics.total_return / self.config.initial_capital) * 100

        # Annualized return (assuming hourly data for forex)
        if len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]
            years = duration.total_seconds() / (365.25 * 24 * 3600)
            if years > 0:
                metrics.annualized_return = ((equity_curve[-1] / self.config.initial_capital) ** (1 / years) - 1) * 100

        # Risk metrics
        metrics.max_drawdown = np.max(drawdown) * self.config.initial_capital
        metrics.max_drawdown_pct = np.max(drawdown) * 100

        # Filter out NaN and zero returns
        valid_returns = returns[~np.isnan(returns) & (returns != 0)]

        if len(valid_returns) > 0:
            metrics.volatility = np.std(valid_returns) * np.sqrt(252 * 24)  # Annualized

            # Downside volatility (for Sortino)
            negative_returns = valid_returns[valid_returns < 0]
            if len(negative_returns) > 0:
                metrics.downside_volatility = np.std(negative_returns) * np.sqrt(252 * 24)

        # Risk-adjusted ratios
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_return = metrics.annualized_return / 100 - risk_free_rate

        if metrics.volatility > 0:
            metrics.sharpe_ratio = excess_return / metrics.volatility

        if metrics.downside_volatility > 0:
            metrics.sortino_ratio = excess_return / metrics.downside_volatility

        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown_pct

        # Trade statistics
        if self._trades:
            metrics.total_trades = len(self._trades)

            winners = [t for t in self._trades if t.is_winner]
            losers = [t for t in self._trades if not t.is_winner]

            metrics.winning_trades = len(winners)
            metrics.losing_trades = len(losers)
            metrics.win_rate = len(winners) / len(self._trades) * 100

            # PnL stats
            gross_profit = sum(t.pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0

            if gross_loss > 0:
                metrics.profit_factor = gross_profit / gross_loss

            metrics.avg_trade_pnl = np.mean([t.pnl for t in self._trades])
            metrics.avg_winner = np.mean([t.pnl for t in winners]) if winners else 0
            metrics.avg_loser = np.mean([t.pnl for t in losers]) if losers else 0
            metrics.avg_holding_period = np.mean([t.holding_periods for t in self._trades])

            # Streaks
            is_winner = [t.is_winner for t in self._trades]
            metrics.max_consecutive_wins = self._max_streak(is_winner, True)
            metrics.max_consecutive_losses = self._max_streak(is_winner, False)

            # Expectancy
            win_prob = metrics.win_rate / 100
            metrics.expectancy = (win_prob * metrics.avg_winner) - ((1 - win_prob) * abs(metrics.avg_loser))

            if metrics.avg_loser != 0:
                metrics.edge_ratio = abs(metrics.avg_winner / metrics.avg_loser)

        return metrics

    @staticmethod
    def _max_streak(values: List[bool], target: bool) -> int:
        """Find maximum consecutive occurrences of target value."""
        max_streak = 0
        current = 0
        for v in values:
            if v == target:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak


class WalkForwardBacktester:
    """
    Walk-forward backtesting that retrains model periodically.

    More realistic than single-train backtest as it:
    - Retrains model as new data becomes available
    - Tests on truly unseen data
    - Captures model decay over time
    """

    def __init__(
        self,
        model_trainer: Any,  # ModelTrainer instance
        backtester: Backtester,
        retrain_period: int = 252,  # Bars between retraining
        warmup_period: int = 500,   # Minimum training samples
    ):
        """
        Args:
            model_trainer: ModelTrainer instance for training
            backtester: Backtester instance for simulation
            retrain_period: Number of bars between model retraining
            warmup_period: Minimum samples before first prediction
        """
        self.model_trainer = model_trainer
        self.backtester = backtester
        self.retrain_period = retrain_period
        self.warmup_period = warmup_period

    def run(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        prices: np.ndarray,
        timestamps: List[datetime],
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run walk-forward backtest with periodic retraining.

        Args:
            X: Feature DataFrame
            y: Target Series
            prices: Close prices
            timestamps: Datetime for each bar
            highs: High prices
            lows: Low prices

        Returns:
            BacktestResult with performance metrics
        """
        n = len(X)
        signals = np.full(n, np.nan)

        log.info(f"Starting walk-forward backtest: {n} bars, "
                 f"retrain every {self.retrain_period} bars")

        # Convert to numpy
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        # Handle NaN
        nan_mask = np.any(np.isnan(X_np), axis=1) | np.isnan(y_np)

        last_train_idx = 0
        current_model = None

        for i in range(self.warmup_period, n):
            # Check if we need to retrain
            if i - last_train_idx >= self.retrain_period or current_model is None:
                # Train on all data up to this point
                train_end = i

                # Get valid training data
                X_train = X_np[:train_end]
                y_train = y_np[:train_end]
                valid_mask = ~nan_mask[:train_end]

                if np.sum(valid_mask) < self.warmup_period:
                    continue

                X_train = X_train[valid_mask]
                y_train = y_train[valid_mask]

                # Train model
                log.debug(f"Retraining at bar {i} with {len(X_train)} samples")
                current_model = self.model_trainer._create_model()
                current_model.fit(X_train, y_train)

                last_train_idx = i

            # Generate signal for current bar
            if current_model is not None and not nan_mask[i]:
                if hasattr(current_model, 'predict_proba'):
                    signals[i] = current_model.predict_proba(X_np[i:i+1])[0, 1]
                else:
                    signals[i] = current_model.predict(X_np[i:i+1])[0]

        # Run backtest with generated signals
        return self.backtester.run(
            prices=prices,
            signals=signals,
            timestamps=timestamps,
            highs=highs,
            lows=lows,
        )


def run_multiple_backtests(
    backtester: Backtester,
    prices: np.ndarray,
    signals: np.ndarray,
    timestamps: List[datetime],
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
    param_grid: Optional[Dict[str, List]] = None,
) -> List[BacktestResult]:
    """
    Run multiple backtests with different parameters.

    Args:
        backtester: Base Backtester instance
        prices, signals, timestamps, highs, lows: Backtest data
        param_grid: Dictionary of parameter names to lists of values

    Returns:
        List of BacktestResult for each parameter combination
    """
    from itertools import product

    if param_grid is None:
        param_grid = {
            'long_threshold': [0.5, 0.55, 0.6],
            'stop_loss_pips': [20, 30, 50],
        }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    results = []

    for combo in combinations:
        # Update config
        for key, val in zip(keys, combo):
            setattr(backtester.config, key, val)

        log.info(f"Running backtest with {dict(zip(keys, combo))}")

        result = backtester.run(
            prices=prices,
            signals=signals,
            timestamps=timestamps,
            highs=highs,
            lows=lows,
        )
        results.append(result)

    return results
