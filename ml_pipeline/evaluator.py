# ml_pipeline/evaluator.py
"""
Evaluation and reporting system for forex ML pipeline.
Provides comprehensive metrics, visualization, and reports.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

import numpy as np
import polars as pl
import pandas as pd

from .backtester import BacktestResult, BacktestMetrics, Trade
from .model_trainer import TrainingResult, ModelMetrics

log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class FeatureAnalysis:
    """Analysis of feature importance and contribution."""
    feature_name: str
    importance: float
    domain: str
    shap_value: Optional[float] = None
    correlation_with_target: Optional[float] = None
    stability_score: Optional[float] = None  # Across folds


@dataclass
class ModelComparison:
    """Comparison between multiple models."""
    models: List[str]
    metrics: Dict[str, Dict[str, float]]
    best_model: str
    best_metric_name: str


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    asset: str
    timeframe: str
    training_metrics: Optional[ModelMetrics]
    backtest_metrics: Optional[BacktestMetrics]
    feature_analysis: List[FeatureAnalysis]
    model_comparison: Optional[ModelComparison]
    recommendations: List[str]
    config_summary: Dict[str, Any]


class Evaluator:
    """
    Comprehensive evaluation and reporting system.

    Provides:
    - Feature importance analysis across domains
    - Model performance comparison
    - Backtest performance attribution
    - Actionable recommendations
    - Export to JSON/HTML reports
    """

    # Feature domains from the unconventional features
    FEATURE_DOMAINS = {
        'session': ['session_', 'tokyo_', 'london_', 'ny_'],
        'stop_hunt': ['stop_hunt_', 'sweep_'],
        'sentiment': ['sentiment_', 'contrarian_', 'crowd_', 'fomo_', 'capitulation_'],
        'retail_trap': ['bull_trap', 'bear_trap', 'trap_'],
        'fibonacci': ['fib_', 'golden_'],
        'lunar': ['lunar_', 'moon_'],
        'chaos': ['lyapunov_', 'entropy_', 'hurst_', 'fractal_'],
        'social_physics': ['herding_', 'social_', 'cascade_'],
        'market_psychology': ['fear_', 'greed_', 'psychology_'],
        'hft': ['hft_', 'microsecond_', 'latency_'],
        'orderflow': ['orderflow_', 'mm_', 'market_maker_', 'iceberg_'],
        'intracandle': ['intracandle_', 'wick_', 'body_ratio_'],
        'quantum': ['quantum_', 'tunneling_', 'superposition_'],
        'game_theory': ['nash_', 'minimax_', 'game_'],
        'neural_oscillation': ['neural_', 'brain_wave_', 'oscillation_'],
        'astro': ['astro_', 'mercury_', 'venus_', 'mars_', 'jupiter_'],
    }

    def __init__(self, output_dir: str = "reports"):
        """
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def classify_feature_domain(self, feature_name: str) -> str:
        """Classify a feature into its domain."""
        feature_lower = feature_name.lower()
        for domain, prefixes in self.FEATURE_DOMAINS.items():
            for prefix in prefixes:
                if prefix in feature_lower:
                    return domain
        return 'other'

    def analyze_features(
        self,
        training_result: TrainingResult,
        target_correlation: Optional[Dict[str, float]] = None,
        stability_scores: Optional[Dict[str, float]] = None,
    ) -> List[FeatureAnalysis]:
        """
        Analyze feature importance and categorize by domain.

        Args:
            training_result: Result from model training
            target_correlation: Optional correlation with target
            stability_scores: Optional cross-fold stability scores

        Returns:
            List of FeatureAnalysis objects
        """
        analyses = []

        for feature, importance in training_result.feature_importance.items():
            domain = self.classify_feature_domain(feature)

            analysis = FeatureAnalysis(
                feature_name=feature,
                importance=importance,
                domain=domain,
                correlation_with_target=target_correlation.get(feature) if target_correlation else None,
                stability_score=stability_scores.get(feature) if stability_scores else None,
            )
            analyses.append(analysis)

        # Sort by importance
        analyses.sort(key=lambda x: x.importance, reverse=True)

        return analyses

    def get_domain_importance(
        self,
        feature_analyses: List[FeatureAnalysis],
    ) -> Dict[str, float]:
        """
        Aggregate feature importance by domain.

        Returns:
            Dictionary mapping domain to total importance
        """
        domain_importance = {}

        for analysis in feature_analyses:
            domain = analysis.domain
            if domain not in domain_importance:
                domain_importance[domain] = 0.0
            domain_importance[domain] += analysis.importance

        # Sort by importance
        return dict(sorted(domain_importance.items(), key=lambda x: x[1], reverse=True))

    def compare_models(
        self,
        results: Dict[str, TrainingResult],
        primary_metric: str = 'f1',
    ) -> ModelComparison:
        """
        Compare multiple model training results.

        Args:
            results: Dictionary mapping model names to TrainingResult
            primary_metric: Metric to use for ranking

        Returns:
            ModelComparison with rankings
        """
        metrics = {}

        for name, result in results.items():
            m = result.metrics
            metrics[name] = {
                'accuracy': m.accuracy,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'auc_roc': m.auc_roc,
                'win_rate': m.win_rate,
            }

        # Find best model
        best_model = max(results.keys(), key=lambda x: metrics[x][primary_metric])

        return ModelComparison(
            models=list(results.keys()),
            metrics=metrics,
            best_model=best_model,
            best_metric_name=primary_metric,
        )

    def analyze_trades(
        self,
        trades: List[Trade],
    ) -> Dict[str, Any]:
        """
        Analyze trade patterns and statistics.

        Returns:
            Dictionary with trade analysis
        """
        if not trades:
            return {}

        # Basic stats
        pnls = [t.pnl for t in trades]
        holdings = [t.holding_periods for t in trades]

        # By position type
        long_trades = [t for t in trades if t.position_type.value == 1]
        short_trades = [t for t in trades if t.position_type.value == -1]

        # By exit reason
        exit_reasons = {}
        for t in trades:
            reason = t.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0.0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += t.pnl

        # Time analysis
        hour_performance = {}
        for t in trades:
            if t.entry_time:
                hour = t.entry_time.hour
                if hour not in hour_performance:
                    hour_performance[hour] = {'count': 0, 'pnl': 0.0}
                hour_performance[hour]['count'] += 1
                hour_performance[hour]['pnl'] += t.pnl

        return {
            'total_trades': len(trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_pnl': sum(t.pnl for t in long_trades),
            'short_pnl': sum(t.pnl for t in short_trades),
            'avg_holding': np.mean(holdings),
            'median_holding': np.median(holdings),
            'pnl_distribution': {
                'mean': np.mean(pnls),
                'std': np.std(pnls),
                'median': np.median(pnls),
                'min': np.min(pnls),
                'max': np.max(pnls),
            },
            'by_exit_reason': exit_reasons,
            'by_hour': hour_performance,
        }

    def generate_recommendations(
        self,
        training_result: Optional[TrainingResult] = None,
        backtest_result: Optional[BacktestResult] = None,
        feature_analyses: Optional[List[FeatureAnalysis]] = None,
    ) -> List[str]:
        """
        Generate actionable recommendations based on results.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if training_result:
            metrics = training_result.metrics

            # Model performance recommendations
            if metrics.accuracy < 0.52:
                recommendations.append(
                    "Model accuracy is near random. Consider feature engineering "
                    "or using different model architecture."
                )
            elif metrics.accuracy > 0.65:
                recommendations.append(
                    "High accuracy detected. Verify no data leakage and check "
                    "out-of-sample performance carefully."
                )

            if metrics.precision < 0.5:
                recommendations.append(
                    "Low precision (many false positives). Increase prediction "
                    "threshold or add confirmation signals."
                )

            if metrics.recall < 0.5:
                recommendations.append(
                    "Low recall (missing opportunities). Consider lowering "
                    "threshold for more aggressive entries."
                )

        if backtest_result:
            bt_metrics = backtest_result.metrics

            # Risk recommendations
            if bt_metrics.max_drawdown_pct > 20:
                recommendations.append(
                    f"Max drawdown of {bt_metrics.max_drawdown_pct:.1f}% is high. "
                    "Consider tighter stop-losses or smaller position sizes."
                )

            if bt_metrics.sharpe_ratio < 1.0:
                recommendations.append(
                    f"Sharpe ratio of {bt_metrics.sharpe_ratio:.2f} is below 1. "
                    "Risk-adjusted returns may not justify the volatility."
                )

            if bt_metrics.profit_factor < 1.5:
                recommendations.append(
                    f"Profit factor of {bt_metrics.profit_factor:.2f} suggests thin edge. "
                    "Focus on reducing costs or improving win rate."
                )

            # Win rate recommendations
            if bt_metrics.win_rate < 40:
                recommendations.append(
                    f"Low win rate ({bt_metrics.win_rate:.1f}%). "
                    "Ensure average winner is significantly larger than average loser."
                )
            elif bt_metrics.win_rate > 70:
                recommendations.append(
                    f"Very high win rate ({bt_metrics.win_rate:.1f}%). "
                    "Verify this isn't due to overly tight take-profits leaving money on table."
                )

            # Trade frequency
            if bt_metrics.total_trades < 30:
                recommendations.append(
                    "Low trade count for statistical significance. "
                    "Consider longer test period or relaxing entry criteria."
                )

        if feature_analyses:
            # Domain-level recommendations
            domain_importance = self.get_domain_importance(feature_analyses)

            top_domains = list(domain_importance.keys())[:3]
            if top_domains:
                recommendations.append(
                    f"Top performing feature domains: {', '.join(top_domains)}. "
                    "Consider developing more features in these areas."
                )

            # Low importance domains
            bottom_domains = [d for d, imp in domain_importance.items() if imp < 0.01]
            if bottom_domains:
                recommendations.append(
                    f"Low-impact domains: {', '.join(bottom_domains)}. "
                    "Consider removing or replacing these features."
                )

        if not recommendations:
            recommendations.append("Results look reasonable. Continue monitoring and validating.")

        return recommendations

    def create_report(
        self,
        asset: str,
        timeframe: str,
        training_result: Optional[TrainingResult] = None,
        backtest_result: Optional[BacktestResult] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """
        Create comprehensive evaluation report.

        Args:
            asset: Currency pair
            timeframe: Timeframe
            training_result: Optional training results
            backtest_result: Optional backtest results
            config: Optional configuration dict

        Returns:
            EvaluationReport object
        """
        # Analyze features
        feature_analyses = []
        if training_result:
            feature_analyses = self.analyze_features(training_result)

        # Generate recommendations
        recommendations = self.generate_recommendations(
            training_result,
            backtest_result,
            feature_analyses,
        )

        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            asset=asset,
            timeframe=timeframe,
            training_metrics=training_result.metrics if training_result else None,
            backtest_metrics=backtest_result.metrics if backtest_result else None,
            feature_analysis=feature_analyses,
            model_comparison=None,
            recommendations=recommendations,
            config_summary=config or {},
        )

        return report

    def save_report(
        self,
        report: EvaluationReport,
        format: str = 'json',
        filename: Optional[str] = None,
    ) -> str:
        """
        Save report to file.

        Args:
            report: EvaluationReport to save
            format: 'json' or 'html'
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"report_{report.asset}_{report.timeframe}_{timestamp}"

        if format == 'json':
            filepath = os.path.join(self.output_dir, f"{filename}.json")
            self._save_json(report, filepath)
        elif format == 'html':
            filepath = os.path.join(self.output_dir, f"{filename}.html")
            self._save_html(report, filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

        log.info(f"Saved report to {filepath}")
        return filepath

    def _save_json(self, report: EvaluationReport, filepath: str):
        """Save report as JSON."""
        data = {
            'timestamp': report.timestamp,
            'asset': report.asset,
            'timeframe': report.timeframe,
            'training_metrics': asdict(report.training_metrics) if report.training_metrics else None,
            'backtest_metrics': asdict(report.backtest_metrics) if report.backtest_metrics else None,
            'feature_analysis': [asdict(f) for f in report.feature_analysis],
            'recommendations': report.recommendations,
            'config_summary': report.config_summary,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def _save_html(self, report: EvaluationReport, filepath: str):
        """Save report as HTML."""
        html = self._generate_html(report)
        with open(filepath, 'w') as f:
            f.write(html)

    def _generate_html(self, report: EvaluationReport) -> str:
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ML Trading Report - {report.asset}/{report.timeframe}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f8f8; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .recommendation {{ background: #fff3cd; padding: 10px 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; font-weight: bold; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        .footer {{ margin-top: 40px; text-align: center; color: #888; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ML Trading Report</h1>
        <p><strong>Asset:</strong> {report.asset} | <strong>Timeframe:</strong> {report.timeframe} | <strong>Generated:</strong> {report.timestamp}</p>
"""

        # Training Metrics
        if report.training_metrics:
            m = report.training_metrics
            html += f"""
        <h2>Model Performance</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{m.accuracy*100:.1f}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.precision*100:.1f}%</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.recall*100:.1f}%</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.f1*100:.1f}%</div>
                <div class="metric-label">F1 Score</div>
            </div>
        </div>
"""

        # Backtest Metrics
        if report.backtest_metrics:
            m = report.backtest_metrics
            return_class = 'positive' if m.total_return_pct > 0 else 'negative'
            html += f"""
        <h2>Backtest Results</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value {return_class}">{m.total_return_pct:.1f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">-{m.max_drawdown_pct:.1f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{m.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.avg_winner:.0f}</div>
                <div class="metric-label">Avg Winner</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.avg_loser:.0f}</div>
                <div class="metric-label">Avg Loser</div>
            </div>
        </div>
"""

        # Feature Analysis
        if report.feature_analysis:
            domain_importance = {}
            for f in report.feature_analysis:
                if f.domain not in domain_importance:
                    domain_importance[f.domain] = 0.0
                domain_importance[f.domain] += f.importance

            html += """
        <h2>Feature Domain Analysis</h2>
        <table>
            <tr><th>Domain</th><th>Total Importance</th></tr>
"""
            for domain, importance in sorted(domain_importance.items(), key=lambda x: x[1], reverse=True):
                html += f"            <tr><td>{domain}</td><td>{importance:.4f}</td></tr>\n"

            html += """        </table>

        <h2>Top 20 Features</h2>
        <table>
            <tr><th>Feature</th><th>Domain</th><th>Importance</th></tr>
"""
            for f in report.feature_analysis[:20]:
                html += f"            <tr><td>{f.feature_name}</td><td>{f.domain}</td><td>{f.importance:.4f}</td></tr>\n"

            html += "        </table>\n"

        # Recommendations
        html += """
        <h2>Recommendations</h2>
"""
        for rec in report.recommendations:
            html += f'        <div class="recommendation">{rec}</div>\n'

        # Footer
        html += """
        <div class="footer">
            Generated by Unconventional Features ML Pipeline
        </div>
    </div>
</body>
</html>
"""
        return html


class PerformanceTracker:
    """
    Track performance over time for production monitoring.
    """

    def __init__(self, metrics_file: str = "performance_history.json"):
        """
        Args:
            metrics_file: File to store historical metrics
        """
        self.metrics_file = metrics_file
        self.history: List[Dict] = []

        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                self.history = json.load(f)

    def log_performance(
        self,
        asset: str,
        timeframe: str,
        metrics: Dict[str, float],
        model_version: str = "",
    ):
        """Log performance metrics."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'asset': asset,
            'timeframe': timeframe,
            'model_version': model_version,
            'metrics': metrics,
        }

        self.history.append(entry)

        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_performance_trend(
        self,
        asset: str,
        timeframe: str,
        metric: str = 'sharpe_ratio',
        n_last: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get performance trend for a specific metric."""
        filtered = [
            h for h in self.history
            if h['asset'] == asset and h['timeframe'] == timeframe
        ]

        trend = [
            (h['timestamp'], h['metrics'].get(metric, 0.0))
            for h in filtered[-n_last:]
        ]

        return trend

    def check_performance_degradation(
        self,
        asset: str,
        timeframe: str,
        metric: str = 'sharpe_ratio',
        threshold: float = 0.2,
        window: int = 5,
    ) -> bool:
        """
        Check if performance has degraded significantly.

        Returns True if recent performance is >threshold worse than historical.
        """
        filtered = [
            h['metrics'].get(metric, 0.0)
            for h in self.history
            if h['asset'] == asset and h['timeframe'] == timeframe
        ]

        if len(filtered) < window * 2:
            return False

        historical_avg = np.mean(filtered[:-window])
        recent_avg = np.mean(filtered[-window:])

        degradation = (historical_avg - recent_avg) / abs(historical_avg) if historical_avg != 0 else 0

        return degradation > threshold


def print_backtest_summary(result: BacktestResult):
    """Print formatted backtest summary to console."""
    m = result.metrics
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    print(f"\n{'RETURNS':-^40}")
    print(f"  Total Return:      ${m.total_return:,.2f} ({m.total_return_pct:.2f}%)")
    print(f"  Annualized Return: {m.annualized_return:.2f}%")

    print(f"\n{'RISK':-^40}")
    print(f"  Max Drawdown:      {m.max_drawdown_pct:.2f}%")
    print(f"  Volatility:        {m.volatility*100:.2f}%")

    print(f"\n{'RISK-ADJUSTED':-^40}")
    print(f"  Sharpe Ratio:      {m.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:     {m.sortino_ratio:.2f}")
    print(f"  Calmar Ratio:      {m.calmar_ratio:.2f}")

    print(f"\n{'TRADING':-^40}")
    print(f"  Total Trades:      {m.total_trades}")
    print(f"  Win Rate:          {m.win_rate:.1f}%")
    print(f"  Profit Factor:     {m.profit_factor:.2f}")
    print(f"  Expectancy:        ${m.expectancy:.2f}")

    print(f"\n{'AVERAGE TRADE':-^40}")
    print(f"  Avg Trade PnL:     ${m.avg_trade_pnl:.2f}")
    print(f"  Avg Winner:        ${m.avg_winner:.2f}")
    print(f"  Avg Loser:         ${m.avg_loser:.2f}")
    print(f"  Edge Ratio:        {m.edge_ratio:.2f}")

    print(f"\n{'STREAKS':-^40}")
    print(f"  Max Consec. Wins:  {m.max_consecutive_wins}")
    print(f"  Max Consec. Losses:{m.max_consecutive_losses}")

    print("\n" + "=" * 60)


def print_training_summary(result: TrainingResult):
    """Print formatted training summary to console."""
    m = result.metrics
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    print(f"\n  Model Type:        {result.model_type}")
    print(f"  Training Time:     {result.training_time:.1f}s")

    print(f"\n{'CLASSIFICATION METRICS':-^40}")
    print(f"  Accuracy:          {m.accuracy:.4f} ({m.accuracy*100:.2f}%)")
    print(f"  Precision:         {m.precision:.4f}")
    print(f"  Recall:            {m.recall:.4f}")
    print(f"  F1 Score:          {m.f1:.4f}")
    print(f"  AUC-ROC:           {m.auc_roc:.4f}")

    # Top features
    print(f"\n{'TOP 10 FEATURES':-^40}")
    sorted_features = sorted(
        result.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for i, (feature, importance) in enumerate(sorted_features, 1):
        print(f"  {i:2d}. {feature[:30]:30s} {importance:.4f}")

    print("\n" + "=" * 60)
