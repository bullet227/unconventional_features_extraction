#!/usr/bin/env python3
# ml_pipeline/run_pipeline.py
"""
Complete ML Pipeline Runner for Forex Trading with Unconventional Features.

This script orchestrates the full pipeline:
1. Load OHLCV data from forex_trading_data database
2. Extract unconventional features
3. Process and select features
4. Train ML models with walk-forward validation
5. Backtest trading strategy
6. Generate evaluation reports

Usage:
    python run_pipeline.py --asset EURUSD --timeframe H1
    python run_pipeline.py --asset EURUSD --timeframe H1 --sample  # Use sample data
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline import (
    DataLoader,
    FeatureProcessor,
    ModelTrainer,
    EnsembleTrainer,
    Backtester,
    BacktestConfig,
    WalkForwardBacktester,
    Evaluator,
    create_sample_data,
    print_backtest_summary,
    print_training_summary,
)

# Import feature extraction
from unconventional_features import UnconventionalFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def run_full_pipeline(
    asset: str = 'EURUSD',
    timeframe: str = 'H1',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_sample_data: bool = False,
    model_type: str = 'xgboost',
    use_ensemble: bool = False,
    feature_selection: str = 'correlation',
    max_features: int = 100,
    n_splits: int = 5,
    output_dir: str = 'reports',
) -> dict:
    """
    Run the complete ML pipeline.

    Args:
        asset: Currency pair (e.g., 'EURUSD')
        timeframe: Timeframe (e.g., 'H1', 'M15', 'D')
        start_date: Start date for data (ISO format)
        end_date: End date for data
        use_sample_data: Use synthetic sample data instead of database
        model_type: 'xgboost', 'lightgbm', or 'ensemble'
        use_ensemble: Use ensemble of models
        feature_selection: 'correlation', 'importance', or 'shap'
        max_features: Maximum features to select
        n_splits: Number of walk-forward splits
        output_dir: Directory for reports

    Returns:
        Dictionary with pipeline results
    """
    results = {}

    log.info("=" * 60)
    log.info("STARTING ML PIPELINE")
    log.info(f"Asset: {asset}, Timeframe: {timeframe}")
    log.info("=" * 60)

    # Step 1: Load Data
    log.info("\n[Step 1/6] Loading OHLCV Data...")

    if use_sample_data:
        log.info("Using synthetic sample data")
        df = create_sample_data(n_rows=5000)
    else:
        loader = DataLoader()
        df = loader.load_ohlcv(asset, timeframe, start=start_date, end=end_date)

    log.info(f"Loaded {len(df)} candles")
    results['data_rows'] = len(df)

    # Step 2: Extract Unconventional Features
    log.info("\n[Step 2/6] Extracting Unconventional Features...")

    extractor = UnconventionalFeatureExtractor(enable_gpu_features=False)
    enriched_df = extractor.enrich(df, asset=asset, timeframe=timeframe)

    feature_cols = [c for c in enriched_df.columns
                   if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]
    log.info(f"Extracted {len(feature_cols)} features")
    results['total_features'] = len(feature_cols)

    # Step 3: Create Target and Prepare Data
    log.info("\n[Step 3/6] Preparing ML Dataset...")

    # Create target: 1 if next period close > current close
    target_horizon = 1
    enriched_df = enriched_df.with_columns([
        (pl.col('close').shift(-target_horizon) > pl.col('close')).cast(pl.Int32).alias('target')
    ])

    # Remove last N rows (no target)
    enriched_df = enriched_df.head(len(enriched_df) - target_horizon)

    # Separate features and target
    X = enriched_df.select(feature_cols)
    y = enriched_df['target']

    # Get timestamps for walk-forward
    timestamps = enriched_df['time'].to_list()
    prices = enriched_df['close'].to_numpy()
    highs = enriched_df['high'].to_numpy()
    lows = enriched_df['low'].to_numpy()

    log.info(f"Dataset: {len(X)} samples, {len(feature_cols)} features")

    # Step 4: Feature Processing and Selection
    log.info("\n[Step 4/6] Processing and Selecting Features...")

    processor = FeatureProcessor()

    # Handle missing values
    X_processed = processor.handle_missing_values(X)

    # Scale features
    X_scaled, scaler = processor.scale_features(X_processed)

    # Select features
    selection_result = processor.select_features(
        X_scaled, y,
        method=feature_selection,
        max_features=max_features,
    )

    # Filter to selected features
    X_selected = X_scaled.select(selection_result.selected_features)

    log.info(f"Selected {len(selection_result.selected_features)} features")
    log.info(f"Top 5 features: {selection_result.selected_features[:5]}")
    results['selected_features'] = len(selection_result.selected_features)

    # Step 5: Train Model
    log.info("\n[Step 5/6] Training Model...")

    if use_ensemble:
        trainer = EnsembleTrainer(
            models=['xgboost', 'lightgbm'],
            task='classification',
        )
        ensemble_results = trainer.train(X_selected, y, timestamps=timestamps)
        training_result = list(ensemble_results.values())[0]  # Use first for reporting
        model_for_backtest = trainer
    else:
        trainer = ModelTrainer(
            model_type=model_type,
            task='classification',
            n_splits=n_splits,
        )
        training_result = trainer.train(X_selected, y, timestamps=timestamps)
        model_for_backtest = trainer

    print_training_summary(training_result)
    results['training_metrics'] = {
        'accuracy': training_result.metrics.accuracy,
        'precision': training_result.metrics.precision,
        'recall': training_result.metrics.recall,
        'f1': training_result.metrics.f1,
    }

    # Generate predictions for backtest
    signals = model_for_backtest.predict_proba(X_selected)

    # Step 6: Backtest
    log.info("\n[Step 6/6] Running Backtest...")

    backtest_config = BacktestConfig(
        initial_capital=100000.0,
        spread_pips=1.0,
        slippage_pips=0.5,
        long_threshold=0.55,
        short_threshold=0.45,
        stop_loss_pips=30,
        take_profit_pips=60,
    )

    backtester = Backtester(config=backtest_config)
    backtest_result = backtester.run(
        prices=prices,
        signals=signals,
        timestamps=timestamps,
        highs=highs,
        lows=lows,
    )

    print_backtest_summary(backtest_result)
    results['backtest_metrics'] = {
        'total_return_pct': backtest_result.metrics.total_return_pct,
        'sharpe_ratio': backtest_result.metrics.sharpe_ratio,
        'max_drawdown_pct': backtest_result.metrics.max_drawdown_pct,
        'total_trades': backtest_result.metrics.total_trades,
        'win_rate': backtest_result.metrics.win_rate,
        'profit_factor': backtest_result.metrics.profit_factor,
    }

    # Generate Report
    log.info("\n[Generating Report...]")

    evaluator = Evaluator(output_dir=output_dir)
    report = evaluator.create_report(
        asset=asset,
        timeframe=timeframe,
        training_result=training_result,
        backtest_result=backtest_result,
        config={
            'model_type': model_type,
            'feature_selection': feature_selection,
            'n_splits': n_splits,
            'use_sample_data': use_sample_data,
        }
    )

    # Save reports
    json_path = evaluator.save_report(report, format='json')
    html_path = evaluator.save_report(report, format='html')

    log.info(f"Reports saved to: {json_path}")
    results['report_path'] = html_path

    # Save model
    model_path = os.path.join(output_dir, f"model_{asset}_{timeframe}")
    if hasattr(model_for_backtest, 'save'):
        model_for_backtest.save(model_path)
        log.info(f"Model saved to: {model_path}")

    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)

    return results


def run_walk_forward_backtest(
    asset: str = 'EURUSD',
    timeframe: str = 'H1',
    retrain_period: int = 500,
    use_sample_data: bool = False,
) -> dict:
    """
    Run walk-forward backtest with periodic model retraining.

    More realistic as it retrains the model as new data becomes available.
    """
    log.info("=" * 60)
    log.info("WALK-FORWARD BACKTEST")
    log.info("=" * 60)

    # Load data
    if use_sample_data:
        df = create_sample_data(n_rows=5000)
    else:
        loader = DataLoader()
        df = loader.load_ohlcv(asset, timeframe)

    # Extract features
    extractor = UnconventionalFeatureExtractor(enable_gpu_features=False)
    enriched_df = extractor.enrich(df, asset=asset, timeframe=timeframe)

    # Prepare data
    feature_cols = [c for c in enriched_df.columns
                   if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]

    enriched_df = enriched_df.with_columns([
        (pl.col('close').shift(-1) > pl.col('close')).cast(pl.Int32).alias('target')
    ])
    enriched_df = enriched_df.head(len(enriched_df) - 1)

    X = enriched_df.select(feature_cols)
    y = enriched_df['target']
    timestamps = enriched_df['time'].to_list()
    prices = enriched_df['close'].to_numpy()
    highs = enriched_df['high'].to_numpy()
    lows = enriched_df['low'].to_numpy()

    # Handle missing values
    processor = FeatureProcessor()
    X = processor.handle_missing_values(X)

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000.0,
        spread_pips=1.0,
        long_threshold=0.55,
    )

    trainer = ModelTrainer(model_type='xgboost')
    backtester = Backtester(config=config)

    wf_backtester = WalkForwardBacktester(
        model_trainer=trainer,
        backtester=backtester,
        retrain_period=retrain_period,
        warmup_period=500,
    )

    result = wf_backtester.run(
        X=X, y=y,
        prices=prices,
        timestamps=timestamps,
        highs=highs,
        lows=lows,
    )

    print_backtest_summary(result)

    return {
        'total_return_pct': result.metrics.total_return_pct,
        'sharpe_ratio': result.metrics.sharpe_ratio,
        'total_trades': result.metrics.total_trades,
    }


def main():
    parser = argparse.ArgumentParser(description='Run Forex ML Pipeline')
    parser.add_argument('--asset', type=str, default='EURUSD',
                        help='Currency pair (e.g., EURUSD)')
    parser.add_argument('--timeframe', type=str, default='H1',
                        help='Timeframe (e.g., H1, M15, D)')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--sample', action='store_true',
                        help='Use sample data instead of database')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm'],
                        help='Model type')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble of models')
    parser.add_argument('--features', type=str, default='correlation',
                        choices=['correlation', 'importance', 'shap'],
                        help='Feature selection method')
    parser.add_argument('--max-features', type=int, default=100,
                        help='Maximum features to select')
    parser.add_argument('--splits', type=int, default=5,
                        help='Number of walk-forward splits')
    parser.add_argument('--output', type=str, default='reports',
                        help='Output directory for reports')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Run walk-forward backtest with retraining')
    parser.add_argument('--retrain-period', type=int, default=500,
                        help='Bars between model retraining (walk-forward mode)')

    args = parser.parse_args()

    if args.walk_forward:
        results = run_walk_forward_backtest(
            asset=args.asset,
            timeframe=args.timeframe,
            retrain_period=args.retrain_period,
            use_sample_data=args.sample,
        )
    else:
        results = run_full_pipeline(
            asset=args.asset,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            use_sample_data=args.sample,
            model_type=args.model,
            use_ensemble=args.ensemble,
            feature_selection=args.features,
            max_features=args.max_features,
            n_splits=args.splits,
            output_dir=args.output,
        )

    print("\nFinal Results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
