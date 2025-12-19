# ml_pipeline/__init__.py
"""
ML Pipeline for Forex Trading using Unconventional Features

Components:
- DataLoader: Load OHLCV data and store/retrieve features
- FeatureProcessor: Feature selection, scaling, and preprocessing
- ModelTrainer: XGBoost/LightGBM training with walk-forward validation
- Backtester: Event-driven backtesting with realistic costs
- Evaluator: Performance analysis and reporting

Example Usage:
    from ml_pipeline import DataLoader, FeatureProcessor, ModelTrainer, Backtester, Evaluator

    # Load data
    loader = DataLoader()
    df = loader.load_ohlcv('EURUSD', 'H1', start='2023-01-01')

    # Process features
    processor = FeatureProcessor()
    X_processed, selected_features = processor.select_features(X, y)

    # Train model
    trainer = ModelTrainer(model_type='xgboost')
    result = trainer.train(X_processed, y)

    # Backtest
    backtester = Backtester()
    bt_result = backtester.run(prices, signals, timestamps)

    # Evaluate
    evaluator = Evaluator()
    report = evaluator.create_report('EURUSD', 'H1', result, bt_result)
"""

from .data_loader import DataLoader, create_sample_data
from .feature_processor import (
    FeatureProcessor,
    FeatureStats,
    FeatureSelectionResult,
    FEATURE_DOMAINS,
)
from .model_trainer import (
    ModelTrainer,
    EnsembleTrainer,
    WalkForwardValidator,
    WalkForwardSplit,
    ModelMetrics,
    TrainingResult,
)
from .backtester import (
    Backtester,
    WalkForwardBacktester,
    BacktestConfig,
    BacktestResult,
    BacktestMetrics,
    Trade,
    PositionType,
    PositionSizing,
    run_multiple_backtests,
)
from .evaluator import (
    Evaluator,
    PerformanceTracker,
    EvaluationReport,
    FeatureAnalysis,
    ModelComparison,
    print_backtest_summary,
    print_training_summary,
)

__all__ = [
    # Data Loading
    'DataLoader',
    'create_sample_data',

    # Feature Processing
    'FeatureProcessor',
    'FeatureStats',
    'FeatureSelectionResult',
    'FEATURE_DOMAINS',

    # Model Training
    'ModelTrainer',
    'EnsembleTrainer',
    'WalkForwardValidator',
    'WalkForwardSplit',
    'ModelMetrics',
    'TrainingResult',

    # Backtesting
    'Backtester',
    'WalkForwardBacktester',
    'BacktestConfig',
    'BacktestResult',
    'BacktestMetrics',
    'Trade',
    'PositionType',
    'PositionSizing',
    'run_multiple_backtests',

    # Evaluation
    'Evaluator',
    'PerformanceTracker',
    'EvaluationReport',
    'FeatureAnalysis',
    'ModelComparison',
    'print_backtest_summary',
    'print_training_summary',
]

__version__ = '0.1.0'
