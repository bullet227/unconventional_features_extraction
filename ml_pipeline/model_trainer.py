# ml_pipeline/model_trainer.py
"""
Model training framework with walk-forward validation for forex ML.
Supports XGBoost, LightGBM, and ensemble methods.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import json
import os

import numpy as np
import polars as pl
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """Single walk-forward split."""
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_dates: Tuple[datetime, datetime] = None
    test_dates: Tuple[datetime, datetime] = None


@dataclass
class ModelMetrics:
    """Metrics for a single model evaluation."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0


@dataclass
class TrainingResult:
    """Results from model training."""
    model: Any
    model_type: str
    metrics: ModelMetrics
    feature_importance: Dict[str, float]
    fold_metrics: List[ModelMetrics]
    params: Dict[str, Any]
    training_time: float = 0.0


class WalkForwardValidator:
    """
    Walk-forward (expanding window) cross-validation for time series.

    Unlike standard k-fold, this respects temporal ordering:
    - Training set always precedes test set
    - Training window expands over time
    - Prevents look-ahead bias

    Example splits with n_splits=3:
        Fold 1: [========Train========][Test]
        Fold 2: [==========Train==========][Test]
        Fold 3: [============Train============][Test]
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.1,
        gap: int = 0,
        min_train_size: float = 0.3,
    ):
        """
        Args:
            n_splits: Number of walk-forward splits
            test_size: Fraction of data for each test set
            gap: Number of samples between train and test (avoid leakage)
            min_train_size: Minimum fraction of data for first training set
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.min_train_size = min_train_size

    def split(
        self,
        X: Union[pl.DataFrame, np.ndarray],
        y: Optional[Union[pl.Series, np.ndarray]] = None,
        timestamps: Optional[List[datetime]] = None,
    ) -> List[WalkForwardSplit]:
        """
        Generate walk-forward splits.

        Returns:
            List of WalkForwardSplit objects
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        test_samples = int(n_samples * self.test_size)
        min_train = int(n_samples * self.min_train_size)

        # Calculate split points
        splits = []
        remaining = n_samples - min_train
        increment = remaining // self.n_splits

        for fold in range(self.n_splits):
            train_end = min_train + fold * increment
            test_start = train_end + self.gap
            test_end = min(test_start + test_samples, n_samples)

            if test_end <= test_start:
                break

            split = WalkForwardSplit(
                fold=fold,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            # Add dates if timestamps provided
            if timestamps:
                split.train_dates = (timestamps[0], timestamps[train_end - 1])
                split.test_dates = (timestamps[test_start], timestamps[test_end - 1])

            splits.append(split)

        log.info(f"Created {len(splits)} walk-forward splits")
        return splits

    def get_split_data(
        self,
        split: WalkForwardSplit,
        X: Union[pl.DataFrame, np.ndarray],
        y: Union[pl.Series, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test data for a specific split."""
        # Convert to numpy
        X_np = X.to_numpy() if isinstance(X, pl.DataFrame) else X
        y_np = y.to_numpy() if isinstance(y, pl.Series) else y

        X_train = X_np[split.train_start:split.train_end]
        y_train = y_np[split.train_start:split.train_end]
        X_test = X_np[split.test_start:split.test_end]
        y_test = y_np[split.test_start:split.test_end]

        return X_train, X_test, y_train, y_test


class ModelTrainer:
    """
    Train and evaluate ML models with walk-forward validation.

    Supports:
    - XGBoost (gradient boosting)
    - LightGBM (gradient boosting)
    - Ensemble methods

    Example usage:
        trainer = ModelTrainer(model_type='xgboost')
        result = trainer.train(X, y, timestamps=df['time'].to_list())
        predictions = trainer.predict(X_new)
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        task: str = 'classification',
        params: Optional[Dict] = None,
        n_splits: int = 5,
        early_stopping: int = 50,
    ):
        """
        Args:
            model_type: 'xgboost', 'lightgbm', or 'ensemble'
            task: 'classification' (direction) or 'regression' (returns)
            params: Model hyperparameters
            n_splits: Number of walk-forward splits
            early_stopping: Early stopping rounds (0 to disable)
        """
        self.model_type = model_type
        self.task = task
        self.n_splits = n_splits
        self.early_stopping = early_stopping

        # Default parameters
        self.params = self._get_default_params()
        if params:
            self.params.update(params)

        self.model = None
        self.validator = WalkForwardValidator(n_splits=n_splits)
        self._training_result: Optional[TrainingResult] = None

    def _get_default_params(self) -> Dict:
        """Get default parameters for model type."""
        if self.model_type == 'xgboost':
            return {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_weight': 3,
                'n_jobs': -1,
                'verbosity': 0,
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_samples': 20,
                'n_jobs': -1,
                'verbosity': -1,
            }
        else:
            return {}

    def _create_model(self):
        """Create model instance based on type and task."""
        if self.model_type == 'xgboost':
            import xgboost as xgb
            if self.task == 'classification':
                return xgb.XGBClassifier(**self.params)
            else:
                return xgb.XGBRegressor(**self.params)

        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            if self.task == 'classification':
                return lgb.LGBMClassifier(**self.params)
            else:
                return lgb.LGBMRegressor(**self.params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> ModelMetrics:
        """Compute evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )

        metrics = ModelMetrics()

        if self.task == 'classification':
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(y_true, y_pred, zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, zero_division=0)
            metrics.f1 = f1_score(y_true, y_pred, zero_division=0)

            if y_proba is not None:
                try:
                    metrics.auc_roc = roc_auc_score(y_true, y_proba)
                except ValueError:
                    pass

            # Win rate for trading
            metrics.win_rate = metrics.precision  # Precision = % of positive predictions that were correct

        else:
            # Regression metrics
            metrics.avg_return = float(np.mean(y_pred))

        return metrics

    def train(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        timestamps: Optional[List[datetime]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> TrainingResult:
        """
        Train model with walk-forward validation.

        Args:
            X: Feature DataFrame
            y: Target Series
            timestamps: Optional list of timestamps for time-based splits
            feature_names: Optional feature names

        Returns:
            TrainingResult with trained model and metrics
        """
        import time
        start_time = time.time()

        # Convert to numpy
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        # Handle NaN values
        nan_mask = np.any(np.isnan(X_np), axis=1) | np.isnan(y_np)
        X_np = X_np[~nan_mask]
        y_np = y_np[~nan_mask]
        if timestamps:
            timestamps = [t for t, m in zip(timestamps, ~nan_mask) if m]

        feature_names = feature_names or list(X.columns)

        # Walk-forward validation
        splits = self.validator.split(X_np, y_np, timestamps)
        fold_metrics = []
        fold_importance = []

        for split in splits:
            X_train, X_test, y_train, y_test = self.validator.get_split_data(split, X_np, y_np)

            # Train model
            model = self._create_model()

            if self.early_stopping > 0:
                # Use last portion of training for validation
                val_size = int(len(X_train) * 0.1)
                X_val = X_train[-val_size:]
                y_val = y_train[-val_size:]
                X_train = X_train[:-val_size]
                y_train = y_train[:-val_size]

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    # early_stopping_rounds=self.early_stopping,
                    verbose=False,
                )
            else:
                model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = None
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]

            metrics = self._compute_metrics(y_test, y_pred, y_proba)
            fold_metrics.append(metrics)

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                fold_importance.append(model.feature_importances_)

            log.info(f"Fold {split.fold}: Accuracy={metrics.accuracy:.4f}, F1={metrics.f1:.4f}")

        # Train final model on all data
        self.model = self._create_model()
        self.model.fit(X_np, y_np)

        # Average metrics
        avg_metrics = ModelMetrics(
            accuracy=np.mean([m.accuracy for m in fold_metrics]),
            precision=np.mean([m.precision for m in fold_metrics]),
            recall=np.mean([m.recall for m in fold_metrics]),
            f1=np.mean([m.f1 for m in fold_metrics]),
            auc_roc=np.mean([m.auc_roc for m in fold_metrics]),
            win_rate=np.mean([m.win_rate for m in fold_metrics]),
        )

        # Average feature importance
        importance = {}
        if fold_importance:
            avg_imp = np.mean(fold_importance, axis=0)
            importance = dict(zip(feature_names, avg_imp))

        training_time = time.time() - start_time

        self._training_result = TrainingResult(
            model=self.model,
            model_type=self.model_type,
            metrics=avg_metrics,
            feature_importance=importance,
            fold_metrics=fold_metrics,
            params=self.params,
            training_time=training_time,
        )

        log.info(f"Training complete: {training_time:.2f}s, Avg Accuracy={avg_metrics.accuracy:.4f}")

        return self._training_result

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_np = X.to_numpy()

        # Handle NaN
        nan_mask = np.any(np.isnan(X_np), axis=1)
        predictions = np.zeros(len(X_np))
        predictions[nan_mask] = np.nan
        predictions[~nan_mask] = self.model.predict(X_np[~nan_mask])

        return predictions

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """Generate probability predictions (classification only)."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions")

        X_np = X.to_numpy()

        # Handle NaN
        nan_mask = np.any(np.isnan(X_np), axis=1)
        probas = np.zeros(len(X_np))
        probas[nan_mask] = np.nan
        probas[~nan_mask] = self.model.predict_proba(X_np[~nan_mask])[:, 1]

        return probas

    def save(self, path: str) -> None:
        """Save trained model and metadata."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        # Create directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save model
        model_path = path + '.model'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save metadata
        meta = {
            'model_type': self.model_type,
            'task': self.task,
            'params': self.params,
            'metrics': {
                'accuracy': self._training_result.metrics.accuracy,
                'precision': self._training_result.metrics.precision,
                'recall': self._training_result.metrics.recall,
                'f1': self._training_result.metrics.f1,
                'auc_roc': self._training_result.metrics.auc_roc,
            } if self._training_result else {},
            'training_time': self._training_result.training_time if self._training_result else 0,
        }

        meta_path = path + '.meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        log.info(f"Saved model to {path}")

    def load(self, path: str) -> None:
        """Load trained model."""
        model_path = path + '.model'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        meta_path = path + '.meta.json'
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.model_type = meta.get('model_type', self.model_type)
            self.task = meta.get('task', self.task)
            self.params = meta.get('params', self.params)

        log.info(f"Loaded model from {path}")

    def get_feature_importance_df(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if self._training_result is None:
            raise ValueError("No training result. Train first.")

        importance = self._training_result.feature_importance
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])
        df = df.sort_values('importance', ascending=False)
        return df


class EnsembleTrainer:
    """
    Train ensemble of multiple models.
    Combines predictions using averaging or stacking.
    """

    def __init__(
        self,
        models: List[str] = ['xgboost', 'lightgbm'],
        task: str = 'classification',
        ensemble_method: str = 'average',
    ):
        """
        Args:
            models: List of model types to ensemble
            task: 'classification' or 'regression'
            ensemble_method: 'average', 'weighted', or 'stacking'
        """
        self.model_types = models
        self.task = task
        self.ensemble_method = ensemble_method
        self.trainers: List[ModelTrainer] = []
        self.weights: Optional[np.ndarray] = None

    def train(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        timestamps: Optional[List[datetime]] = None,
    ) -> Dict[str, TrainingResult]:
        """Train all models in the ensemble."""
        results = {}

        for model_type in self.model_types:
            log.info(f"Training {model_type}...")
            trainer = ModelTrainer(model_type=model_type, task=self.task)
            result = trainer.train(X, y, timestamps)
            self.trainers.append(trainer)
            results[model_type] = result

        # Compute weights based on validation performance
        if self.ensemble_method == 'weighted':
            scores = [r.metrics.f1 for r in results.values()]
            total = sum(scores)
            self.weights = np.array([s / total for s in scores])
        else:
            self.weights = np.ones(len(self.trainers)) / len(self.trainers)

        return results

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        predictions = []
        for trainer in self.trainers:
            if self.task == 'classification':
                pred = trainer.predict_proba(X)
            else:
                pred = trainer.predict(X)
            predictions.append(pred)

        # Weighted average
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        if self.task == 'classification':
            return (ensemble_pred > 0.5).astype(int)
        return ensemble_pred

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """Generate ensemble probability predictions."""
        predictions = []
        for trainer in self.trainers:
            pred = trainer.predict_proba(X)
            predictions.append(pred)

        predictions = np.array(predictions)
        return np.average(predictions, axis=0, weights=self.weights)
