# ml_pipeline/feature_processor.py
"""
Feature processing and selection for forex ML pipeline.
Handles preprocessing, correlation analysis, importance ranking, and domain grouping.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import pandas as pd

log = logging.getLogger(__name__)


# Feature domain groupings based on the tier system
FEATURE_DOMAINS = {
    'session': [
        'is_asian', 'is_london', 'is_ny', 'is_killzone',
        'asian_london_overlap', 'london_ny_overlap', 'weekday', 'is_friday', 'is_monday'
    ],
    'stop_hunt': [
        'upper_stop_hunt', 'lower_stop_hunt', 'high_sweep', 'low_sweep',
        'upper_wick_z', 'lower_wick_z', 'volume_z'
    ],
    'sentiment': [
        'retail_long_pct', 'contrarian_weight', 'rsi_14', 'crowd_overbought',
        'crowd_oversold', 'fade_longs_signal', 'fade_shorts_signal',
        'bull_trap', 'bear_trap', 'fomo_long_detected', 'fomo_short_detected',
        'long_capitulation', 'short_capitulation', 'contrarian_short_score', 'contrarian_long_score'
    ],
    'psychology': [
        'fear_spike', 'greed_spike', 'fear_index', 'greed_index',
        'market_mood', 'capitulation_signal', 'euphoria_signal',
        'herd_strength', 'overreaction', 'underreaction', 'market_confusion',
        'loss_aversion', 'profit_taking'
    ],
    'fibonacci': [
        'is_swing_high', 'is_swing_low', 'bars_since_high', 'bars_since_low',
        'high_low_time_ratio', 'near_fib_time', 'time_acceleration'
    ],
    'lunar': [
        'moon_phase', 'is_full_moon', 'is_new_moon', 'is_waxing',
        'lunar_volatility_factor'
    ],
    'chaos': [
        'returns', 'fractal_dimension', 'phase_velocity', 'phase_expansion',
        'price_recurrence', 'local_lyapunov', 'trajectory_divergence',
        'attractor_distance', 'attractor_strength', 'entropy_ratio',
        'approximate_entropy', 'predictability'
    ],
    'social_physics': [
        'price_velocity', 'price_acceleration', 'market_mass', 'kinetic_energy',
        'market_momentum', 'market_force', 'market_temperature', 'market_entropy',
        'volume_pressure', 'price_compression', 'crowd_coherence', 'potential_energy'
    ],
    'game_theory': [
        'selling_pressure', 'buying_pressure', 'coordination_index',
        'nash_equilibrium_strength', 'bulls_won', 'bears_won',
        'minimax_ratio', 'pareto_optimal', 'spread_proxy', 'liquidity_quality'
    ],
    'quantum': [
        'wave_particle_ratio', 'particle_probability', 'wave_probability',
        'uncertainty_product', 'precision_index', 'bull_state_prob', 'bear_state_prob',
        'superposition_coherence', 'entanglement_strength', 'decoherence',
        'upside_tunnel_prob', 'downside_tunnel_prob'
    ],
    'neural': [
        'delta_trend', 'theta_trend', 'alpha_trend', 'beta_trend', 'gamma_trend',
        'delta_theta_coherence', 'alpha_beta_coherence', 'overall_coherence',
        'delta_dominant', 'gamma_dominant', 'neural_complexity'
    ],
    'hft': [
        'tick_momentum', 'micro_trend', 'order_flow_imbalance', 'price_efficiency',
        'volume_acceleration', 'liquidity_grab_high', 'liquidity_grab_low',
        'mm_accumulation', 'mm_distribution'
    ],
    'orderflow': [
        'est_buy_volume', 'est_sell_volume', 'net_order_flow', 'buy_sell_ratio',
        'cumulative_delta', 'bearish_delta_divergence', 'bullish_delta_divergence',
        'buying_absorption', 'selling_absorption'
    ],
    'intracandle': [
        'body_position', 'body_range_ratio', 'upper_wick_pct', 'lower_wick_pct',
        'wick_ratio', 'close_position', 'momentum_continuation'
    ],
    'astro': [
        'mercury_pos', 'venus_pos', 'mars_pos', 'jupiter_pos', 'saturn_pos',
        'major_aspect_count', 'retrograde_count', 'eclipse_proximity'
    ],
}


@dataclass
class FeatureStats:
    """Statistics about a feature."""
    name: str
    domain: str
    mean: float
    std: float
    min: float
    max: float
    null_pct: float
    inf_pct: float
    importance: float = 0.0
    correlation_with_target: float = 0.0


@dataclass
class FeatureSelectionResult:
    """Results of feature selection."""
    selected_features: List[str]
    removed_correlated: List[str]
    removed_low_importance: List[str]
    feature_stats: Dict[str, FeatureStats]
    importance_ranking: List[Tuple[str, float]]


class FeatureProcessor:
    """
    Process and select features for ML training.

    Features:
    - Handle NaN/Inf values
    - Scale features (standardization or normalization)
    - Remove highly correlated features
    - Rank features by importance (SHAP or correlation)
    - Domain-based feature grouping
    """

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        importance_threshold: float = 0.01,
        scaling_method: str = 'standard',
        handle_inf: str = 'clip',
        handle_nan: str = 'forward_fill',
    ):
        """
        Args:
            correlation_threshold: Remove features with correlation above this
            importance_threshold: Remove features with importance below this
            scaling_method: 'standard' (z-score), 'minmax', or 'robust'
            handle_inf: 'clip' (to max/min) or 'nan' (convert to NaN)
            handle_nan: 'forward_fill', 'mean', 'median', 'zero', or 'drop'
        """
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.scaling_method = scaling_method
        self.handle_inf = handle_inf
        self.handle_nan = handle_nan

        self._scalers: Dict[str, Dict] = {}
        self._feature_stats: Dict[str, FeatureStats] = {}
        self._selected_features: List[str] = []

    def get_feature_domain(self, feature_name: str) -> str:
        """Get the domain (tier) for a feature."""
        for domain, features in FEATURE_DOMAINS.items():
            # Check if feature name starts with any known feature
            for known_feature in features:
                if feature_name.startswith(known_feature) or feature_name == known_feature:
                    return domain

        # Check for pattern matches
        if 'fib_time' in feature_name or 'fib_ratio' in feature_name:
            return 'fibonacci'
        if 'phase_' in feature_name:
            return 'chaos'
        if 'wave_' in feature_name or 'quantum' in feature_name:
            return 'quantum'

        return 'unknown'

    def compute_feature_stats(
        self,
        X: pl.DataFrame,
        y: Optional[pl.Series] = None,
    ) -> Dict[str, FeatureStats]:
        """Compute statistics for all features."""
        stats = {}

        for col in X.columns:
            values = X[col].to_numpy().astype(float)

            # Handle inf values for stats calculation
            finite_values = values[np.isfinite(values)]

            if len(finite_values) == 0:
                continue

            null_pct = np.isnan(values).mean() * 100
            inf_pct = np.isinf(values).mean() * 100

            # Compute correlation with target if provided
            corr_with_target = 0.0
            if y is not None:
                y_arr = y.to_numpy().astype(float)
                valid = np.isfinite(values) & np.isfinite(y_arr)
                if valid.sum() > 10:
                    corr_with_target = np.corrcoef(values[valid], y_arr[valid])[0, 1]
                    if np.isnan(corr_with_target):
                        corr_with_target = 0.0

            stats[col] = FeatureStats(
                name=col,
                domain=self.get_feature_domain(col),
                mean=float(np.mean(finite_values)),
                std=float(np.std(finite_values)),
                min=float(np.min(finite_values)),
                max=float(np.max(finite_values)),
                null_pct=null_pct,
                inf_pct=inf_pct,
                correlation_with_target=corr_with_target,
            )

        self._feature_stats = stats
        return stats

    def handle_missing_values(self, X: pl.DataFrame) -> pl.DataFrame:
        """Handle NaN and Inf values."""
        result = X.clone()

        for col in result.columns:
            col_data = result[col]
            dtype = col_data.dtype

            # Skip non-numeric columns (is_finite only works on floats)
            if dtype not in [pl.Float32, pl.Float64]:
                # For non-float columns, just handle nulls
                if dtype in [pl.Int32, pl.Int64, pl.Int16, pl.Int8]:
                    if self.handle_nan == 'zero':
                        result = result.with_columns(col_data.fill_null(0).alias(col))
                    elif self.handle_nan == 'forward_fill':
                        result = result.with_columns(col_data.forward_fill().alias(col))
                elif dtype == pl.Boolean:
                    result = result.with_columns(col_data.fill_null(False).alias(col))
                continue

            # Handle inf for float columns
            if self.handle_inf == 'clip':
                finite_vals = col_data.filter(col_data.is_finite())
                if len(finite_vals) > 0:
                    max_val = finite_vals.max()
                    min_val = finite_vals.min()
                    result = result.with_columns(
                        pl.when(col_data == float('inf')).then(max_val)
                        .when(col_data == float('-inf')).then(min_val)
                        .otherwise(col_data)
                        .alias(col)
                    )
            elif self.handle_inf == 'nan':
                result = result.with_columns(
                    pl.when(col_data.is_infinite()).then(None)
                    .otherwise(col_data)
                    .alias(col)
                )

            # Handle NaN for float columns
            col_data = result[col]
            if self.handle_nan == 'forward_fill':
                result = result.with_columns(col_data.forward_fill().alias(col))
            elif self.handle_nan == 'mean':
                mean_val = col_data.mean()
                result = result.with_columns(col_data.fill_null(mean_val).alias(col))
            elif self.handle_nan == 'median':
                median_val = col_data.median()
                result = result.with_columns(col_data.fill_null(median_val).alias(col))
            elif self.handle_nan == 'zero':
                result = result.with_columns(col_data.fill_null(0.0).alias(col))

        return result

    def scale_features(
        self,
        X: pl.DataFrame,
        fit: bool = True,
    ) -> pl.DataFrame:
        """Scale features using the specified method."""
        result = X.clone()

        for col in result.columns:
            col_data = result[col].to_numpy().astype(float)

            if fit:
                if self.scaling_method == 'standard':
                    mean = np.nanmean(col_data)
                    std = np.nanstd(col_data)
                    std = std if std > 1e-10 else 1.0
                    self._scalers[col] = {'mean': mean, 'std': std}
                elif self.scaling_method == 'minmax':
                    min_val = np.nanmin(col_data)
                    max_val = np.nanmax(col_data)
                    range_val = max_val - min_val
                    range_val = range_val if range_val > 1e-10 else 1.0
                    self._scalers[col] = {'min': min_val, 'range': range_val}
                elif self.scaling_method == 'robust':
                    median = np.nanmedian(col_data)
                    q1 = np.nanpercentile(col_data, 25)
                    q3 = np.nanpercentile(col_data, 75)
                    iqr = q3 - q1
                    iqr = iqr if iqr > 1e-10 else 1.0
                    self._scalers[col] = {'median': median, 'iqr': iqr}

            # Apply scaling
            scaler = self._scalers.get(col, {})
            if self.scaling_method == 'standard':
                scaled = (col_data - scaler.get('mean', 0)) / scaler.get('std', 1)
            elif self.scaling_method == 'minmax':
                scaled = (col_data - scaler.get('min', 0)) / scaler.get('range', 1)
            elif self.scaling_method == 'robust':
                scaled = (col_data - scaler.get('median', 0)) / scaler.get('iqr', 1)
            else:
                scaled = col_data

            result = result.with_columns(pl.Series(col, scaled))

        return result

    def remove_correlated_features(
        self,
        X: pl.DataFrame,
        y: Optional[pl.Series] = None,
    ) -> Tuple[pl.DataFrame, List[str]]:
        """
        Remove highly correlated features.
        When two features are correlated, keep the one with higher target correlation.
        """
        df_pd = X.to_pandas()
        corr_matrix = df_pd.corr().abs()

        # Get target correlations
        target_corr = {}
        if y is not None:
            y_arr = y.to_numpy()
            for col in df_pd.columns:
                valid = np.isfinite(df_pd[col]) & np.isfinite(y_arr)
                if valid.sum() > 10:
                    tc = abs(np.corrcoef(df_pd[col][valid], y_arr[valid])[0, 1])
                    target_corr[col] = tc if not np.isnan(tc) else 0
                else:
                    target_corr[col] = 0

        # Find correlated pairs and remove the less important one
        removed = []
        cols_to_keep = list(df_pd.columns)

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]

                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    if col_i in cols_to_keep and col_j in cols_to_keep:
                        # Remove the one with lower target correlation
                        if target_corr.get(col_i, 0) >= target_corr.get(col_j, 0):
                            cols_to_keep.remove(col_j)
                            removed.append(col_j)
                        else:
                            cols_to_keep.remove(col_i)
                            removed.append(col_i)

        result = X.select(cols_to_keep)
        log.info(f"Removed {len(removed)} correlated features (threshold={self.correlation_threshold})")

        return result, removed

    def compute_feature_importance(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        method: str = 'xgboost',
    ) -> List[Tuple[str, float]]:
        """
        Compute feature importance using specified method.

        Args:
            X: Feature DataFrame
            y: Target Series
            method: 'xgboost', 'lightgbm', 'correlation', or 'shap'

        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        # Remove NaN rows
        valid = np.all(np.isfinite(X_np), axis=1) & np.isfinite(y_np)
        X_np = X_np[valid]
        y_np = y_np[valid]

        if method == 'correlation':
            # Simple correlation-based importance
            importance = []
            for i, col in enumerate(X.columns):
                corr = abs(np.corrcoef(X_np[:, i], y_np)[0, 1])
                importance.append((col, corr if not np.isnan(corr) else 0))

        elif method == 'xgboost':
            try:
                import xgboost as xgb

                # Train quick model
                dtrain = xgb.DMatrix(X_np, label=y_np)
                params = {
                    'objective': 'binary:logistic' if len(np.unique(y_np)) == 2 else 'reg:squarederror',
                    'max_depth': 4,
                    'learning_rate': 0.1,
                    'n_jobs': -1,
                    'verbosity': 0,
                }
                model = xgb.train(params, dtrain, num_boost_round=50)
                imp = model.get_score(importance_type='gain')

                # Map feature indices to names
                importance = [(X.columns[int(k[1:])], v) for k, v in imp.items() if k.startswith('f')]

                # Add zero importance for unused features
                used_features = set(pair[0] for pair in importance)
                for col in X.columns:
                    if col not in used_features:
                        importance.append((col, 0.0))

            except ImportError:
                log.warning("XGBoost not installed, falling back to correlation method")
                return self.compute_feature_importance(X, y, method='correlation')

        elif method == 'lightgbm':
            try:
                import lightgbm as lgb

                is_binary = len(np.unique(y_np)) == 2
                model = lgb.LGBMClassifier() if is_binary else lgb.LGBMRegressor()
                model.set_params(n_estimators=50, max_depth=4, verbosity=-1)
                model.fit(X_np, y_np)

                importance = list(zip(X.columns, model.feature_importances_))

            except ImportError:
                log.warning("LightGBM not installed, falling back to correlation method")
                return self.compute_feature_importance(X, y, method='correlation')

        elif method == 'shap':
            try:
                import shap
                import xgboost as xgb

                # Train model
                is_binary = len(np.unique(y_np)) == 2
                model = xgb.XGBClassifier() if is_binary else xgb.XGBRegressor()
                model.set_params(n_estimators=50, max_depth=4, verbosity=0)
                model.fit(X_np, y_np)

                # Compute SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_np[:1000])  # Sample for speed

                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if is_binary else shap_values[0]

                mean_shap = np.abs(shap_values).mean(axis=0)
                importance = list(zip(X.columns, mean_shap))

            except ImportError:
                log.warning("SHAP not installed, falling back to XGBoost method")
                return self.compute_feature_importance(X, y, method='xgboost')

        else:
            raise ValueError(f"Unknown importance method: {method}")

        # Sort by importance (descending)
        importance.sort(key=lambda x: x[1], reverse=True)

        # Update feature stats
        for name, imp in importance:
            if name in self._feature_stats:
                self._feature_stats[name].importance = imp

        return importance

    def select_features(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        importance_method: str = 'xgboost',
        top_n: Optional[int] = None,
        domains: Optional[List[str]] = None,
    ) -> FeatureSelectionResult:
        """
        Full feature selection pipeline.

        Args:
            X: Feature DataFrame
            y: Target Series
            importance_method: Method for computing importance
            top_n: Keep only top N features (None for all above threshold)
            domains: Only consider features from these domains (None for all)

        Returns:
            FeatureSelectionResult with selected features and statistics
        """
        log.info(f"Starting feature selection: {len(X.columns)} features")

        # Compute initial stats
        self.compute_feature_stats(X, y)

        # Filter by domain if specified
        if domains:
            domain_features = []
            for col in X.columns:
                if self.get_feature_domain(col) in domains:
                    domain_features.append(col)
            X = X.select(domain_features)
            log.info(f"Filtered to {len(domain_features)} features from domains: {domains}")

        # Handle missing values
        X = self.handle_missing_values(X)

        # Remove correlated features
        X, removed_correlated = self.remove_correlated_features(X, y)

        # Compute importance
        importance = self.compute_feature_importance(X, y, method=importance_method)

        # Filter by importance threshold
        removed_low_importance = []
        if self.importance_threshold > 0:
            max_importance = max(imp for _, imp in importance)
            threshold = max_importance * self.importance_threshold
            selected = [(name, imp) for name, imp in importance if imp >= threshold]
            removed_low_importance = [name for name, imp in importance if imp < threshold]
        else:
            selected = importance

        # Take top N if specified
        if top_n and top_n < len(selected):
            removed_low_importance.extend([name for name, _ in selected[top_n:]])
            selected = selected[:top_n]

        selected_features = [name for name, _ in selected]
        self._selected_features = selected_features

        log.info(f"Selected {len(selected_features)} features")

        return FeatureSelectionResult(
            selected_features=selected_features,
            removed_correlated=removed_correlated,
            removed_low_importance=removed_low_importance,
            feature_stats=self._feature_stats,
            importance_ranking=importance,
        )

    def transform(self, X: pl.DataFrame, fit: bool = False) -> pl.DataFrame:
        """
        Apply full preprocessing pipeline.

        Args:
            X: Feature DataFrame
            fit: Whether to fit scalers (True for training, False for inference)

        Returns:
            Processed DataFrame with selected features
        """
        # Handle missing values
        X = self.handle_missing_values(X)

        # Scale features
        X = self.scale_features(X, fit=fit)

        # Select only previously selected features
        if self._selected_features:
            available = [c for c in self._selected_features if c in X.columns]
            X = X.select(available)

        return X

    def get_domain_features(self, domain: str) -> List[str]:
        """Get all selected features from a specific domain."""
        return [
            f for f in self._selected_features
            if self.get_feature_domain(f) == domain
        ]

    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all features and their statistics."""
        rows = []
        for name, stats in self._feature_stats.items():
            rows.append({
                'feature': name,
                'domain': stats.domain,
                'importance': stats.importance,
                'target_corr': stats.correlation_with_target,
                'mean': stats.mean,
                'std': stats.std,
                'null_pct': stats.null_pct,
                'selected': name in self._selected_features,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('importance', ascending=False)
        return df
