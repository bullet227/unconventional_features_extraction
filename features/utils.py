"""
Utility functions for feature extraction.

This module provides helper functions used across all feature modules,
including expression handling, validation, and common transformations.
"""
from __future__ import annotations
from typing import Union, Any, List
import polars as pl


def ensure_expr(value: Any, alias: str = None) -> pl.Expr:
    """
    Ensure a value is a Polars expression.

    This helper prevents the common error:
        'float' object has no attribute 'alias'

    It converts scalars (int, float, bool, str) to pl.lit() expressions,
    while passing through existing expressions unchanged.

    Args:
        value: Any value - can be a Polars expression, scalar, or column name
        alias: Optional alias to apply to the resulting expression

    Returns:
        A Polars expression

    Examples:
        >>> ensure_expr(0.5)
        pl.lit(0.5)

        >>> ensure_expr(pl.col("close") / pl.col("open"))
        pl.col("close") / pl.col("open")

        >>> ensure_expr(42, alias="answer")
        pl.lit(42).alias("answer")
    """
    # Already an expression
    if isinstance(value, pl.Expr):
        return value.alias(alias) if alias else value

    # String could be a column reference
    if isinstance(value, str):
        # Treat as literal string, not column name
        # Use pl.col("name") explicitly for column references
        expr = pl.lit(value)
    elif isinstance(value, (int, float, bool)):
        expr = pl.lit(value)
    elif value is None:
        expr = pl.lit(None)
    else:
        # For other types, try to convert to literal
        try:
            expr = pl.lit(value)
        except Exception:
            raise TypeError(
                f"Cannot convert {type(value).__name__} to Polars expression. "
                f"Value: {value!r}"
            )

    return expr.alias(alias) if alias else expr


def safe_divide(
    numerator: Union[pl.Expr, float],
    denominator: Union[pl.Expr, float],
    default: float = 0.0,
    epsilon: float = 1e-10,
) -> pl.Expr:
    """
    Safely divide two expressions, handling division by zero.

    Args:
        numerator: The numerator expression or scalar
        denominator: The denominator expression or scalar
        default: Value to return when denominator is zero
        epsilon: Small value added to denominator to prevent exact zero

    Returns:
        Division result with zero-division protection
    """
    num = ensure_expr(numerator)
    denom = ensure_expr(denominator)

    return pl.when(denom.abs() < epsilon).then(pl.lit(default)).otherwise(num / denom)


def clip_expr(
    expr: Union[pl.Expr, float],
    min_val: float = None,
    max_val: float = None,
) -> pl.Expr:
    """
    Clip expression values to a range.

    Args:
        expr: The expression to clip
        min_val: Minimum value (None for no lower bound)
        max_val: Maximum value (None for no upper bound)

    Returns:
        Clipped expression
    """
    result = ensure_expr(expr)

    if min_val is not None:
        result = pl.when(result < min_val).then(pl.lit(min_val)).otherwise(result)

    if max_val is not None:
        result = pl.when(result > max_val).then(pl.lit(max_val)).otherwise(result)

    return result


def normalize_expr(
    expr: pl.Expr,
    window_size: int = 20,
    epsilon: float = 1e-10,
) -> pl.Expr:
    """
    Normalize an expression using rolling z-score.

    Args:
        expr: The expression to normalize
        window_size: Rolling window size for mean/std calculation
        epsilon: Small value to prevent division by zero

    Returns:
        Z-score normalized expression
    """
    rolling_mean = expr.rolling_mean(window_size=window_size)
    rolling_std = expr.rolling_std(window_size=window_size)

    return (expr - rolling_mean) / (rolling_std + epsilon)


def validate_ohlcv_columns(df: pl.DataFrame) -> None:
    """
    Validate that a DataFrame has required OHLCV columns.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required = {"time", "open", "high", "low", "close", "volume"}
    available = set(c.lower() for c in df.columns)

    # Check case-insensitive
    missing = required - available

    if missing:
        raise ValueError(
            f"Missing required OHLCV columns: {missing}. "
            f"Available columns: {df.columns}"
        )


def ensure_column_list(
    expressions: List[Any],
    prefix: str = "",
) -> List[pl.Expr]:
    """
    Ensure all items in a list are valid Polars expressions.

    This is useful when building with_columns() calls to prevent
    the 'float has no attribute alias' error.

    Args:
        expressions: List of values that should be expressions
        prefix: Optional prefix to add to any auto-generated aliases

    Returns:
        List of Polars expressions
    """
    result = []
    for i, expr in enumerate(expressions):
        if isinstance(expr, pl.Expr):
            result.append(expr)
        else:
            # Convert to literal with auto-generated alias
            alias = f"{prefix}const_{i}" if prefix else f"const_{i}"
            result.append(ensure_expr(expr, alias=alias))
    return result
