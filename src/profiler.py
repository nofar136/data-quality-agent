"""Column-level data profiling (Phase 2).

Builds one ColumnProfile per column: basic completeness/uniqueness stats
(from pandas alone) plus type-specific stats chosen according to the
inferred logical type (numeric stats, date range, or text length stats).

Dataset-level profiling (run IDs, an overall quality score, memory usage,
etc.) is introduced in later phases; a Phase 1 dataset overview already
covers row/column/missing/duplicate counts in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from src.config import IQR_OUTLIER_MULTIPLIER, MAX_EXAMPLE_VALUES
from src.schema_inference import LogicalType, infer_logical_type, normalize_column_name


@dataclass
class ColumnProfile:
    """Full statistical and type profile for a single column.

    ``logical_type``/``confidence``/``evidence`` are always the *inferred*
    values from schema_inference -- never mutated. ``effective_logical_type``
    is what downstream issue detection should actually use: the inferred
    type by default, or a user's override (see src/type_override.py), which
    never touches the raw data or the inferred fields themselves.
    """

    original_name: str
    normalized_name: str
    pandas_dtype: str
    logical_type: str
    confidence: float
    non_null_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_ratio: float
    duplicate_value_count: int
    example_values: list[str] = field(default_factory=list)
    effective_logical_type: str = ""
    type_overridden: bool = False
    user_selected_type: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    min_text_length: Optional[float] = None
    max_text_length: Optional[float] = None
    avg_text_length: Optional[float] = None
    outlier_count: Optional[int] = None
    evidence: dict[str, Any] = field(default_factory=dict)


_NUMERIC_LOGICAL_TYPES = {LogicalType.INTEGER, LogicalType.DECIMAL, LogicalType.NUMERIC_TEXT, LogicalType.CURRENCY}
_DATE_LOGICAL_TYPES = {LogicalType.DATE, LogicalType.DATETIME, LogicalType.DATE_TEXT}
_NO_TEXT_STATS_TYPES = _NUMERIC_LOGICAL_TYPES | _DATE_LOGICAL_TYPES | {LogicalType.EMPTY, LogicalType.BOOLEAN}


def numeric_values(series: pd.Series, logical_type: LogicalType) -> pd.Series:
    """Coerce a column's non-null values to numeric for stats purposes."""
    non_null = series.dropna()
    if logical_type in (LogicalType.INTEGER, LogicalType.DECIMAL):
        return pd.to_numeric(non_null, errors="coerce")

    cleaned = non_null.astype(str).str.strip()
    is_paren_negative = cleaned.str.match(r"^\(.*\)$")
    cleaned = cleaned.str.strip("()")
    cleaned = cleaned.str.replace(r"[$€£¥₪,%\s]", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric.where(~is_paren_negative, -numeric)


def date_values(series: pd.Series, logical_type: LogicalType) -> pd.Series:
    """Coerce a column's non-null values to datetimes for min/max purposes."""
    non_null = series.dropna()
    if logical_type in (LogicalType.DATE, LogicalType.DATETIME):
        return non_null
    # format="mixed" parses each value with its own inferred format, rather
    # than locking the whole column to one format and turning genuinely
    # valid but differently-formatted dates into NaT.
    return pd.to_datetime(non_null.astype(str).str.strip(), errors="coerce", format="mixed")


def _count_iqr_outliers(numeric: pd.Series) -> Optional[int]:
    """Count values outside [Q1 - k*IQR, Q3 + k*IQR]; None if too few points."""
    values = numeric.dropna()
    if len(values) < 4:
        return None
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower = q1 - IQR_OUTLIER_MULTIPLIER * iqr
    upper = q3 + IQR_OUTLIER_MULTIPLIER * iqr
    return int(((values < lower) | (values > upper)).sum())


def profile_column(series: pd.Series, column_name: str) -> ColumnProfile:
    """Build a complete statistical + type profile for a single column.

    Args:
        series: The column's values.
        column_name: The original column name (used for display and as a
            weak signal in type inference).

    Returns:
        A ColumnProfile with completeness/uniqueness stats plus
        type-appropriate numeric, date, or text-length stats.
    """
    total = len(series)
    non_null = series.dropna()
    missing_count = total - len(non_null)
    missing_pct = round((missing_count / total * 100), 2) if total else 0.0
    unique_count = int(non_null.nunique())
    unique_ratio = round(unique_count / len(non_null), 3) if len(non_null) else 0.0
    duplicate_value_count = max(len(non_null) - unique_count, 0)

    inference = infer_logical_type(series, column_name)
    logical_type = inference.logical_type

    example_values = [str(v) for v in pd.Series(non_null.unique()[:MAX_EXAMPLE_VALUES])]

    profile = ColumnProfile(
        original_name=column_name,
        normalized_name=normalize_column_name(column_name),
        pandas_dtype=str(series.dtype),
        logical_type=logical_type.value,
        confidence=round(inference.confidence, 3),
        non_null_count=int(len(non_null)),
        missing_count=int(missing_count),
        missing_pct=missing_pct,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        duplicate_value_count=duplicate_value_count,
        example_values=example_values,
        evidence=inference.evidence,
        effective_logical_type=logical_type.value,
    )

    if logical_type in _NUMERIC_LOGICAL_TYPES:
        numeric = numeric_values(series, logical_type).dropna()
        if not numeric.empty:
            profile.min_value = float(numeric.min())
            profile.max_value = float(numeric.max())
            profile.mean = round(float(numeric.mean()), 4)
            profile.median = round(float(numeric.median()), 4)
            profile.std = round(float(numeric.std()), 4) if len(numeric) > 1 else 0.0
            profile.outlier_count = _count_iqr_outliers(numeric)

    elif logical_type in _DATE_LOGICAL_TYPES:
        dates = date_values(series, logical_type).dropna()
        if not dates.empty:
            profile.min_date = str(dates.min())
            profile.max_date = str(dates.max())

    elif logical_type not in _NO_TEXT_STATS_TYPES:
        str_values = non_null.astype(str)
        if not str_values.empty:
            lengths = str_values.str.len()
            profile.min_text_length = float(lengths.min())
            profile.max_text_length = float(lengths.max())
            profile.avg_text_length = round(float(lengths.mean()), 2)

    return profile


def profile_dataframe(df: pd.DataFrame) -> list[ColumnProfile]:
    """Build a ColumnProfile for every column in a DataFrame, in column order.

    Args:
        df: The dataset to profile.

    Returns:
        One ColumnProfile per column.
    """
    return [profile_column(df[col], str(col)) for col in df.columns]
