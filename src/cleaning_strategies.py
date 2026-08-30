"""Cleaning Strategy Engine (Phase 7B).

Given an issue group's effective logical type, issue type, and the column's
current values, decides which remediation strategies are safe and
meaningful to offer -- it never decides *which* one to apply, and it never
touches a DataFrame. A human always chooses; ``src/cleaning_engine.py``
executes only the choice a human confirmed.

Design rules this engine follows throughout (see individual catalog
functions for how each applies):

- Every catalog always includes "Do not clean" (make no change) and, for
  missing-value strategies, "Keep as NULL" (an explicit, logged decision
  that NULL is acceptable) -- both are decisions, not fixes; see
  ``src/models.py:CleaningDecisionLogEntry``.
- A statistic (mean/median/mode) is only offered when it can actually be
  computed from the column's own current data -- never a fabricated number.
- Row-level/reference-like types (Identifier, Email, Phone, URL) never get
  bulk statistical strategies (no Mean/Median/Mode/0), regardless of the
  column's raw pandas dtype -- only NULL, a per-row custom value, or no
  action.
- Outliers and negative values always offer "keep" as a first-class choice
  and are never associated with a strategy that implies they are
  automatically wrong.
- Mixed/Unknown effective types get no type-specific strategies at all --
  the caller should prompt the user to confirm a type on the Column
  Profiling page first (Phase 7A's type override).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from src.config import IQR_OUTLIER_MULTIPLIER
from src.profiler import date_values, numeric_values
from src.schema_inference import LogicalType

# --- Strategy option model ---------------------------------------------------------


@dataclass(frozen=True)
class CleaningStrategyOption:
    """One candidate remediation offered to the user for an issue group."""

    strategy_id: str
    title: str
    description: str
    requires_custom_value: bool = False
    requires_row_scope: bool = False


@dataclass
class StrategyCatalogResult:
    """Available strategies for one issue group, plus any stats to display."""

    options: list[CleaningStrategyOption] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    note: Optional[str] = None


# --- Shared strategy option constants -----------------------------------------------

KEEP_AS_NULL = CleaningStrategyOption("keep_as_null", "Keep as NULL", "Leave the missing value(s) as a proper missing value -- an explicit, logged decision.")
DO_NOT_CLEAN = CleaningStrategyOption("do_not_clean", "Do not clean", "Make no change at this time.")
REPLACE_MEAN = CleaningStrategyOption("replace_mean", "Replace with Mean", "Fill with the column's current mean value.")
REPLACE_MEDIAN = CleaningStrategyOption("replace_median", "Replace with Median", "Fill/replace with the column's current median value.")
REPLACE_MODE = CleaningStrategyOption("replace_mode", "Replace with Mode", "Fill with the column's current most frequent value.")
REPLACE_ZERO = CleaningStrategyOption("replace_zero", "Replace with 0", "Fill/replace with 0.")
CUSTOM_NUMERIC = CleaningStrategyOption("custom_numeric", "Enter custom numeric value", "Fill/replace with a value you provide.", requires_custom_value=True)
REPLACE_UNKNOWN_LABEL = CleaningStrategyOption("replace_unknown_label", 'Replace with "Unknown"', 'Fill with the literal placeholder "Unknown".')
REPLACE_NOT_PROVIDED = CleaningStrategyOption("replace_not_provided", 'Replace with "Not Provided"', 'Fill with the literal placeholder "Not Provided".')
CUSTOM_TEXT = CleaningStrategyOption("custom_text", "Enter custom value", "Fill with a value you provide.", requires_custom_value=True)
REPLACE_TRUE = CleaningStrategyOption("replace_true", "Replace with True", "Fill with True.")
REPLACE_FALSE = CleaningStrategyOption("replace_false", "Replace with False", "Fill with False.")
CUSTOM_DATE = CleaningStrategyOption("custom_date", "Enter custom date", "Fill with a date you provide.", requires_custom_value=True)
CUSTOM_PER_ROW = CleaningStrategyOption(
    "custom_per_row", "Enter custom value for selected row(s)",
    "Provide a value individually for each affected row -- never one bulk value for every row, to avoid creating duplicate identifiers/contacts.",
    requires_custom_value=True, requires_row_scope=True,
)
KEEP_OUTLIER = CleaningStrategyOption("keep_outlier", "Keep outlier(s)", "Leave the value(s) unchanged -- a statistical outlier is not necessarily an error.")
SET_NULL = CleaningStrategyOption("set_null", "Set to NULL", "Replace the value(s) with a proper missing value.")
IQR_CAP = CleaningStrategyOption("iqr_cap", "Cap to IQR boundary (Winsorize)", "Cap the value at the nearest IQR lower/upper fence.")
NEGATIVE_VALID = CleaningStrategyOption("negative_valid", "Negative values are valid -- keep them", "Confirms these negative values are legitimate for this column; no change is made.")
NEGATIVE_TREAT_INVALID = CleaningStrategyOption("negative_treat_invalid", "Treat selected negative values as invalid", "Opens remediation options (NULL, 0, median, or a custom value) for the negative values you select.")
KEEP_VARIANTS = CleaningStrategyOption("keep_variants", "Keep all variants", "Leave every spelling/formatting variant as-is.")
STANDARDIZE_MOST_FREQUENT = CleaningStrategyOption("standardize_most_frequent", "Standardize to the most frequent variant", "Rewrite the other variants to match whichever spelling already appears most often.")
SELECT_CANONICAL = CleaningStrategyOption("select_canonical", "Select the canonical value", "Choose which existing variant should become the standard spelling.")
CUSTOM_CANONICAL = CleaningStrategyOption("custom_canonical", "Enter a custom canonical value", "Provide a brand-new standard spelling for every variant in this group.", requires_custom_value=True)

_NUMERIC_LIKE_TYPES = {LogicalType.INTEGER, LogicalType.DECIMAL, LogicalType.NUMERIC_TEXT, LogicalType.CURRENCY}
_ROW_LEVEL_ONLY_TYPES = {LogicalType.IDENTIFIER, LogicalType.EMAIL, LogicalType.PHONE, LogicalType.URL}
_DATE_LIKE_TYPES = {LogicalType.DATE, LogicalType.DATETIME, LogicalType.DATE_TEXT}
_UNCERTAIN_TYPES = {LogicalType.MIXED, LogicalType.UNKNOWN}

TYPE_CONFIRMATION_NOTE = "Please confirm the logical column type in Column Profiling before applying type-specific cleaning."


# --- Statistic computation (also used by the UI for display) ------------------------


def compute_numeric_stats(values: pd.Series) -> dict[str, Optional[float]]:
    """Compute mean/median/mode/IQR bounds from a column's current non-null values.

    Args:
        values: Already-coerced numeric values (see src.profiler.numeric_values).

    Returns:
        A dict with keys "mean", "median", "mode", "q1", "q3", "iqr",
        "lower_bound", "upper_bound" -- each None if it cannot be safely
        computed (e.g. no data at all).
    """
    clean = values.dropna()
    if clean.empty:
        return {"mean": None, "median": None, "mode": None, "q1": None, "q3": None, "iqr": None, "lower_bound": None, "upper_bound": None}

    mode_series = clean.mode()
    mode_value = round(float(mode_series.iloc[0]), 4) if not mode_series.empty else None
    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    iqr = q3 - q1

    return {
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "mode": mode_value,
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(q1 - IQR_OUTLIER_MULTIPLIER * iqr, 4),
        "upper_bound": round(q3 + IQR_OUTLIER_MULTIPLIER * iqr, 4),
    }


def compute_categorical_mode(values: pd.Series) -> Optional[str]:
    """Most frequent non-blank value in a categorical/text column, if any."""
    non_null = values.dropna().astype(str)
    non_null = non_null[non_null.str.strip() != ""]
    if non_null.empty:
        return None
    return str(non_null.mode().iloc[0])


def compute_boolean_mode(values: pd.Series) -> Optional[bool]:
    """Most frequent value in a boolean column, if any."""
    non_null = values.dropna()
    if non_null.empty:
        return None
    return bool(non_null.mode().iloc[0])


# --- Missing-value strategy catalog --------------------------------------------------


def get_missing_value_strategies(effective_type: LogicalType, series: pd.Series) -> StrategyCatalogResult:
    """Decide which missing-value remediation strategies apply to a column.

    Args:
        effective_type: The column's effective logical type (post type-override).
        series: The column's current values (including nulls) -- used only
            to compute display statistics and to decide whether a
            statistic-based strategy is safely offerable.

    Returns:
        A StrategyCatalogResult. ``options`` is empty with ``note`` set for
        Mixed/Unknown types.
    """
    if effective_type in _UNCERTAIN_TYPES:
        return StrategyCatalogResult(options=[], note=TYPE_CONFIRMATION_NOTE)

    if effective_type in _NUMERIC_LIKE_TYPES:
        numeric = numeric_values(series, effective_type)
        stats = compute_numeric_stats(numeric)
        options = [KEEP_AS_NULL]
        if stats["mean"] is not None:
            options.append(REPLACE_MEAN)
        if stats["median"] is not None:
            options.append(REPLACE_MEDIAN)
        if stats["mode"] is not None:
            options.append(REPLACE_MODE)
        options += [REPLACE_ZERO, CUSTOM_NUMERIC, DO_NOT_CLEAN]
        return StrategyCatalogResult(options=options, stats={k: v for k, v in stats.items() if k in ("mean", "median", "mode")})

    if effective_type == LogicalType.CATEGORICAL:
        mode = compute_categorical_mode(series)
        options = [KEEP_AS_NULL]
        if mode is not None:
            options.append(REPLACE_MODE)
        options += [REPLACE_UNKNOWN_LABEL, CUSTOM_TEXT, DO_NOT_CLEAN]
        return StrategyCatalogResult(options=options, stats={"mode": mode})

    if effective_type == LogicalType.FREE_TEXT:
        return StrategyCatalogResult(options=[KEEP_AS_NULL, REPLACE_UNKNOWN_LABEL, REPLACE_NOT_PROVIDED, CUSTOM_TEXT, DO_NOT_CLEAN])

    if effective_type == LogicalType.BOOLEAN:
        mode = compute_boolean_mode(series)
        options = [KEEP_AS_NULL, REPLACE_TRUE, REPLACE_FALSE]
        if mode is not None:
            options.append(REPLACE_MODE)
        options.append(DO_NOT_CLEAN)
        return StrategyCatalogResult(options=options, stats={"mode": mode})

    if effective_type in _DATE_LIKE_TYPES:
        # Never auto-compute an average/median date -- only a user-provided one.
        return StrategyCatalogResult(options=[KEEP_AS_NULL, CUSTOM_DATE, DO_NOT_CLEAN])

    if effective_type in _ROW_LEVEL_ONLY_TYPES:
        return StrategyCatalogResult(options=[KEEP_AS_NULL, CUSTOM_PER_ROW, DO_NOT_CLEAN])

    return StrategyCatalogResult(options=[KEEP_AS_NULL, DO_NOT_CLEAN])


# --- Outlier strategy catalog ---------------------------------------------------------


def get_outlier_strategies(effective_type: LogicalType, series: pd.Series) -> StrategyCatalogResult:
    """Decide which outlier remediation strategies apply, with IQR stats for display.

    Never returns a strategy list implying the outlier must be changed --
    "Keep outlier(s)" and "Do not clean" are always present.
    """
    if effective_type not in _NUMERIC_LIKE_TYPES:
        return StrategyCatalogResult(options=[], note="Outlier review only applies to numeric/currency columns.")

    numeric = numeric_values(series, effective_type)
    stats = compute_numeric_stats(numeric)
    options = [KEEP_OUTLIER, SET_NULL]
    if stats["median"] is not None:
        options.append(REPLACE_MEDIAN)
    if stats["lower_bound"] is not None and stats["upper_bound"] is not None:
        options.append(IQR_CAP)
    options += [CUSTOM_NUMERIC, DO_NOT_CLEAN]
    return StrategyCatalogResult(options=options, stats=stats)


# --- Negative-value review -------------------------------------------------------------


def get_negative_value_decision_options() -> list[CleaningStrategyOption]:
    """The first-stage decision offered for a column's negative values."""
    return [NEGATIVE_VALID, NEGATIVE_TREAT_INVALID]


def get_negative_value_treatment_strategies(effective_type: LogicalType, series: pd.Series) -> StrategyCatalogResult:
    """Second-stage remediation strategies, once negatives are marked invalid."""
    if effective_type not in _NUMERIC_LIKE_TYPES:
        return StrategyCatalogResult(options=[])
    numeric = numeric_values(series, effective_type)
    stats = compute_numeric_stats(numeric)
    options = [SET_NULL, REPLACE_ZERO]
    if stats["median"] is not None:
        options.append(REPLACE_MEDIAN)
    options.append(CUSTOM_NUMERIC)
    return StrategyCatalogResult(options=options, stats=stats)


# --- Categorical variant strategies ----------------------------------------------------


def get_categorical_variant_strategies() -> list[CleaningStrategyOption]:
    """Strategies for a cluster of values that may represent the same category.

    Never includes an "automatically merge" option -- merging always
    requires an explicit canonical choice from the user.
    """
    return [KEEP_VARIANTS, STANDARDIZE_MOST_FREQUENT, SELECT_CANONICAL, CUSTOM_CANONICAL, DO_NOT_CLEAN]


# --- Type-conversion eligibility (Date/Numeric stored as text) --------------------------


def is_safe_full_column_conversion(series: pd.Series, effective_type: LogicalType) -> bool:
    """Whether every non-null value in a text-stored column converts safely.

    Reused by the UI to decide whether to mention type conversion at all for
    a given issue group; the actual conversion itself is still performed by
    src.cleaning_engine's existing safe fixes (unchanged from Phase 4/6).
    """
    non_null = series.dropna()
    if non_null.empty:
        return False
    if effective_type == LogicalType.DATE_TEXT:
        return bool(date_values(series, effective_type).notna().all())
    if effective_type == LogicalType.NUMERIC_TEXT:
        return bool(numeric_values(series, effective_type).notna().all())
    return False
