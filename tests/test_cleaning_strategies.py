"""Tests for src.cleaning_strategies -- the Cleaning Strategy Engine (Phase 7B)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.cleaning_strategies import (
    TYPE_CONFIRMATION_NOTE,
    compute_boolean_mode,
    compute_categorical_mode,
    compute_numeric_stats,
    get_categorical_variant_strategies,
    get_missing_value_strategies,
    get_negative_value_decision_options,
    get_negative_value_treatment_strategies,
    get_outlier_strategies,
    is_safe_full_column_conversion,
)
from src.schema_inference import LogicalType


def _titles(options) -> set[str]:
    return {o.title for o in options}


# --- Numeric / Currency missing-value strategies ------------------------------------------


@pytest.mark.parametrize("logical_type", [LogicalType.INTEGER, LogicalType.DECIMAL, LogicalType.CURRENCY, LogicalType.NUMERIC_TEXT])
def test_numeric_missing_value_strategies_include_all_expected_options(logical_type) -> None:
    series = pd.Series([100.0, 200.0, 300.0, 300.0, None])
    result = get_missing_value_strategies(logical_type, series)

    titles = _titles(result.options)
    assert titles == {
        "Keep as NULL", "Replace with Mean", "Replace with Median", "Replace with Mode",
        "Replace with 0", "Enter custom numeric value", "Do not clean",
    }
    assert result.stats["mean"] == pytest.approx(225.0)
    assert result.stats["median"] == pytest.approx(250.0)
    assert result.stats["mode"] == pytest.approx(300.0)


def test_numeric_missing_value_strategies_omit_uncomputable_statistics() -> None:
    # No non-null values at all -- mean/median/mode cannot be safely computed.
    series = pd.Series([None, None, None])
    result = get_missing_value_strategies(LogicalType.DECIMAL, series)

    titles = _titles(result.options)
    assert "Replace with Mean" not in titles
    assert "Replace with Median" not in titles
    assert "Replace with Mode" not in titles
    assert titles == {"Keep as NULL", "Replace with 0", "Enter custom numeric value", "Do not clean"}


def test_compute_numeric_stats_are_accurate() -> None:
    values = pd.Series([10.0, 20.0, 30.0, 40.0])
    stats = compute_numeric_stats(values)
    assert stats["mean"] == 25.0
    assert stats["median"] == 25.0
    assert stats["q1"] == pytest.approx(17.5)
    assert stats["q3"] == pytest.approx(32.5)


# --- Categorical missing-value strategies ---------------------------------------------------


def test_categorical_missing_value_strategies() -> None:
    series = pd.Series(["Engineering", "Sales", "Engineering", None])
    result = get_missing_value_strategies(LogicalType.CATEGORICAL, series)

    assert _titles(result.options) == {"Keep as NULL", "Replace with Mode", 'Replace with "Unknown"', "Enter custom value", "Do not clean"}
    assert result.stats["mode"] == "Engineering"


def test_categorical_missing_value_strategies_without_mode() -> None:
    series = pd.Series([None, None])
    result = get_missing_value_strategies(LogicalType.CATEGORICAL, series)
    assert "Replace with Mode" not in _titles(result.options)


# --- Text missing-value strategies -----------------------------------------------------------


def test_text_missing_value_strategies() -> None:
    series = pd.Series(["a review", None, "another review"])
    result = get_missing_value_strategies(LogicalType.FREE_TEXT, series)
    assert _titles(result.options) == {"Keep as NULL", 'Replace with "Unknown"', 'Replace with "Not Provided"', "Enter custom value", "Do not clean"}


# --- Boolean missing-value strategies ----------------------------------------------------------


def test_boolean_missing_value_strategies() -> None:
    series = pd.Series([True, False, True, None])
    result = get_missing_value_strategies(LogicalType.BOOLEAN, series)
    assert _titles(result.options) == {"Keep as NULL", "Replace with True", "Replace with False", "Replace with Mode", "Do not clean"}
    assert compute_boolean_mode(series) is True


def test_boolean_missing_value_strategies_without_mode() -> None:
    series = pd.Series([None, None])
    result = get_missing_value_strategies(LogicalType.BOOLEAN, series)
    assert "Replace with Mode" not in _titles(result.options)


# --- Date missing-value strategies: never an average date --------------------------------------


@pytest.mark.parametrize("logical_type", [LogicalType.DATE, LogicalType.DATETIME, LogicalType.DATE_TEXT])
def test_date_missing_value_strategies_never_offer_average(logical_type) -> None:
    series = pd.Series(["2023-01-01", None, "2023-06-15"])
    result = get_missing_value_strategies(logical_type, series)

    titles = _titles(result.options)
    assert titles == {"Keep as NULL", "Enter custom date", "Do not clean"}
    assert not any("mean" in t.lower() or "average" in t.lower() or "median" in t.lower() for t in titles)


# --- Identifier / Email / Phone / URL: no statistical strategies ever --------------------------


@pytest.mark.parametrize("logical_type", [LogicalType.IDENTIFIER, LogicalType.EMAIL, LogicalType.PHONE, LogicalType.URL])
def test_row_level_only_types_never_offer_statistical_strategies(logical_type) -> None:
    # Numeric-dtype-looking values (e.g. phone digits) must not unlock
    # numeric statistics just because pandas happens to store them as numbers.
    series = pd.Series([501234561, 501234562, None, 501234564])
    result = get_missing_value_strategies(logical_type, series)

    titles = _titles(result.options)
    assert titles == {"Keep as NULL", "Enter custom value for selected row(s)", "Do not clean"}
    assert "Replace with Mean" not in titles
    assert "Replace with Median" not in titles
    assert "Replace with Mode" not in titles
    assert "Replace with 0" not in titles

    custom_option = next(o for o in result.options if o.strategy_id == "custom_per_row")
    assert custom_option.requires_row_scope is True
    assert custom_option.requires_custom_value is True


# --- Mixed / Unknown: no type-specific strategies -----------------------------------------------


@pytest.mark.parametrize("logical_type", [LogicalType.MIXED, LogicalType.UNKNOWN])
def test_uncertain_types_offer_no_strategies(logical_type) -> None:
    result = get_missing_value_strategies(logical_type, pd.Series([1, "a", None]))
    assert result.options == []
    assert result.note == TYPE_CONFIRMATION_NOTE


# --- Outlier strategies ---------------------------------------------------------------------------


def test_outlier_strategies_always_include_keep_and_do_not_clean() -> None:
    series = pd.Series([10, 11, 9, 10, 12, 11, 9, 10, 1000])
    result = get_outlier_strategies(LogicalType.DECIMAL, series)

    titles = _titles(result.options)
    assert "Keep outlier(s)" in titles
    assert "Do not clean" in titles
    assert titles == {"Keep outlier(s)", "Set to NULL", "Replace with Median", "Cap to IQR boundary (Winsorize)", "Enter custom numeric value", "Do not clean"}
    assert result.stats["median"] is not None
    assert result.stats["lower_bound"] is not None
    assert result.stats["upper_bound"] is not None


def test_outlier_strategies_not_offered_for_non_numeric_types() -> None:
    result = get_outlier_strategies(LogicalType.CATEGORICAL, pd.Series(["a", "b"]))
    assert result.options == []


# --- Negative value review ---------------------------------------------------------------------------


def test_negative_value_decision_options_default_to_valid_and_invalid_choice() -> None:
    options = get_negative_value_decision_options()
    titles = {o.title for o in options}
    assert titles == {"Negative values are valid -- keep them", "Treat selected negative values as invalid"}


def test_negative_value_treatment_strategies_after_marking_invalid() -> None:
    series = pd.Series([-100.0, -50.0, 200.0, 300.0])
    result = get_negative_value_treatment_strategies(LogicalType.DECIMAL, series)
    titles = _titles(result.options)
    assert titles == {"Set to NULL", "Replace with 0", "Replace with Median", "Enter custom numeric value"}


# --- Categorical variant strategies ------------------------------------------------------------------


def test_categorical_variant_strategies_never_auto_merge() -> None:
    options = get_categorical_variant_strategies()
    titles = {o.title for o in options}
    assert titles == {
        "Keep all variants", "Standardize to the most frequent variant",
        "Select the canonical value", "Enter a custom canonical value", "Do not clean",
    }
    # No option should imply an automatic fuzzy merge without an explicit human choice.
    assert all(o.strategy_id != "auto_merge" for o in options)


# --- Type-conversion safety gate (reused for Date/Numeric stored as text) --------------------------


def test_is_safe_full_column_conversion_true_when_all_values_convert() -> None:
    series = pd.Series([f"2023-01-{i:02d}" for i in range(1, 10)])
    assert is_safe_full_column_conversion(series, LogicalType.DATE_TEXT) is True


def test_is_safe_full_column_conversion_false_on_partial_failure() -> None:
    series = pd.Series([f"2023-01-{i:02d}" for i in range(1, 9)] + ["not-a-date"])
    assert is_safe_full_column_conversion(series, LogicalType.DATE_TEXT) is False


def test_is_safe_full_column_conversion_numeric() -> None:
    good = pd.Series([str(i) for i in range(10)])
    bad = pd.Series([str(i) for i in range(9)] + ["abc"])
    assert is_safe_full_column_conversion(good, LogicalType.NUMERIC_TEXT) is True
    assert is_safe_full_column_conversion(bad, LogicalType.NUMERIC_TEXT) is False
